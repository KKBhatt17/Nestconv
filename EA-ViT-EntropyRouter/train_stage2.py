import os
from datetime import datetime

import torch
import torch.nn.functional as F
from timm.loss import LabelSmoothingCrossEntropy
from tqdm import tqdm
import wandb

from config import get_args_parser
from dataloader.entropy_image_datasets import build_entropy_image_dataset, create_entropy_dataloader
from models.model_stage2 import EAViTStage2, ModifiedBlock
from utils.entropy_conditioning import (
    build_router_input,
    default_entropy_lookup_path,
    encoding_to_mask_tensors,
    entropy_score_from_vectors,
    load_entropy_lookup,
    select_lookup_entry,
)
from utils.eval_flag import eval_stage2
from utils.lr_sched import adjust_learning_rate, adjust_learning_rate_by_step
from utils.set_wandb import set_wandb


def resolve_lookup_target(entropy_vectors, device, lookup_rows):
    entropy_mean = entropy_score_from_vectors(entropy_vectors)
    lookup_entry = select_lookup_entry(lookup_rows, entropy_mean)
    label_mlp_mask, label_mha_mask, label_emb_mask, label_depth_mlp_mask, label_depth_attn_mask = encoding_to_mask_tensors(
        lookup_entry["encoding"],
        device,
    )
    target_macs = torch.tensor([lookup_entry["macs"]], device=device, dtype=torch.float32)
    return (
        label_mlp_mask,
        label_mha_mask,
        label_emb_mask,
        label_depth_mlp_mask,
        label_depth_attn_mask,
        target_macs,
        entropy_mean,
    )


def infinite_loader(data_loader):
    while True:
        for batch in data_loader:
            yield batch


def train(args):
    if str(args.device).startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))
    dataset_train, dataset_val, nb_classes = build_entropy_image_dataset(args)

    trainDataLoader = create_entropy_dataloader(args, dataset_train, shuffle_batches=True)
    valDataLoader = create_entropy_dataloader(args, dataset_val, shuffle_batches=False)

    router_input_dim = (args.input_size // args.entropy_patch_size) ** 2

    model = EAViTStage2(
        embed_dim=768,
        depth=12,
        mlp_ratio=4,
        num_heads=12,
        num_classes=nb_classes,
        drop_path_rate=args.drop_path,
        qkv_bias=True,
        block=ModifiedBlock,
        router_input_dim=router_input_dim,
    )

    checkpoint = torch.load(args.stage1_checkpoint_path, map_location=args.device)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    model.to(args.device)

    for name, param in model.named_parameters():
        param.requires_grad = "router" in name

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name} is trainable")
        else:
            print(f"{name} is frozen")

    if args.smoothing > 0.0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    param_groups = [
        {"params": [p for n, p in model.named_parameters() if "router" not in n], "lr": args.max_lr},
        {"params": [p for n, p in model.named_parameters() if "router" in n], "lr": 1e-3, "lr_scale": 1e3},
    ]
    optimizer = torch.optim.AdamW(param_groups)

    folder_path = os.path.join("logs_weight", "stage_2_entropy", args.dataset)
    os.makedirs(folder_path, exist_ok=True)
    time = datetime.now().strftime("%b%d_%H-%M-%S")
    log_dir = os.path.join(folder_path, time)
    os.makedirs(log_dir)

    weight_path = os.path.join(log_dir, "weight")
    os.makedirs(weight_path)

    set_wandb(args, name="EA-ViT_stage2_entropy")

    lookup_path = args.entropy_lookup_path or default_entropy_lookup_path(args.nsga_path)
    lookup_rows = load_entropy_lookup(lookup_path)

    global_step = 0
    micro_step = 0
    warmup_loader = infinite_loader(trainDataLoader)

    optimizer.zero_grad()
    with tqdm(total=args.total_steps, mininterval=0.3) as pbar:
        for img, label, entropy_vectors, _ in warmup_loader:
            if global_step >= args.total_steps:
                break

            cur_lr = adjust_learning_rate_by_step(optimizer, global_step, args)

            img = img.to(args.device)
            label = label.to(args.device)
            router_input = build_router_input(entropy_vectors, args.device)

            optimizer.zero_grad()
            model.configure_router_input(router_input=router_input, tau=1)

            preds, attn_mask, mlp_mask, embed_mask, depth_attn_mask, depth_mlp_mask, total_macs = model(img)
            total_macs = total_macs.unsqueeze(0)

            ce_loss = criterion(preds, label)
            (
                label_mlp_mask,
                label_mha_mask,
                label_emb_mask,
                label_depth_mlp_mask,
                label_depth_attn_mask,
                target_macs,
                entropy_mean,
            ) = resolve_lookup_target(entropy_vectors, args.device, lookup_rows)

            constraint_loss = F.mse_loss(total_macs, target_macs)
            label_mask_loss = (
                F.mse_loss(attn_mask, label_mha_mask)
                + F.mse_loss(mlp_mask, label_mlp_mask)
                + F.mse_loss(embed_mask, label_emb_mask)
                + F.mse_loss(depth_mlp_mask, label_depth_mlp_mask)
                + F.mse_loss(depth_attn_mask, label_depth_attn_mask)
            )
            loss = ce_loss + constraint_loss * 20 + label_mask_loss * 20
            scaled_loss = loss / args.grad_accum_steps

            scaled_loss.backward()
            micro_step += 1

            if micro_step % args.grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            if global_step > 0 and micro_step % args.grad_accum_steps == 0 and global_step % 10 == 0:
                wandb.log(
                    {
                        "stage2_loss/total": loss.item(),
                        "stage2_loss/ce": ce_loss.item(),
                        "stage2_loss/constraint": constraint_loss.item(),
                        "stage2_loss/label_mask": label_mask_loss.item(),
                        "stage2_signal/entropy_mean": entropy_mean,
                        "stage2_signal/target_macs": target_macs.item(),
                        "lr": cur_lr,
                        "grad_accum_steps": args.grad_accum_steps,
                    },
                    step=global_step,
                )

            pbar.set_description(f"step {global_step}/{args.total_steps}")
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{cur_lr:.2e}")
            if micro_step % args.grad_accum_steps == 0:
                pbar.update(1)

    if micro_step % args.grad_accum_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    for _, param in model.named_parameters():
        param.requires_grad = True

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name} is trainable")
        else:
            print(f"{name} is frozen")

    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch + 1, args)

        with tqdm(total=len(trainDataLoader), postfix=dict, mininterval=0.3) as pbar:
            pbar.set_description(f"train Epoch {epoch + 1}/{args.epochs}")

            wandb.log({"Epoch": epoch + 1, "lr/vit learning_rate": optimizer.param_groups[0]["lr"]})
            wandb.log({"Epoch": epoch + 1, "lr/router learning_rate": optimizer.param_groups[1]["lr"]})

            model.train()

            total_loss = 0
            total_ce_loss = 0
            total_constraint_loss = 0
            total_label_mask_loss = 0

            total_attn_mask = 0
            total_mlp_mask = 0
            total_embed_mask = 0
            total_depth_mlp_mask = 0
            total_depth_attn_mask = 0

            optimizer.zero_grad()
            for batch_idx, (img, label, entropy_vectors, _) in enumerate(trainDataLoader):
                img = img.to(args.device)
                label = label.to(args.device)
                router_input = build_router_input(entropy_vectors, args.device)

                model.configure_router_input(router_input=router_input, tau=1)

                preds, attn_mask, mlp_mask, embed_mask, depth_attn_mask, depth_mlp_mask, total_macs = model(img)
                total_macs = total_macs.unsqueeze(0)

                ce_loss = criterion(preds, label)
                (
                    label_mlp_mask,
                    label_mha_mask,
                    label_emb_mask,
                    label_depth_mlp_mask,
                    label_depth_attn_mask,
                    target_macs,
                    entropy_mean,
                ) = resolve_lookup_target(entropy_vectors, args.device, lookup_rows)

                constraint_loss = F.mse_loss(total_macs, target_macs)
                label_mask_loss = (
                    F.mse_loss(attn_mask, label_mha_mask)
                    + F.mse_loss(mlp_mask, label_mlp_mask)
                    + F.mse_loss(embed_mask, label_emb_mask)
                    + F.mse_loss(depth_mlp_mask, label_depth_mlp_mask)
                    + F.mse_loss(depth_attn_mask, label_depth_attn_mask)
                )

                loss = ce_loss + constraint_loss * 20 + label_mask_loss * 20 * (1 - epoch / args.epochs)
                scaled_loss = loss / args.grad_accum_steps

                attn_mask_mean = torch.mean(attn_mask)
                mlp_mask_mean = torch.mean(mlp_mask)
                embed_mask_mean = torch.mean(embed_mask)
                depth_mlp_mask_mean = torch.mean(depth_mlp_mask)
                depth_attn_mask_mean = torch.mean(depth_attn_mask)

                if batch_idx % 10 == 0:
                    wandb.log({"train_batch_loss/batch cross entropy loss": ce_loss})
                    wandb.log({"train_batch_loss/batch constraint loss": constraint_loss})
                    wandb.log({"train_batch_loss/batch label mask loss": label_mask_loss})
                    wandb.log({"train_batch_loss/train Batch Loss": loss.item()})
                    wandb.log({"train_batch_signal/entropy_mean": entropy_mean})
                    wandb.log({"train_batch_signal/target_macs": target_macs.item()})

                total_loss += loss.item()
                total_ce_loss += ce_loss.item()
                total_constraint_loss += constraint_loss.item()
                total_label_mask_loss += label_mask_loss.item()

                total_attn_mask += attn_mask_mean.item()
                total_mlp_mask += mlp_mask_mean.item()
                total_embed_mask += embed_mask_mean.item()
                total_depth_mlp_mask += depth_mlp_mask_mean.item()
                total_depth_attn_mask += depth_attn_mask_mean.item()

                scaled_loss.backward()

                should_step = ((batch_idx + 1) % args.grad_accum_steps == 0) or ((batch_idx + 1) == len(trainDataLoader))
                if should_step:
                    optimizer.step()
                    optimizer.zero_grad()

                pbar.set_postfix(**{"loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]})
                pbar.update(1)

            epoch_loss = total_loss / len(trainDataLoader)
            epoch_ce_loss = total_ce_loss / len(trainDataLoader)
            epoch_constraint_loss = total_constraint_loss / len(trainDataLoader)
            epoch_label_mask_loss = total_label_mask_loss / len(trainDataLoader)

            epoch_attn_mask = total_attn_mask / len(trainDataLoader)
            epoch_mlp_mask = total_mlp_mask / len(trainDataLoader)
            epoch_embed_mask = total_embed_mask / len(trainDataLoader)
            epoch_depth_mlp_mask = total_depth_mlp_mask / len(trainDataLoader)
            epoch_depth_attn_mask = total_depth_attn_mask / len(trainDataLoader)

            print("train loss", epoch_loss)

            wandb.log({"Epoch": epoch + 1, "train_epoch_loss/Train epoch Loss": epoch_loss})
            wandb.log({"Epoch": epoch + 1, "train_epoch_loss/Train epoch cross entropy loss": epoch_ce_loss})
            wandb.log({"Epoch": epoch + 1, "train_epoch_loss/Train epoch constraint loss": epoch_constraint_loss})
            wandb.log({"Epoch": epoch + 1, "train_epoch_loss/Train epoch label mask loss": epoch_label_mask_loss})

            wandb.log({"Epoch": epoch + 1, "train_mask/Train attn mask": epoch_attn_mask})
            wandb.log({"Epoch": epoch + 1, "train_mask/Train mlp mask": epoch_mlp_mask})
            wandb.log({"Epoch": epoch + 1, "train_mask/Train embed mask": epoch_embed_mask})
            wandb.log({"Epoch": epoch + 1, "train_mask/Train depth mlp mask": epoch_depth_mlp_mask})
            wandb.log({"Epoch": epoch + 1, "train_mask/Train depth attn mask": epoch_depth_attn_mask})

        if epoch % 2 == 0:
            eval_stage2(
                model,
                valDataLoader,
                criterion,
                epoch,
                optimizer,
                args,
                flag="entropy",
                lookup_rows=lookup_rows,
                device=args.device,
            )

        torch.save(model.state_dict(), os.path.join(weight_path, "stage2_entropy.pth"))


if __name__ == "__main__":
    os.environ["WANDB_MODE"] = "offline"

    arguments = get_args_parser()
    train(arguments)
