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
from utils.constraint_guidance import (
    get_preset_mask_nsga,
    get_target_constraint_from_entropy,
    load_constraint_guide,
    load_pareto_data,
)
from utils.entropy_conditioning import build_router_input
from utils.eval_flag import eval_stage2
from utils.lr_sched import adjust_learning_rate, adjust_learning_rate_by_step
from utils.set_wandb import set_wandb


def infinite_loader(data_loader):
    while True:
        for batch in data_loader:
            yield batch


def set_trainable_phase(model, phase):
    for name, param in model.named_parameters():
        if phase == "constraint_mlp":
            param.requires_grad = "constraint_mlp" in name
        elif phase == "router_warmup":
            param.requires_grad = ("router" in name) or ("constraint_mlp" in name)
        elif phase == "joint":
            param.requires_grad = True
        else:
            raise ValueError(phase)


def log_trainable_parameters(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name} is trainable")
        else:
            print(f"{name} is frozen")


def compute_stage2_losses(model, img, label, entropy_vectors, device, guide_rows, pareto_data, gen_id, criterion):
    router_input = build_router_input(entropy_vectors, device)
    predicted_constraint = model.predict_constraint(router_input)
    target_constraint, entropy_mean = get_target_constraint_from_entropy(entropy_vectors, guide_rows, device)

    model.configure_constraint(constraint=predicted_constraint, tau=1)
    preds, attn_mask, mlp_mask, embed_mask, depth_attn_mask, depth_mlp_mask, total_macs = model(img)
    total_macs = total_macs.unsqueeze(0)

    ce_loss = criterion(preds, label)
    constraint_loss = F.mse_loss(total_macs, target_constraint)
    guide_loss = F.mse_loss(predicted_constraint, target_constraint)

    label_mlp_mask, label_mha_mask, label_emb_mask, label_depth_mlp_mask, label_depth_attn_mask = get_preset_mask_nsga(
        gen_id,
        target_constraint,
        device,
        pareto_data,
    )
    label_mask_loss = (
        F.mse_loss(attn_mask, label_mha_mask)
        + F.mse_loss(mlp_mask, label_mlp_mask)
        + F.mse_loss(embed_mask, label_emb_mask)
        + F.mse_loss(depth_mlp_mask, label_depth_mlp_mask)
        + F.mse_loss(depth_attn_mask, label_depth_attn_mask)
    )

    return {
        "preds": preds,
        "attn_mask": attn_mask,
        "mlp_mask": mlp_mask,
        "embed_mask": embed_mask,
        "depth_attn_mask": depth_attn_mask,
        "depth_mlp_mask": depth_mlp_mask,
        "total_macs": total_macs,
        "predicted_constraint": predicted_constraint,
        "target_constraint": target_constraint,
        "entropy_mean": entropy_mean,
        "ce_loss": ce_loss,
        "constraint_loss": constraint_loss,
        "guide_loss": guide_loss,
        "label_mask_loss": label_mask_loss,
    }


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

    if args.smoothing > 0.0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    param_groups = [
        {
            "params": [p for n, p in model.named_parameters() if "router" not in n and "constraint_mlp" not in n],
            "lr": args.max_lr,
        },
        {
            "params": [p for n, p in model.named_parameters() if "router" in n and "constraint_mlp" not in n],
            "lr": 1e-3,
            "lr_scale": 1e3,
        },
        {
            "params": [p for n, p in model.named_parameters() if "constraint_mlp" in n],
            "lr": 1e-3,
            "lr_scale": 1e3,
        },
    ]
    optimizer = torch.optim.AdamW(param_groups)

    folder_path = os.path.join("logs_weight", "stage_2_entropy", args.dataset)
    os.makedirs(folder_path, exist_ok=True)
    time = datetime.now().strftime("%b%d_%H-%M-%S")
    log_dir = os.path.join(folder_path, time)
    os.makedirs(log_dir)

    weight_path = os.path.join(log_dir, "weight")
    os.makedirs(weight_path)

    set_wandb(args, name="EA-ViT_stage2_entropy_constraint")

    pareto_data = load_pareto_data(args.nsga_path)
    guide_rows = load_constraint_guide(args.constraint_guide_path)

    train_stream = infinite_loader(trainDataLoader)

    set_trainable_phase(model, "constraint_mlp")
    log_trainable_parameters(model)
    optimizer.zero_grad()
    with tqdm(total=args.constraint_mlp_warmup_steps, mininterval=0.3) as pbar:
        for warmup_step in range(args.constraint_mlp_warmup_steps):
            cur_lr = adjust_learning_rate_by_step(optimizer, min(warmup_step, max(0, args.total_steps - 1)), args)
            guide_loss_total = 0.0
            predicted_constraint_total = 0.0
            target_constraint_total = 0.0
            entropy_mean_total = 0.0

            for _ in range(args.grad_accum_steps):
                img, label, entropy_vectors, _ = next(train_stream)
                router_input = build_router_input(entropy_vectors, args.device)
                target_constraint, entropy_mean = get_target_constraint_from_entropy(entropy_vectors, guide_rows, args.device)
                predicted_constraint = model.predict_constraint(router_input)
                guide_loss = F.mse_loss(predicted_constraint, target_constraint)

                (guide_loss / args.grad_accum_steps).backward()

                guide_loss_total += guide_loss.item()
                predicted_constraint_total += predicted_constraint.item()
                target_constraint_total += target_constraint.item()
                entropy_mean_total += entropy_mean

            optimizer.step()
            optimizer.zero_grad()

            if warmup_step % 10 == 0:
                wandb.log(
                    {
                        "constraint_mlp_warmup/guide_loss": guide_loss_total / args.grad_accum_steps,
                        "constraint_mlp_warmup/predicted_constraint": predicted_constraint_total / args.grad_accum_steps,
                        "constraint_mlp_warmup/target_constraint": target_constraint_total / args.grad_accum_steps,
                        "constraint_mlp_warmup/entropy_mean": entropy_mean_total / args.grad_accum_steps,
                        "constraint_mlp_warmup/lr": cur_lr,
                    },
                    step=warmup_step,
                )

            pbar.set_description(f"constraint mlp warmup {warmup_step + 1}/{args.constraint_mlp_warmup_steps}")
            pbar.set_postfix(loss=f"{guide_loss_total / args.grad_accum_steps:.4f}", lr=f"{cur_lr:.2e}")
            pbar.update(1)

    set_trainable_phase(model, "router_warmup")
    log_trainable_parameters(model)
    optimizer.zero_grad()
    with tqdm(total=args.total_steps, mininterval=0.3) as pbar:
        for global_step in range(args.total_steps):
            cur_lr = adjust_learning_rate_by_step(optimizer, global_step, args)

            loss_total = 0.0
            ce_total = 0.0
            constraint_total = 0.0
            label_mask_total = 0.0
            guide_total = 0.0
            macs_total = 0.0
            predicted_constraint_total = 0.0
            target_constraint_total = 0.0
            entropy_mean_total = 0.0

            for _ in range(args.grad_accum_steps):
                img, label, entropy_vectors, _ = next(train_stream)
                img = img.to(args.device)
                label = label.to(args.device)

                outputs = compute_stage2_losses(
                    model,
                    img,
                    label,
                    entropy_vectors,
                    args.device,
                    guide_rows,
                    pareto_data,
                    args.gen_id,
                    criterion,
                )

                guide_weight = max(0.0, 1.0 - global_step / max(1, args.total_steps))
                loss = (
                    outputs["ce_loss"]
                    + outputs["constraint_loss"] * 20
                    + outputs["label_mask_loss"] * 20
                    + outputs["guide_loss"] * guide_weight
                )
                (loss / args.grad_accum_steps).backward()

                loss_total += loss.item()
                ce_total += outputs["ce_loss"].item()
                constraint_total += outputs["constraint_loss"].item()
                label_mask_total += outputs["label_mask_loss"].item()
                guide_total += outputs["guide_loss"].item()
                macs_total += outputs["total_macs"].item()
                predicted_constraint_total += outputs["predicted_constraint"].item()
                target_constraint_total += outputs["target_constraint"].item()
                entropy_mean_total += outputs["entropy_mean"]

            optimizer.step()
            optimizer.zero_grad()

            if global_step % 10 == 0:
                wandb.log(
                    {
                        "stage2_warmup/total": loss_total / args.grad_accum_steps,
                        "stage2_warmup/ce": ce_total / args.grad_accum_steps,
                        "stage2_warmup/constraint": constraint_total / args.grad_accum_steps,
                        "stage2_warmup/label_mask": label_mask_total / args.grad_accum_steps,
                        "stage2_warmup/guide": guide_total / args.grad_accum_steps,
                        "stage2_warmup/macs": macs_total / args.grad_accum_steps,
                        "stage2_warmup/predicted_constraint": predicted_constraint_total / args.grad_accum_steps,
                        "stage2_warmup/target_constraint": target_constraint_total / args.grad_accum_steps,
                        "stage2_warmup/entropy_mean": entropy_mean_total / args.grad_accum_steps,
                        "lr": cur_lr,
                    },
                    step=global_step,
                )

            pbar.set_description(f"router warmup {global_step + 1}/{args.total_steps}")
            pbar.set_postfix(loss=f"{loss_total / args.grad_accum_steps:.4f}", lr=f"{cur_lr:.2e}")
            pbar.update(1)

    set_trainable_phase(model, "joint")
    log_trainable_parameters(model)

    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch + 1, args)

        with tqdm(total=len(trainDataLoader), postfix=dict, mininterval=0.3) as pbar:
            pbar.set_description(f"train Epoch {epoch + 1}/{args.epochs}")

            wandb.log({"Epoch": epoch + 1, "lr/vit learning_rate": optimizer.param_groups[0]["lr"]})
            wandb.log({"Epoch": epoch + 1, "lr/router learning_rate": optimizer.param_groups[1]["lr"]})
            wandb.log({"Epoch": epoch + 1, "lr/constraint mlp learning_rate": optimizer.param_groups[2]["lr"]})

            model.train()
            optimizer.zero_grad()

            total_loss = 0.0
            total_ce_loss = 0.0
            total_constraint_loss = 0.0
            total_label_mask_loss = 0.0
            total_guide_loss = 0.0

            total_attn_mask = 0.0
            total_mlp_mask = 0.0
            total_embed_mask = 0.0
            total_depth_mlp_mask = 0.0
            total_depth_attn_mask = 0.0

            for batch_idx, (img, label, entropy_vectors, _) in enumerate(trainDataLoader):
                img = img.to(args.device)
                label = label.to(args.device)

                outputs = compute_stage2_losses(
                    model,
                    img,
                    label,
                    entropy_vectors,
                    args.device,
                    guide_rows,
                    pareto_data,
                    args.gen_id,
                    criterion,
                )

                guide_weight = max(0.0, 1.0 - epoch / max(1, args.epochs))
                loss = (
                    outputs["ce_loss"]
                    + outputs["constraint_loss"] * 20
                    + outputs["label_mask_loss"] * 20 * (1 - epoch / args.epochs)
                    + outputs["guide_loss"] * guide_weight
                )
                (loss / args.grad_accum_steps).backward()

                should_step = ((batch_idx + 1) % args.grad_accum_steps == 0) or ((batch_idx + 1) == len(trainDataLoader))
                if should_step:
                    optimizer.step()
                    optimizer.zero_grad()

                attn_mask_mean = torch.mean(outputs["attn_mask"])
                mlp_mask_mean = torch.mean(outputs["mlp_mask"])
                embed_mask_mean = torch.mean(outputs["embed_mask"])
                depth_mlp_mask_mean = torch.mean(outputs["depth_mlp_mask"])
                depth_attn_mask_mean = torch.mean(outputs["depth_attn_mask"])

                if batch_idx % 10 == 0:
                    wandb.log({"train_batch_loss/batch cross entropy loss": outputs["ce_loss"]})
                    wandb.log({"train_batch_loss/batch constraint loss": outputs["constraint_loss"]})
                    wandb.log({"train_batch_loss/batch label mask loss": outputs["label_mask_loss"]})
                    wandb.log({"train_batch_loss/batch guide loss": outputs["guide_loss"]})
                    wandb.log({"train_batch_loss/train Batch Loss": loss.item()})
                    wandb.log({"train_batch_signal/entropy_mean": outputs["entropy_mean"]})
                    wandb.log({"train_batch_signal/predicted_constraint": outputs["predicted_constraint"].item()})
                    wandb.log({"train_batch_signal/target_constraint": outputs["target_constraint"].item()})
                    wandb.log({"train_batch_signal/routed_macs": outputs["total_macs"].item()})

                total_loss += loss.item()
                total_ce_loss += outputs["ce_loss"].item()
                total_constraint_loss += outputs["constraint_loss"].item()
                total_label_mask_loss += outputs["label_mask_loss"].item()
                total_guide_loss += outputs["guide_loss"].item()

                total_attn_mask += attn_mask_mean.item()
                total_mlp_mask += mlp_mask_mean.item()
                total_embed_mask += embed_mask_mean.item()
                total_depth_mlp_mask += depth_mlp_mask_mean.item()
                total_depth_attn_mask += depth_attn_mask_mean.item()

                pbar.set_postfix(**{"loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]})
                pbar.update(1)

            epoch_loss = total_loss / len(trainDataLoader)
            epoch_ce_loss = total_ce_loss / len(trainDataLoader)
            epoch_constraint_loss = total_constraint_loss / len(trainDataLoader)
            epoch_label_mask_loss = total_label_mask_loss / len(trainDataLoader)
            epoch_guide_loss = total_guide_loss / len(trainDataLoader)

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
            wandb.log({"Epoch": epoch + 1, "train_epoch_loss/Train epoch guide loss": epoch_guide_loss})

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
                flag="dynamic",
                guide_rows=guide_rows,
                pareto_data=pareto_data,
                device=args.device,
            )

        torch.save(model.state_dict(), os.path.join(weight_path, "stage2_constraint_guided.pth"))


if __name__ == "__main__":
    os.environ["WANDB_MODE"] = "offline"
    arguments = get_args_parser()
    train(arguments)
