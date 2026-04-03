import csv
import math
import os
from collections import defaultdict
from datetime import datetime
from itertools import cycle

import torch
import torch.nn.functional as F
import wandb
from timm.loss import LabelSmoothingCrossEntropy
from tqdm import tqdm

from config import get_args_parser
from dataloader.entropy_guided import build_entropy_guided_dataloaders
from models.model_stage2 import EAViTStage2, ModifiedBlock
from utils.eval_flag import eval_stage2_entropy
from utils.lr_sched import adjust_learning_rate
from utils.set_wandb import set_wandb


def load_pareto_data(file_name="pareto_front.csv"):
    data = defaultdict(list)
    with open(file_name, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            generation = int(row['Generation'])
            macs = float(row['MACs'])
            accuracy = float(row['Accuracy'])
            encoding = row['Encoding']
            data[generation].append((macs, accuracy, encoding))
    return data


def get_preset_mask_nsga(gen_id, constraint, device, data):
    generation = data[gen_id]
    sorted_individuals = sorted(generation, key=lambda x: x[0])

    macs = [ind[0] for ind in sorted_individuals]
    encodings = [ind[2] for ind in sorted_individuals]
    index = min(range(len(macs)), key=lambda i: abs(macs[i] - constraint))

    encoding = [int(digit) for digit in list(encodings[index])]

    embed_sum = int(sum(encoding[:12]))
    emb_mask = torch.cat((torch.ones(embed_sum, device=device), torch.zeros(12 - embed_sum, device=device)), dim=0)

    depth_attn_mask = torch.tensor([float(i) for i in encoding[12:24]], device=device)
    depth_mlp_mask = torch.tensor([float(i) for i in encoding[24:36]], device=device)

    mha_list = []
    mlp_list = []

    for i in range(12):
        attn_sum = int(sum(encoding[36 + i * 12: 36 + (i + 1) * 12]))
        mha_list.append(torch.cat((torch.ones(attn_sum, device=device), torch.zeros(12 - attn_sum, device=device)), dim=0))

    mha_mask = torch.stack(mha_list)

    for i in range(12):
        mlp_sum = int(sum(encoding[180 + i * 8: 180 + (i + 1) * 8]))
        mlp_list.append(torch.cat((torch.ones(mlp_sum, device=device), torch.zeros(8 - mlp_sum, device=device)), dim=0))

    mlp_mask = torch.stack(mlp_list)

    return mlp_mask, mha_mask, emb_mask, depth_mlp_mask, depth_attn_mask


class ConstraintGuide:
    def __init__(self, file_name):
        if not file_name:
            raise ValueError("constraint_guide_path must be provided for entropy-guided stage 2 training.")

        entropy_values = []
        constraint_values = []
        with open(file_name, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                entropy_values.append(float(row['EntropyMean']))
                constraint_values.append(float(row['MACs']))

        if not entropy_values:
            raise ValueError(f"No rows found in constraint guide file: {file_name}")

        pairs = sorted(zip(entropy_values, constraint_values), key=lambda x: x[0])
        self.entropy_values = torch.tensor([pair[0] for pair in pairs], dtype=torch.float32)
        self.constraint_values = torch.tensor([pair[1] for pair in pairs], dtype=torch.float32)

    def lookup(self, entropy_value, device):
        reference_entropy = self.entropy_values.to(device)
        reference_constraint = self.constraint_values.to(device)
        query = entropy_value.reshape(-1)

        if len(reference_entropy) == 1:
            return reference_constraint[:1].expand_as(query).view_as(entropy_value)

        insertion_index = torch.searchsorted(reference_entropy, query)
        clamped_index = insertion_index.clamp(1, len(reference_entropy) - 1)

        left_index = clamped_index - 1
        right_index = clamped_index

        left_entropy = reference_entropy[left_index]
        right_entropy = reference_entropy[right_index]
        left_constraint = reference_constraint[left_index]
        right_constraint = reference_constraint[right_index]

        denominator = (right_entropy - left_entropy).clamp_min(1e-8)
        interpolation_weight = (query - left_entropy) / denominator
        target_constraint = left_constraint + interpolation_weight * (right_constraint - left_constraint)

        target_constraint = torch.where(insertion_index == 0, reference_constraint[0], target_constraint)
        target_constraint = torch.where(
            insertion_index >= len(reference_entropy),
            reference_constraint[-1],
            target_constraint,
        )
        return target_constraint.view_as(entropy_value)


def adjust_learning_rate_by_total_steps(optimizer, step, total_steps, args):
    warmup_steps = min(args.warmup_steps, max(1, total_steps))
    if step < warmup_steps:
        lr = args.lr * step / max(1, warmup_steps)
    else:
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * progress))

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr * param_group.get("lr_scale", 1.0)
    return lr


def is_router_or_predictor(name):
    return ("router" in name) or ("constraint_predictor" in name)


def is_predictor(name):
    return "constraint_predictor" in name


def set_requires_grad(model, trainable_filter):
    for name, param in model.named_parameters():
        param.requires_grad = trainable_filter(name)


def print_trainable_state(model):
    for name, param in model.named_parameters():
        state = "trainable" if param.requires_grad else "frozen"
        print(f"{name} is {state}")


def maybe_step_optimizer(model, optimizer, pending_batches, args):
    if pending_batches == 0:
        return 0

    if args.clip_grad is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return 0


def get_batch_entropy(entropy_mean, device):
    return entropy_mean.to(device).float().mean().reshape(1)


def compute_stage2_losses(model, criterion, img, label, batch_entropy, predefined, guide, args, epoch_decay=None):
    predicted_constraint = model.predict_constraint(batch_entropy)
    model.configure_constraint(constraint=predicted_constraint, tau=1)

    preds, attn_mask, mlp_mask, embed_mask, depth_attn_mask, depth_mlp_mask, total_macs = model(img)
    total_macs = total_macs.unsqueeze(0)

    ce_loss = criterion(preds, label)
    constraint_loss = F.mse_loss(total_macs, predicted_constraint)

    target_constraint = guide.lookup(batch_entropy, img.device)
    guide_loss = F.mse_loss(predicted_constraint, target_constraint)

    label_mlp_mask, label_mha_mask, label_emb_mask, label_depth_mlp_mask, label_depth_attn_mask = get_preset_mask_nsga(
        args.gen_id,
        float(predicted_constraint.detach().item()),
        img.device,
        predefined,
    )

    label_mask_loss = (
        F.mse_loss(attn_mask, label_mha_mask)
        + F.mse_loss(mlp_mask, label_mlp_mask)
        + F.mse_loss(embed_mask, label_emb_mask)
        + F.mse_loss(depth_mlp_mask, label_depth_mlp_mask)
        + F.mse_loss(depth_attn_mask, label_depth_attn_mask)
    )

    guide_scale = 1.0 if epoch_decay is None else epoch_decay
    loss = ce_loss + constraint_loss * 20 + label_mask_loss * 20 * guide_scale + guide_loss * args.guide_loss_weight * guide_scale

    return {
        "loss": loss,
        "ce_loss": ce_loss,
        "constraint_loss": constraint_loss,
        "label_mask_loss": label_mask_loss,
        "guide_loss": guide_loss,
        "predicted_constraint": predicted_constraint.detach().item(),
        "total_macs": total_macs.detach().item(),
        "attn_mask_mean": torch.mean(attn_mask).detach().item(),
        "mlp_mask_mean": torch.mean(mlp_mask).detach().item(),
        "embed_mask_mean": torch.mean(embed_mask).detach().item(),
        "depth_mlp_mask_mean": torch.mean(depth_mlp_mask).detach().item(),
        "depth_attn_mask_mean": torch.mean(depth_attn_mask).detach().item(),
        "entropy_mean": batch_entropy.detach().item(),
    }


def run_constraint_predictor_init(model, trainDataLoader, optimizer, guide, args):
    if args.constraint_init_steps <= 0:
        return

    set_requires_grad(model, is_predictor)
    print_trainable_state(model)
    model.train()

    optimizer.zero_grad(set_to_none=True)
    pending_batches = 0

    with tqdm(total=args.constraint_init_steps, mininterval=0.3) as pbar:
        for global_step, (_, _, entropy_mean) in enumerate(cycle(trainDataLoader)):
            if global_step >= args.constraint_init_steps:
                break

            cur_lr = adjust_learning_rate_by_total_steps(optimizer, global_step, args.constraint_init_steps, args)
            batch_entropy = get_batch_entropy(entropy_mean, args.device)
            target_constraint = guide.lookup(batch_entropy, args.device)
            predicted_constraint = model.predict_constraint(batch_entropy)

            loss = F.mse_loss(predicted_constraint, target_constraint)
            (loss / args.accum_iter).backward()
            pending_batches += 1

            if pending_batches == args.accum_iter:
                pending_batches = maybe_step_optimizer(model, optimizer, pending_batches, args)

            if global_step % 10 == 0:
                wandb.log({
                    "constraint_init/loss": loss.item(),
                    "constraint_init/lr": cur_lr,
                    "constraint_init/entropy_mean": batch_entropy.item(),
                    "constraint_init/predicted_constraint": predicted_constraint.item(),
                    "constraint_init/target_constraint": target_constraint.item(),
                }, step=global_step)

            pbar.set_description(f"constraint init {global_step}/{args.constraint_init_steps}")
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{cur_lr:.2e}")
            pbar.update(1)

    maybe_step_optimizer(model, optimizer, pending_batches, args)


def run_router_warmup(model, trainDataLoader, optimizer, criterion, predefined, guide, args):
    set_requires_grad(model, is_router_or_predictor)
    print_trainable_state(model)
    model.train()

    optimizer.zero_grad(set_to_none=True)
    pending_batches = 0

    with tqdm(total=args.total_steps, mininterval=0.3) as pbar:
        for global_step, (img, label, entropy_mean) in enumerate(cycle(trainDataLoader)):
            if global_step >= args.total_steps:
                break

            cur_lr = adjust_learning_rate_by_total_steps(optimizer, global_step, args.total_steps, args)
            img = img.to(args.device)
            label = label.to(args.device)
            batch_entropy = get_batch_entropy(entropy_mean, args.device)

            metrics = compute_stage2_losses(model, criterion, img, label, batch_entropy, predefined, guide, args)
            (metrics["loss"] / args.accum_iter).backward()
            pending_batches += 1

            if pending_batches == args.accum_iter:
                pending_batches = maybe_step_optimizer(model, optimizer, pending_batches, args)

            if global_step % 10 == 0:
                wandb.log({
                    "warmup/loss": metrics["loss"].item(),
                    "warmup/ce": metrics["ce_loss"].item(),
                    "warmup/constraint": metrics["constraint_loss"].item(),
                    "warmup/label_mask": metrics["label_mask_loss"].item(),
                    "warmup/guide": metrics["guide_loss"].item(),
                    "warmup/lr": cur_lr,
                    "warmup/predicted_constraint": metrics["predicted_constraint"],
                    "warmup/batch_entropy": metrics["entropy_mean"],
                    "warmup/total_macs": metrics["total_macs"],
                }, step=global_step)

            pbar.set_description(f"router warmup {global_step}/{args.total_steps}")
            pbar.set_postfix(loss=f"{metrics['loss'].item():.4f}", lr=f"{cur_lr:.2e}")
            pbar.update(1)

    maybe_step_optimizer(model, optimizer, pending_batches, args)


def train(args):
    torch.cuda.set_device(args.device)
    trainDataLoader, valDataLoader, nb_classes = build_entropy_guided_dataloaders(args)

    model = EAViTStage2(
        embed_dim=768,
        depth=12,
        mlp_ratio=4,
        num_heads=12,
        num_classes=nb_classes,
        drop_path_rate=args.drop_path,
        qkv_bias=True,
        block=ModifiedBlock,
    )

    checkpoint = torch.load(args.stage1_checkpoint_path, map_location=args.device)
    model.load_state_dict(checkpoint, strict=False)
    model = model.to(args.device)

    if args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    param_groups = [
        {'params': [p for n, p in model.named_parameters() if not is_router_or_predictor(n)], 'lr': args.lr},
        {'params': [p for n, p in model.named_parameters() if is_router_or_predictor(n)], 'lr': args.lr * 1e3, 'lr_scale': 1e3},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    folder_path = os.path.join('logs_weight', 'stage_2_entropy_guided', args.dataset)
    os.makedirs(folder_path, exist_ok=True)
    time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(folder_path, time)
    os.makedirs(log_dir, exist_ok=True)

    weight_path = os.path.join(log_dir, 'weight')
    os.makedirs(weight_path, exist_ok=True)

    set_wandb(args, name='EA-ViT_stage2_entropy_guided')

    predefined = load_pareto_data(args.nsga_path)
    guide = ConstraintGuide(args.constraint_guide_path)

    run_constraint_predictor_init(model, trainDataLoader, optimizer, guide, args)
    run_router_warmup(model, trainDataLoader, optimizer, criterion, predefined, guide, args)

    set_requires_grad(model, lambda _name: True)
    print_trainable_state(model)

    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch + 1, args)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pending_batches = 0

        total_loss = 0.0
        total_ce_loss = 0.0
        total_constraint_loss = 0.0
        total_label_mask_loss = 0.0
        total_guide_loss = 0.0
        total_predicted_constraint = 0.0
        total_entropy_mean = 0.0
        total_macs = 0.0
        total_attn_mask = 0.0
        total_mlp_mask = 0.0
        total_embed_mask = 0.0
        total_depth_mlp_mask = 0.0
        total_depth_attn_mask = 0.0

        with tqdm(total=len(trainDataLoader), postfix=dict, mininterval=0.3) as pbar:
            pbar.set_description(f'train Epoch {epoch + 1}/{args.epochs}')

            wandb.log({"Epoch": epoch + 1, "lr/vit learning_rate": optimizer.param_groups[0]['lr']})
            wandb.log({"Epoch": epoch + 1, "lr/router learning_rate": optimizer.param_groups[1]['lr']})

            decay = max(0.0, 1 - epoch / args.epochs)

            for batch_idx, (img, label, entropy_mean) in enumerate(trainDataLoader):
                img = img.to(args.device)
                label = label.to(args.device)
                batch_entropy = get_batch_entropy(entropy_mean, args.device)

                metrics = compute_stage2_losses(
                    model,
                    criterion,
                    img,
                    label,
                    batch_entropy,
                    predefined,
                    guide,
                    args,
                    epoch_decay=decay,
                )
                (metrics["loss"] / args.accum_iter).backward()
                pending_batches += 1

                if pending_batches == args.accum_iter:
                    pending_batches = maybe_step_optimizer(model, optimizer, pending_batches, args)

                if batch_idx % 10 == 0:
                    wandb.log({
                        "train_batch_loss/total": metrics["loss"].item(),
                        "train_batch_loss/ce": metrics["ce_loss"].item(),
                        "train_batch_loss/constraint": metrics["constraint_loss"].item(),
                        "train_batch_loss/label_mask": metrics["label_mask_loss"].item(),
                        "train_batch_loss/guide": metrics["guide_loss"].item(),
                        "train_batch_loss/predicted_constraint": metrics["predicted_constraint"],
                        "train_batch_loss/batch_entropy": metrics["entropy_mean"],
                        "train_batch_loss/total_macs": metrics["total_macs"],
                    })

                total_loss += metrics["loss"].item()
                total_ce_loss += metrics["ce_loss"].item()
                total_constraint_loss += metrics["constraint_loss"].item()
                total_label_mask_loss += metrics["label_mask_loss"].item()
                total_guide_loss += metrics["guide_loss"].item()
                total_predicted_constraint += metrics["predicted_constraint"]
                total_entropy_mean += metrics["entropy_mean"]
                total_macs += metrics["total_macs"]
                total_attn_mask += metrics["attn_mask_mean"]
                total_mlp_mask += metrics["mlp_mask_mean"]
                total_embed_mask += metrics["embed_mask_mean"]
                total_depth_mlp_mask += metrics["depth_mlp_mask_mean"]
                total_depth_attn_mask += metrics["depth_attn_mask_mean"]

                pbar.set_postfix(**{"loss": metrics["loss"].item(), "lr": optimizer.param_groups[0]['lr']})
                pbar.update(1)

        maybe_step_optimizer(model, optimizer, pending_batches, args)

        num_batches = len(trainDataLoader)
        wandb.log({"Epoch": epoch + 1, "train_epoch_loss/Train epoch Loss": total_loss / num_batches})
        wandb.log({"Epoch": epoch + 1, "train_epoch_loss/Train epoch cross entropy loss": total_ce_loss / num_batches})
        wandb.log({"Epoch": epoch + 1, "train_epoch_loss/Train epoch constraint loss": total_constraint_loss / num_batches})
        wandb.log({"Epoch": epoch + 1, "train_epoch_loss/Train epoch label mask loss": total_label_mask_loss / num_batches})
        wandb.log({"Epoch": epoch + 1, "train_epoch_loss/Train epoch guide loss": total_guide_loss / num_batches})
        wandb.log({"Epoch": epoch + 1, "train_epoch/Train predicted constraint": total_predicted_constraint / num_batches})
        wandb.log({"Epoch": epoch + 1, "train_epoch/Train entropy mean": total_entropy_mean / num_batches})
        wandb.log({"Epoch": epoch + 1, "train_epoch/Train macs": total_macs / num_batches})
        wandb.log({"Epoch": epoch + 1, "train_mask/Train attn mask": total_attn_mask / num_batches})
        wandb.log({"Epoch": epoch + 1, "train_mask/Train mlp mask": total_mlp_mask / num_batches})
        wandb.log({"Epoch": epoch + 1, "train_mask/Train embed mask": total_embed_mask / num_batches})
        wandb.log({"Epoch": epoch + 1, "train_mask/Train depth mlp mask": total_depth_mlp_mask / num_batches})
        wandb.log({"Epoch": epoch + 1, "train_mask/Train depth attn mask": total_depth_attn_mask / num_batches})

        if epoch % 2 == 0:
            eval_stage2_entropy(model, valDataLoader, criterion, epoch, optimizer, args, device=args.device, guide=guide)

        torch.save(model.state_dict(), os.path.join(weight_path, 'stage2_entropy_guided.pth'))


if __name__ == '__main__':
    os.environ["WANDB_MODE"] = "offline"

    args = get_args_parser()
    train(args)
