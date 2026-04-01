import torch
from tqdm import tqdm
import wandb

from utils.entropy_conditioning import (
    build_router_input,
    encoding_to_mask_tensors,
    entropy_score_from_vectors,
    select_lookup_entry,
)

def eval_stage1(model, valDataLoader, criterion, epoch, optimizer, args, flag, **kwargs):
    with tqdm(total=len(valDataLoader), postfix=dict, mininterval=0.3) as pbar:
        pbar.set_description(f'eval Epoch {epoch + 1}/{args.epochs}')

        model.eval()

        with torch.no_grad():
            total_loss = 0.0
            correct = 0
            total = 0
            for batch_idx, (img, label) in enumerate(valDataLoader):
                img = img.to(args.device)
                label = label.to(args.device)

                model.configure_subnetwork(**kwargs)
                preds = model(img)

                loss = criterion(preds, label)
                total_loss += loss.item()

                _, predicted = torch.max(preds, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

                pbar.set_postfix(**{"loss": loss.item(), "lr": optimizer.param_groups[0]['lr']})
                pbar.update(1)

            avg_loss = total_loss / len(valDataLoader)
            accuracy = 100.0 * correct / total
            print("val loss", avg_loss)
            print("val acc", accuracy)
            wandb.log({"Epoch": epoch + 1, "val/Val Loss_" + flag: avg_loss})
            wandb.log({"Epoch": epoch + 1, "val/Val Acc_" + flag: accuracy})

def eval_stage2(model, valDataLoader, criterion, epoch, optimizer, args, flag, lookup_rows, device):
    with tqdm(total=len(valDataLoader), postfix=dict, mininterval=0.3) as pbar:
        pbar.set_description(f'eval Epoch {epoch + 1}/{args.epochs}')

        model.eval()

        with torch.no_grad():
            total_loss = 0.0
            total_ce_loss = 0.0
            total_label_mask_loss = 0.0

            total_attn_mask = 0
            total_mlp_mask = 0
            total_embed_mask = 0

            total_depth_mlp_mask = 0
            total_depth_attn_mask = 0

            total_macs_sum = 0

            correct = 0
            total = 0

            for batch_idx, (img, label, entropy_vectors, _) in enumerate(valDataLoader):
                img = img.to(device)
                label = label.to(device)

                router_input = build_router_input(entropy_vectors, device)
                target_entry = select_lookup_entry(lookup_rows, entropy_score_from_vectors(entropy_vectors))
                label_mlp_mask, label_mha_mask, label_emb_mask, label_depth_mlp_mask, label_depth_attn_mask = encoding_to_mask_tensors(
                    target_entry["encoding"],
                    device,
                )

                t = 1
                model.configure_router_input(router_input=router_input, tau=t)
                preds, attn_mask, mlp_mask, embed_mask, depth_attn_mask, depth_mlp_mask, total_macs = model(img)

                attn_mask_mean = torch.mean(attn_mask)
                mlp_mask_mean = torch.mean(mlp_mask)
                embed_mask_mean = torch.mean(embed_mask)
                depth_mlp_mask_mean = torch.mean(depth_mlp_mask)
                depth_attn_mask_mean = torch.mean(depth_attn_mask)

                ce_loss = criterion(preds, label)

                label_mask_loss = (
                    torch.nn.functional.mse_loss(attn_mask, label_mha_mask)
                    + torch.nn.functional.mse_loss(mlp_mask, label_mlp_mask)
                    + torch.nn.functional.mse_loss(embed_mask, label_emb_mask)
                    + torch.nn.functional.mse_loss(depth_mlp_mask, label_depth_mlp_mask)
                    + torch.nn.functional.mse_loss(depth_attn_mask, label_depth_attn_mask)
                )

                loss = ce_loss + label_mask_loss

                total_loss += loss.item()
                total_ce_loss += ce_loss.item()
                total_label_mask_loss += label_mask_loss.item()

                total_attn_mask += attn_mask_mean.item()
                total_mlp_mask += mlp_mask_mean.item()
                total_embed_mask += embed_mask_mean.item()
                total_depth_mlp_mask += depth_mlp_mask_mean.item()
                total_depth_attn_mask += depth_attn_mask_mean.item()

                total_macs_sum += total_macs.item()

                _, predicted = torch.max(preds, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

                pbar.set_postfix(**{"loss": loss.item(), "lr": optimizer.param_groups[0]['lr']})
                pbar.update(1)

            avg_loss = total_loss / len(valDataLoader)
            avg_ce_loss = total_ce_loss / len(valDataLoader)
            avg_label_mask_loss = total_label_mask_loss / len(valDataLoader)

            val_attn_mask = total_attn_mask / len(valDataLoader)
            val_mlp_mask = total_mlp_mask / len(valDataLoader)
            val_embed_mask = total_embed_mask / len(valDataLoader)
            val_depth_mlp_mask = total_depth_mlp_mask / len(valDataLoader)
            val_depth_attn_mask = total_depth_attn_mask / len(valDataLoader)
            val_macs = total_macs_sum / len(valDataLoader)

            correct = torch.tensor(correct, dtype=torch.float32, device=device)
            total = torch.tensor(total, dtype=torch.float32, device=device)

            accuracy = correct / total
            print(f"Accuracy: {accuracy.item()}")
            wandb.log({"Epoch": epoch + 1, "val_loss/Val Loss_" + flag: avg_loss})
            wandb.log({"Epoch": epoch + 1, "val_loss/Val ce Loss_" + flag: avg_ce_loss})
            wandb.log({"Epoch": epoch + 1, "val_loss/Val encoding Loss_" + flag: avg_label_mask_loss})

            wandb.log({"Epoch": epoch + 1, "acc/Val Acc_" + flag: accuracy})

            wandb.log({"Epoch": epoch + 1, "val_attn_mask/Val attn mask_" + flag: val_attn_mask})
            wandb.log({"Epoch": epoch + 1, "val_mlp_mask/Val mlp mask_" + flag: val_mlp_mask})
            wandb.log({"Epoch": epoch + 1, "val_embed_mask/Val embed mask_" + flag: val_embed_mask})

            wandb.log({"Epoch": epoch + 1, "val_depth_mlp_mask/Val depth mlp mask_" + flag: val_depth_mlp_mask})
            wandb.log({"Epoch": epoch + 1, "val_depth_attn_mask/Val depth attn mask_" + flag: val_depth_attn_mask})

            wandb.log({"Epoch": epoch + 1, "val_mac/Val macs_" + flag: val_macs})

            wandb.log({"Epoch": epoch + 1, "val_rate/macs rate_" + flag: accuracy/val_macs})




