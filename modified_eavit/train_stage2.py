from config import get_args_parser
from datetime import datetime
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from models.model_stage2 import EAViTStage2, ModifiedBlock
from utils.set_wandb import set_wandb
import wandb
from utils.lr_sched import adjust_learning_rate, adjust_learning_rate_by_step
from utils.eval_flag import eval_stage2
from utils.entropy_router import build_entropy_router_input, get_entropy_feature_dim
from timm.loss import LabelSmoothingCrossEntropy
import torch.nn.functional as F
from dataloader.image_datasets import build_image_dataset
from collections import defaultdict
import csv
from itertools import cycle

flags_list = ['0.2', '0.4', '0.6', '0.8', '1.0']

constraint_list = [0.2, 0.4, 0.6, 0.8, 1.0]

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
    constraints = torch.as_tensor(constraint, device=device, dtype=torch.float32).reshape(-1)

    mlp_masks = []
    mha_masks = []
    emb_masks = []
    depth_mlp_masks = []
    depth_attn_masks = []

    for value in constraints.tolist():
        index = min(range(len(macs)), key=lambda i: abs(macs[i] - value))
        encoding = [int(digit) for digit in list(encodings[index])]

        embed_sum = int(sum(encoding[:12]))
        emb_masks.append(torch.cat((torch.ones(embed_sum, device=device), torch.zeros(12-embed_sum, device=device)), dim=0))

        depth_attn_masks.append(torch.tensor([float(i) for i in encoding[12:24]], device=device))
        depth_mlp_masks.append(torch.tensor([float(i) for i in encoding[24:36]], device=device))

        mha_list = []
        mlp_list = []

        for i in range(12):
            attn_sum = int(sum(encoding[36 + i * 12: 36 + (i + 1) * 12]))
            mha_list.append(torch.cat((torch.ones(attn_sum, device=device), torch.zeros(12-attn_sum, device=device)), dim=0))

        for i in range(12):
            mlp_sum = int(sum(encoding[180+i*8 : 180+(i+1)*8]))
            mlp_list.append(torch.cat((torch.ones(mlp_sum, device=device), torch.zeros(8 - mlp_sum, device=device)), dim=0))

        mha_masks.append(torch.stack(mha_list))
        mlp_masks.append(torch.stack(mlp_list))

    if constraints.numel() == 1:
        return mlp_masks[0], mha_masks[0], emb_masks[0], depth_mlp_masks[0], depth_attn_masks[0]

    return (
        torch.stack(mlp_masks),
        torch.stack(mha_masks),
        torch.stack(emb_masks),
        torch.stack(depth_mlp_masks),
        torch.stack(depth_attn_masks),
    )


def train(args):
    torch.cuda.set_device(args.device)
    dataset_train, dataset_val, nb_classes = build_image_dataset(args)

    trainDataLoader = DataLoader(dataset_train, args.batch_size, shuffle=True, num_workers=args.num_workers)
    valDataLoader = DataLoader(dataset_val, args.batch_size, shuffle=True, num_workers=args.num_workers)

    router_input_dim = get_entropy_feature_dim(args.input_size, args.entropy_patch_size)
    model = EAViTStage2(embed_dim=768, depth=12, mlp_ratio=4, num_heads=12, num_classes=nb_classes,
                                 drop_path_rate=args.drop_path, qkv_bias=True, block=ModifiedBlock,
                                 router_input_dim=router_input_dim)

    stage1_checkpoint_path = args.stage1_checkpoint_path
    checkpoint = torch.load(stage1_checkpoint_path, map_location=args.device)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()

    model.to(args.device)

    for name, param in model.named_parameters():
        param.requires_grad = True
        if 'router' not in name:
            param.requires_grad = False

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name} is trainable")
        else:
            print(f"{name} is frozen")

    if args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    param_groups = [
        {'params': [p for n, p in model.named_parameters() if "router" not in n], 'lr':args.max_lr},
        {'params': [p for n, p in model.named_parameters() if "router" in n], 'lr':1e-3, 'lr_scale':1e3},
    ]

    optimizer = torch.optim.AdamW(param_groups)

    folder_path = 'logs_weight/stage_2/'+args.dataset

    os.makedirs(folder_path, exist_ok=True)
    time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(folder_path, time)
    os.makedirs(log_dir)

    weight_path = os.path.join(log_dir, 'weight')
    os.makedirs(weight_path)

    set_wandb(args, name='EA-ViT_stage2')

    predefined = load_pareto_data(args.nsga_path)

    global_step = 0

    with tqdm(total=args.total_steps, mininterval=0.3) as pbar:
        for img, label in cycle(trainDataLoader):

            if global_step >= args.total_steps:
                break

            cur_lr = adjust_learning_rate_by_step(optimizer, global_step, args)

            img, label = img.to(args.device), label.to(args.device)
            optimizer.zero_grad()

            constraint = (torch.rand(img.size(0), device=args.device) * 0.9 + 0.2)
            router_input = build_entropy_router_input(img, patch_size=args.entropy_patch_size)
            model.configure_router_input(router_input=router_input, tau=1)

            preds, attn_mask, mlp_mask, embed_mask, depth_attn_mask, depth_mlp_mask, total_macs = model(img)

            ce_loss = criterion(preds, label)

            constraint_loss = F.mse_loss(total_macs, constraint)

            label_mlp_mask, label_mha_mask, label_emb_mask, label_depth_mlp_mask, label_depth_attn_mask = get_preset_mask_nsga(args.gen_id, constraint, args.device, predefined)

            label_mask_loss = F.mse_loss(attn_mask, label_mha_mask) + F.mse_loss(mlp_mask, label_mlp_mask) + F.mse_loss(
                embed_mask, label_emb_mask) + F.mse_loss(depth_mlp_mask, label_depth_mlp_mask) + F.mse_loss(
                depth_attn_mask, label_depth_attn_mask)

            loss = ce_loss + constraint_loss * 20 + label_mask_loss * 20

            loss.backward()
            optimizer.step()

            if global_step % 10 == 0:
                wandb.log({
                    "stage2_loss/total": loss.item(),
                    "stage2_loss/ce": ce_loss.item(),
                    "stage2_loss/constraint": constraint_loss.item(),
                    "stage2_loss/label_mask": label_mask_loss.item(),
                    "lr": cur_lr,
                }, step=global_step)

            pbar.set_description(f"step {global_step}/{args.total_steps}")
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{cur_lr:.2e}")
            pbar.update(1)

            global_step += 1

            break




    for name, param in model.named_parameters():
        param.requires_grad = True

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name} is trainable")
        else:
            print(f"{name} is frozen")


    for epoch in range(args.epochs):

        adjust_learning_rate(optimizer, epoch + 1, args)

        with tqdm(total=len(trainDataLoader), postfix=dict, mininterval=0.3) as pbar:
            pbar.set_description(f'train Epoch {epoch + 1}/{args.epochs}')

            wandb.log({"Epoch": epoch + 1, "lr/vit learning_rate": optimizer.param_groups[0]['lr']})
            wandb.log({"Epoch": epoch + 1, "lr/router learning_rate": optimizer.param_groups[1]['lr']})

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

            for batch_idx, (img, label) in enumerate(trainDataLoader):

                img = img.to(args.device)
                label = label.to(args.device)

                optimizer.zero_grad()

                constraint = (torch.rand(img.size(0), device=args.device) * 0.9 + 0.2)

                t = 1
                router_input = build_entropy_router_input(img, patch_size=args.entropy_patch_size)
                model.configure_router_input(router_input=router_input, tau=t)

                preds, attn_mask, mlp_mask, embed_mask, depth_attn_mask, depth_mlp_mask, total_macs = model(img)

                ce_loss = criterion(preds, label)

                constraint_loss = F.mse_loss(total_macs, constraint)


                label_mlp_mask, label_mha_mask, label_emb_mask, label_depth_mlp_mask, label_depth_attn_mask = get_preset_mask_nsga(args.gen_id, constraint, args.device, predefined)


                label_mask_loss = F.mse_loss(attn_mask, label_mha_mask) + F.mse_loss(mlp_mask, label_mlp_mask) + F.mse_loss(embed_mask, label_emb_mask) + F.mse_loss(depth_mlp_mask, label_depth_mlp_mask) + F.mse_loss(depth_attn_mask, label_depth_attn_mask)

                loss = ce_loss + constraint_loss * 20 + label_mask_loss * 20 * (1-epoch/args.epochs)


                attn_mask = torch.mean(attn_mask)
                mlp_mask = torch.mean(mlp_mask)
                embed_mask = torch.mean(embed_mask)
                depth_mlp_mask = torch.mean(depth_mlp_mask)
                depth_attn_mask = torch.mean(depth_attn_mask)

                if batch_idx % 10 == 0:
                    wandb.log({"train_batch_loss/batch cross entropy loss": ce_loss})
                    wandb.log({"train_batch_loss/batch constraint loss": constraint_loss})
                    wandb.log({"train_batch_loss/batch label mask loss": label_mask_loss})
                    wandb.log({"train_batch_loss/train Batch Loss": loss.item()})

                total_loss += loss.item()
                total_ce_loss += ce_loss.item()
                total_constraint_loss += constraint_loss.item()
                total_label_mask_loss += label_mask_loss.item()

                total_attn_mask += attn_mask.item()
                total_mlp_mask += mlp_mask.item()
                total_embed_mask += embed_mask.item()
                total_depth_mlp_mask += depth_mlp_mask.item()
                total_depth_attn_mask += depth_attn_mask.item()

                loss.backward()
                optimizer.step()

                pbar.set_postfix(**{"loss": loss.item(), "lr": optimizer.param_groups[0]['lr']})
                pbar.update(1)

                break

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

            pbar.close()


        if epoch % 2 == 0:
            for index, f in enumerate(flags_list):
                constraint = torch.tensor(constraint_list[index]).to(args.device).unsqueeze(0)
                eval_stage2(model, valDataLoader, criterion, epoch, optimizer, args, flag=f, constraint=constraint, device=args.device)

        torch.save(model.state_dict(), weight_path+'/stage2.pth')

if __name__ == '__main__':
    import os
    os.environ["WANDB_MODE"] = "offline"

    args = get_args_parser()
    train(args)




