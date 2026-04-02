import torch
from tqdm import tqdm

from config import get_args_parser
from dataloader.entropy_guided import build_entropy_guided_eval_loader
from models.model_stage2 import EAViTStage2, ModifiedBlock


def eval_dynamic(model, valDataLoader, device):
    with tqdm(total=len(valDataLoader), postfix=dict, mininterval=0.3) as pbar:
        pbar.set_description('eval')

        model.eval()

        with torch.no_grad():
            correct = 0
            total = 0
            total_constraint = 0.0
            total_entropy = 0.0
            total_macs = 0.0

            for img, label, entropy_mean in valDataLoader:
                img = img.to(device)
                label = label.to(device)
                batch_entropy = entropy_mean.to(device).float().mean().reshape(1)

                predicted_constraint = model.predict_constraint(batch_entropy)
                model.configure_constraint(constraint=predicted_constraint, tau=1)

                preds, _, _, _, _, _, sample_macs = model(img)

                total_constraint += predicted_constraint.item()
                total_entropy += batch_entropy.item()
                total_macs += sample_macs.item()

                _, predicted = torch.max(preds, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

                pbar.update(1)

            accuracy = 100.0 * correct / total
            print()
            print("val acc", accuracy)
            print("avg predicted constraint", total_constraint / len(valDataLoader))
            print("avg entropy mean", total_entropy / len(valDataLoader))
            print("avg macs", total_macs / len(valDataLoader))


if __name__ == '__main__':
    args = get_args_parser()
    valDataLoader, nb_classes = build_entropy_guided_eval_loader(args)

    device = args.device
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

    para = torch.load(args.stage2_checkpoint_path, map_location=device)
    model.load_state_dict(para, strict=False)
    model = model.to(device)
    model.eval()

    eval_dynamic(model, valDataLoader, device=device)
