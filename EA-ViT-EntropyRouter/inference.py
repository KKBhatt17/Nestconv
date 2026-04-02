from timm.loss import LabelSmoothingCrossEntropy
import torch
from tqdm import tqdm

from config import get_args_parser
from dataloader.entropy_image_datasets import build_entropy_image_dataset, create_entropy_dataloader
from models.model_stage2 import EAViTStage2, ModifiedBlock
from utils.entropy_conditioning import build_router_input


def eval_dynamic(model, valDataLoader, criterion, device):
    with tqdm(total=len(valDataLoader), postfix=dict, mininterval=0.3) as pbar:
        pbar.set_description("eval")

        model.eval()

        with torch.no_grad():
            correct = 0
            total = 0
            total_loss = 0.0
            total_macs = 0.0
            total_constraint = 0.0

            for img, label, entropy_vectors, _ in valDataLoader:
                img = img.to(device)
                label = label.to(device)

                router_input = build_router_input(entropy_vectors, device)
                predicted_constraint = model.predict_constraint(router_input)
                model.configure_constraint(predicted_constraint, tau=1)

                preds, _, _, _, _, _, sample_macs = model(img)
                loss = criterion(preds, label)
                total_loss += loss.item()
                total_macs += sample_macs.item()
                total_constraint += predicted_constraint.item()

                _, predicted = torch.max(preds, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

                pbar.update(1)

            accuracy = 100.0 * correct / total
            average_loss = total_loss / len(valDataLoader)
            average_macs = total_macs / len(valDataLoader)
            average_constraint = total_constraint / len(valDataLoader)

            print()
            print("val loss", average_loss)
            print("val acc", accuracy)
            print("avg predicted constraint", average_constraint)
            print("avg routed macs", average_macs)
            pbar.close()

    return accuracy


if __name__ == "__main__":
    args = get_args_parser()
    dataset_train, dataset_val, nb_classes = build_entropy_image_dataset(args)
    valDataLoader = create_entropy_dataloader(args, dataset_val, shuffle_batches=False, single_image=True)

    if args.smoothing > 0.0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    device = args.device
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

    para = torch.load(args.stage2_checkpoint_path, map_location=device)
    model.load_state_dict(para, strict=False)
    model = model.to(device)
    model.eval()

    eval_dynamic(model, valDataLoader, criterion, device=device)
