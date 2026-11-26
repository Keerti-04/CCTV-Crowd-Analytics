import argparse
import copy
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from csrnet_model import CSRNet
from dataset import ShanghaiTechPartADataset


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune CSRNet on ShanghaiTech Part A")
    parser.add_argument("--img-dir", type=str, default="datasets/shanghai/part_A_final/train_data/images")
    parser.add_argument("--gt-dir", type=str, default="datasets/shanghai/part_A_final/train_data/ground_truth")
    parser.add_argument("--weights", type=str, default="models/CSRNett.pth")
    parser.add_argument("--output", type=str, default="models/CSRNet_SE.pth")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--img-size", type=int, nargs=2, default=(512, 512))
    return parser.parse_args()


def main():
    args = parse_args()

    # Use GPU automatically if available (works on Colab T4 as well)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(args.img_dir):
        raise FileNotFoundError(f"Image directory not found: {args.img_dir}")
    if not os.path.exists(args.gt_dir):
        raise FileNotFoundError(f"Ground truth directory not found: {args.gt_dir}")
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Pretrained weights not found: {args.weights}")

    dataset = ShanghaiTechPartADataset(
        img_dir=args.img_dir,
        gt_dir=args.gt_dir,
        img_size=tuple(args.img_size),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2 if device.type == "cuda" else 0,
        pin_memory=(device.type == "cuda"),
    )

  
    model = CSRNet().to(device)

    
    state_dict = torch.load(args.weights, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    print(f"Loaded pretrained weights from {args.weights}")

    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_loss = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0

        for images, targets in dataloader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}/{args.epochs} Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = copy.deepcopy(model.state_dict())

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    if best_state is not None:
        torch.save(best_state, args.output)
        print(f"Best model saved to {args.output}")
    else:
        print("Training finished, but no best checkpoint was saved.")


if __name__ == "__main__":
    main()
