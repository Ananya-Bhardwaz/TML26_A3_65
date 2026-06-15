import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader
from tqdm import tqdm

from attacks import pgd_attack
from dataset import RobustnessDataset
from model_factory import create_model


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def train_one_epoch(model, train_loader, optimizer, device, epsilon, alpha, pgd_steps):
    model.train()

    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    progress_bar = tqdm(train_loader, desc="Training", leave=False)

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        adversarial_images = pgd_attack(
            model=model,
            images=images,
            labels=labels,
            epsilon=epsilon,
            alpha=alpha,
            steps=pgd_steps
        )

        training_images = torch.cat([images, adversarial_images], dim=0)
        training_labels = torch.cat([labels, labels], dim=0)

        optimizer.zero_grad(set_to_none=True)

        logits = model(training_images)
        loss = functional.cross_entropy(logits, training_labels)

        loss.backward()
        optimizer.step()

        batch_size = training_labels.size(0)
        total_loss += loss.item() * batch_size

        predictions = torch.argmax(logits.detach(), dim=1)
        correct_predictions += (predictions == training_labels).sum().item()
        total_predictions += batch_size

        progress_bar.set_postfix(loss=loss.item())

    average_loss = total_loss / total_predictions
    average_accuracy = correct_predictions / total_predictions

    return average_loss, average_accuracy


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default="data/train.npz")
    parser.add_argument("--model_name", type=str, default="resnet18")
    parser.add_argument("--resume_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)

    parser.add_argument("--epsilon", type=float, default=8/255)
    parser.add_argument("--alpha", type=float, default=1.5/255)
    parser.add_argument("--pgd_steps", type=int, default=7)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_random_seed(args.seed)
    os.makedirs("checkpoints", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Model: {args.model_name}")
    print("Training on FULL dataset: no validation split")
    print(f"Loading checkpoint from: {args.resume_path}")

    full_train_dataset = RobustnessDataset(
        npz_path=args.data_path,
        indices=None,
        use_augmentation=True
    )

    train_loader = DataLoader(
        full_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    model = create_model(args.model_name)
    state_dict = torch.load(args.resume_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_accuracy = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epsilon=args.epsilon,
            alpha=args.alpha,
            pgd_steps=args.pgd_steps
        )

        scheduler.step()

        print(f"Train loss:     {train_loss:.4f}")
        print(f"Train accuracy: {train_accuracy:.4f}")

        torch.save(model.state_dict(), args.save_path)
        print(f"Saved model to {args.save_path}")

    print("\nFull-data training finished.")
    print(f"Final model path: {args.save_path}")


if __name__ == "__main__":
    main()
