import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader
from tqdm import tqdm

from attacks import fgsm_attack, pgd_attack
from dataset import create_datasets
from evaluate import calculate_accuracy
from model_factory import create_model


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True


def train_one_epoch(model, train_loader, optimizer, device, training_mode, epsilon, alpha, pgd_steps):
    model.train()

    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    progress_bar = tqdm(train_loader, desc="Training", leave=False)

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        if training_mode == "clean":
            training_images = images
            training_labels = labels

        elif training_mode == "fgsm":
            adversarial_images = fgsm_attack(
                model=model,
                images=images,
                labels=labels,
                epsilon=epsilon
            )

            training_images = torch.cat([images, adversarial_images], dim=0)
            training_labels = torch.cat([labels, labels], dim=0)

        elif training_mode == "pgd":
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

        else:
            raise ValueError("training_mode must be clean, fgsm, or pgd")

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
    parser.add_argument("--model_name", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50"])
    parser.add_argument("--training_mode", type=str, default="clean", choices=["clean", "fgsm", "pgd"])
    parser.add_argument("--resume_path", type=str, default=None)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)

    parser.add_argument("--epsilon", type=float, default=8/255)
    parser.add_argument("--alpha", type=float, default=2/255)
    parser.add_argument("--pgd_steps", type=int, default=7)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default="checkpoints/best_model.pt")

    args = parser.parse_args()

    set_random_seed(args.seed)

    os.makedirs("checkpoints", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Model: {args.model_name}")
    print(f"Training mode: {args.training_mode}")

    train_dataset, validation_dataset = create_datasets(
        npz_path=args.data_path,
        validation_size=5000,
        seed=args.seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    model = create_model(args.model_name)
    if args.resume_path is not None:
        print(f"Loading checkpoint from: {args.resume_path}")
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

    best_score = -1.0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_accuracy = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            training_mode=args.training_mode,
            epsilon=args.epsilon,
            alpha=args.alpha,
            pgd_steps=args.pgd_steps
        )

        scheduler.step()

        clean_accuracy = calculate_accuracy(
            model=model,
            data_loader=validation_loader,
            device=device,
            attack_name="clean",
            epsilon=args.epsilon,
            alpha=args.alpha,
            attack_steps=args.pgd_steps
        )

        if args.training_mode == "clean":
            pgd_accuracy = 0.0
            fgsm_accuracy = 0.0
        else:
            fgsm_accuracy = calculate_accuracy(
                model=model,
                data_loader=validation_loader,
                device=device,
                attack_name="fgsm",
                epsilon=args.epsilon,
                alpha=args.alpha,
                attack_steps=args.pgd_steps
            )

            pgd_accuracy = calculate_accuracy(
                model=model,
                data_loader=validation_loader,
                device=device,
                attack_name="pgd",
                epsilon=args.epsilon,
                alpha=args.alpha,
                attack_steps=args.pgd_steps
            )

        selection_score = 0.5 * clean_accuracy + 0.5 * pgd_accuracy

        print(f"Train loss:       {train_loss:.4f}")
        print(f"Train accuracy:   {train_accuracy:.4f}")
        print(f"Clean accuracy:   {clean_accuracy:.4f}")
        print(f"FGSM accuracy:    {fgsm_accuracy:.4f}")
        print(f"PGD accuracy:     {pgd_accuracy:.4f}")
        print(f"Selection score:  {selection_score:.4f}")

        if clean_accuracy > 0.50 and selection_score > best_score:
            best_score = selection_score
            torch.save(model.state_dict(), args.save_path)
            print(f"Saved model to {args.save_path}")

    print("\nTraining finished.")
    print(f"Best validation selection score: {best_score:.4f}")
    print(f"Best model path: {args.save_path}")


if __name__ == "__main__":
    main()