import argparse
import csv
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18, resnet34, resnet50


NUM_CLASSES = 9
VAL_EPS = 8 / 255.0
VAL_ALPHA = 2 / 255.0
VAL_STEPS = 20


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class ImageDataset(Dataset):
    def __init__(self, images, labels, augment):
        self.images = images
        self.labels = labels

        if augment:
            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def build_model(arch):
    if arch == "resnet18":
        model = resnet18(weights=None)
    elif arch == "resnet34":
        model = resnet34(weights=None)
    elif arch == "resnet50":
        model = resnet50(weights=None)
    else:
        raise ValueError("arch must be resnet18, resnet34, or resnet50")

    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def pgd_attack(model, images, labels, epsilon, alpha, steps):
    was_training = model.training
    model.eval()

    adversarial = images + torch.empty_like(images).uniform_(-epsilon, epsilon)
    adversarial = torch.clamp(adversarial, 0.0, 1.0).detach()

    for _ in range(steps):
        adversarial.requires_grad_(True)

        logits = model(adversarial)
        loss = nn.CrossEntropyLoss()(logits, labels)

        gradient = torch.autograd.grad(loss, adversarial)[0]

        adversarial = adversarial.detach() + alpha * gradient.sign()
        perturbation = torch.clamp(adversarial - images, -epsilon, epsilon)
        adversarial = torch.clamp(images + perturbation, 0.0, 1.0).detach()

    if was_training:
        model.train()

    return adversarial


def evaluate(model, loader, device):
    model.eval()

    clean_correct = 0
    clean_total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            predictions = torch.argmax(logits, dim=1)

            clean_correct += (predictions == labels).sum().item()
            clean_total += labels.size(0)

    clean_accuracy = clean_correct / clean_total

    robust_correct = 0
    robust_total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        adversarial_images = pgd_attack(
            model=model,
            images=images,
            labels=labels,
            epsilon=VAL_EPS,
            alpha=VAL_ALPHA,
            steps=VAL_STEPS
        )

        with torch.no_grad():
            logits = model(adversarial_images)
            predictions = torch.argmax(logits, dim=1)

        robust_correct += (predictions == labels).sum().item()
        robust_total += labels.size(0)

    robust_accuracy = robust_correct / robust_total

    return clean_accuracy, robust_accuracy


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--arch", type=str, required=True)
    parser.add_argument("--eps_int", type=int, required=True)
    parser.add_argument("--pgd_steps", type=int, required=True)

    parser.add_argument("--data_path", type=str, default="data/train.npz")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--results_dir", type=str, default="results")

    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr_max", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--alpha_ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tag = f"gridrun_{args.arch}_eps{args.eps_int}_steps{args.pgd_steps}"
    checkpoint_path = os.path.join(args.checkpoint_dir, f"{tag}.pt")
    result_path = os.path.join(args.results_dir, f"{tag}.csv")

    epsilon = args.eps_int / 255.0
    alpha = epsilon * args.alpha_ratio

    print("=" * 70, flush=True)
    print(f"Run: {tag}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Architecture: {args.arch}", flush=True)
    print(f"Training epsilon: {epsilon:.6f}", flush=True)
    print(f"Training alpha: {alpha:.6f}", flush=True)
    print(f"Training PGD steps: {args.pgd_steps}", flush=True)
    print("=" * 70, flush=True)

    data = np.load(args.data_path)
    images = torch.from_numpy(data["images"]).float() / 255.0
    labels = torch.from_numpy(data["labels"]).long()

    number_of_samples = len(labels)
    number_of_validation_samples = int(0.10 * number_of_samples)

    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(number_of_samples, generator=generator)

    validation_indices = indices[:number_of_validation_samples]
    train_indices = indices[number_of_validation_samples:]

    train_dataset = ImageDataset(
        images=images[train_indices],
        labels=labels[train_indices],
        augment=True
    )

    validation_dataset = ImageDataset(
        images=images[validation_indices],
        labels=labels[validation_indices],
        augment=False
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
        batch_size=256,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    model = build_model(args.arch).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr_max,
        momentum=0.9,
        weight_decay=args.weight_decay,
        nesterov=True
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr_max,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        pct_start=0.05,
        anneal_strategy="cos"
    )

    best_score = -1.0
    best_clean = 0.0
    best_robust = 0.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for images_batch, labels_batch in train_loader:
            images_batch = images_batch.to(device)
            labels_batch = labels_batch.to(device)

            adversarial_images = pgd_attack(
                model=model,
                images=images_batch,
                labels=labels_batch,
                epsilon=epsilon,
                alpha=alpha,
                steps=args.pgd_steps
            )

            model.train()
            optimizer.zero_grad(set_to_none=True)

            logits = model(adversarial_images)
            loss = criterion(logits, labels_batch)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * labels_batch.size(0)
            predictions = torch.argmax(logits.detach(), dim=1)
            correct += (predictions == labels_batch).sum().item()
            total += labels_batch.size(0)

        train_loss = total_loss / total
        train_adv_accuracy = correct / total

        print(
            f"[{tag}] epoch {epoch:03d}/{args.epochs} "
            f"loss={train_loss:.4f} adv_acc={train_adv_accuracy:.4f}",
            flush=True
        )

        if epoch % 25 == 0 or epoch == args.epochs:
            clean_accuracy, robust_accuracy = evaluate(
                model=model,
                loader=validation_loader,
                device=device
            )

            score = 0.5 * clean_accuracy + 0.5 * robust_accuracy

            print(
                f"[{tag}] validation epoch={epoch} "
                f"clean={clean_accuracy:.4f} "
                f"robust={robust_accuracy:.4f} "
                f"score={score:.4f}",
                flush=True
            )

            if clean_accuracy > 0.50 and score > best_score:
                best_score = score
                best_clean = clean_accuracy
                best_robust = robust_accuracy
                best_epoch = epoch

                torch.save(model.state_dict(), checkpoint_path)

                print(
                    f"[{tag}] saved best checkpoint: {checkpoint_path}",
                    flush=True
                )

    with open(result_path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "tag",
                "arch",
                "eps_int",
                "pgd_steps",
                "best_epoch",
                "clean_accuracy",
                "robust_accuracy",
                "score",
                "checkpoint_path"
            ]
        )

        writer.writeheader()
        writer.writerow({
            "tag": tag,
            "arch": args.arch,
            "eps_int": args.eps_int,
            "pgd_steps": args.pgd_steps,
            "best_epoch": best_epoch,
            "clean_accuracy": round(best_clean, 4),
            "robust_accuracy": round(best_robust, 4),
            "score": round(best_score, 4),
            "checkpoint_path": checkpoint_path
        })

    print("=" * 70, flush=True)
    print(f"Finished: {tag}", flush=True)
    print(f"Best epoch: {best_epoch}", flush=True)
    print(f"Best clean accuracy: {best_clean:.4f}", flush=True)
    print(f"Best robust accuracy: {best_robust:.4f}", flush=True)
    print(f"Best score: {best_score:.4f}", flush=True)
    print(f"Checkpoint: {checkpoint_path}", flush=True)
    print(f"Result CSV: {result_path}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()