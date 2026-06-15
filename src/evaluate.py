import torch
from attacks import fgsm_attack, pgd_attack


def calculate_accuracy(model, data_loader, device, attack_name="clean", epsilon=8/255, alpha=2/255, attack_steps=7):
    model.eval()

    correct_predictions = 0
    total_predictions = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        if attack_name == "clean":
            evaluation_images = images

        elif attack_name == "fgsm":
            evaluation_images = fgsm_attack(
                model=model,
                images=images,
                labels=labels,
                epsilon=epsilon
            )

        elif attack_name == "pgd":
            evaluation_images = pgd_attack(
                model=model,
                images=images,
                labels=labels,
                epsilon=epsilon,
                alpha=alpha,
                steps=attack_steps
            )

        else:
            raise ValueError("attack_name must be clean, fgsm, or pgd")

        with torch.no_grad():
            logits = model(evaluation_images)
            predictions = torch.argmax(logits, dim=1)

        correct_predictions += (predictions == labels).sum().item()
        total_predictions += labels.size(0)

    accuracy = correct_predictions / total_predictions
    return accuracy