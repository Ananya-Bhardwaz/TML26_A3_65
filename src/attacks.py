import torch
import torch.nn.functional as functional


def fgsm_attack(model, images, labels, epsilon):
    attacked_images = images.detach().clone()
    attacked_images.requires_grad = True

    model.zero_grad(set_to_none=True)

    logits = model(attacked_images)
    loss = functional.cross_entropy(logits, labels)
    loss.backward()

    gradient_sign = attacked_images.grad.detach().sign()
    adversarial_images = attacked_images + epsilon * gradient_sign
    adversarial_images = torch.clamp(adversarial_images, 0.0, 1.0)

    return adversarial_images.detach()


def pgd_attack(model, images, labels, epsilon, alpha, steps):
    original_images = images.detach()

    adversarial_images = original_images + torch.empty_like(original_images).uniform_(
        -epsilon,
        epsilon
    )
    adversarial_images = torch.clamp(adversarial_images, 0.0, 1.0)

    for _ in range(steps):
        adversarial_images.requires_grad = True

        model.zero_grad(set_to_none=True)

        logits = model(adversarial_images)
        loss = functional.cross_entropy(logits, labels)
        loss.backward()

        gradient_sign = adversarial_images.grad.detach().sign()

        adversarial_images = adversarial_images.detach() + alpha * gradient_sign

        perturbation = torch.clamp(
            adversarial_images - original_images,
            min=-epsilon,
            max=epsilon
        )

        adversarial_images = torch.clamp(original_images + perturbation, 0.0, 1.0)

    return adversarial_images.detach()