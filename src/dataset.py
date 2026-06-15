import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class RobustnessDataset(Dataset):
    def __init__(self, npz_path, indices=None, use_augmentation=False):
        loaded_data = np.load(npz_path, mmap_mode="r")

        self.images = loaded_data["images"]
        self.labels = loaded_data["labels"]

        if indices is None:
            self.indices = np.arange(len(self.labels))
        else:
            self.indices = np.array(indices)

        self.use_augmentation = use_augmentation

        self.augmentation = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()
        ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        real_index = self.indices[index]

        image = self.images[real_index].astype(np.float32) / 255.0
        label = int(self.labels[real_index])

        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        if self.use_augmentation:
            image = self.augmentation(image)

        return image, label


def create_datasets(npz_path, validation_size=5000, seed=42):
    loaded_data = np.load(npz_path, mmap_mode="r")
    number_of_samples = len(loaded_data["labels"])

    all_indices = np.arange(number_of_samples)

    random_generator = np.random.default_rng(seed)
    random_generator.shuffle(all_indices)

    validation_indices = all_indices[:validation_size]
    train_indices = all_indices[validation_size:]

    train_dataset = RobustnessDataset(
        npz_path=npz_path,
        indices=train_indices,
        use_augmentation=True
    )

    validation_dataset = RobustnessDataset(
        npz_path=npz_path,
        indices=validation_indices,
        use_augmentation=False
    )

    return train_dataset, validation_dataset