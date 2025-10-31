import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import logging
import os

def get_fashion_mnist_loaders(batch_size: int = 64, data_dir: str = './data') -> (DataLoader, DataLoader):
    """
    Loads the Fashion-MNIST dataset and creates DataLoaders for training and testing.

    Args:
        batch_size (int): Batch size for the DataLoaders.
        data_dir (str): Directory to save/load the Fashion-MNIST dataset.

    Returns:
        tuple[DataLoader, DataLoader]: Tuple containing train_loader and test_loader.
    """
    try:
        # Define transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            # Normalize to [-1, 1] for potential tanh activation (though sigmoid is used in decoder)
            # Normalizing to [0, 1] with just ToTensor() is also common for BCE loss.
            # Here we keep original normalization (-0.5 to 0.5) from notebook but adjust loss.
            transforms.Normalize((0.5,), (0.5,)) 
        ])

        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)

        # Load training dataset
        train_dataset = datasets.FashionMNIST(
            root=data_dir, train=True, transform=transform, download=True
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count() // 2 if os.cpu_count() else 0)
        
        # Load test dataset
        test_dataset = datasets.FashionMNIST(
            root=data_dir, train=False, transform=transform, download=True
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count() // 2 if os.cpu_count() else 0)

        logging.info(f"Successfully loaded Fashion-MNIST dataset into '{data_dir}'.")
        logging.info(f"Train dataset size: {len(train_dataset)}, Test dataset size: {len(test_dataset)}")
        return train_loader, test_loader

    except Exception as e:
        logging.error(f"Error loading Fashion-MNIST dataset: {e}", exc_info=True)
        raise # Re-raise the exception to indicate a critical failure
