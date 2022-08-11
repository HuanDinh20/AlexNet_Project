import torch
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader


def get_FashionMNIST(root, aug_train, aug_test):
    train_set = FashionMNIST(root=root, train=True,
                             transform=aug_train, download=True)
    test_set = FashionMNIST(root=root, train=False, transform=aug_test, download=True)
    return train_set, test_set


def create_dataloader(train_set, test_set, batch_size):
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, num_workers=2)
    validation_loader = DataLoader(dataset=test_set, batch_size=batch_size, num_workers=2)
    return train_loader, validation_loader


def image_augmentation():
    aug_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    aug_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return aug_train, aug_test


def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device
