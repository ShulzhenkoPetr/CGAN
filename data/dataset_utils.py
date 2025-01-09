from typing import Dict, Tuple
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_mnist_loaders(config: Dict) -> Tuple[DataLoader]:

    root_path = '../data/'

    train_transform = transforms.Compose([
                                        transforms.RandomHorizontalFlip(),
                                        # transforms.RandomCrop(28,4),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.2860,), (0.3204,))
                                    ])
    test_transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.2860,), (0.3204,))
                                    ])
    
    train_dataset = datasets.FashionMNIST(root=root_path, train=True, download=True, transform=train_transform)
    train_size = int(config.size_train_set * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=test_transform)
    
    train_sampler = validation_sampler = test_sampler = None
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=(train_sampler is None and config.transform_train),
                                                   num_workers=config.workers, pin_memory=True, sampler=train_sampler,
                                                   persistent_workers=config.workers > 0)
    valid_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,
                                                   num_workers=config.workers, pin_memory=True, sampler=validation_sampler,
                                                   persistent_workers=config.workers > 0)
    
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                                                  num_workers=config.workers, pin_memory=True, sampler=test_sampler,
                                                  persistent_workers=config.workers > 0)
    
    return train_dataloader, valid_dataloader, test_dataloader