import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from model import ResNet34

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(client_id, num_clients=3, batch_size=64):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761))
    ])

    train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    test_dataset  = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    
    total_len = len(train_dataset)
    shard_len = total_len // num_clients
    start = client_id * shard_len
    end = start + shard_len if client_id != num_clients - 1 else total_len

    indices = list(range(start, end))
    client_trainset = Subset(train_dataset, indices)

    train_loader = DataLoader(client_trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def get_model():
    model = ResNet34()
    return model  


