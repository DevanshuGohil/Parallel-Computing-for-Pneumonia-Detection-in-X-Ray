import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, DistributedSampler
import os
import time
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import csv
from sklearn.metrics import accuracy_score
import argparse
import torch.distributed as dist
import torch.nn.functional as F

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train and test a model on distributed CPUs.')
    parser.add_argument('--num_cpus', type=int, default=1, help='Number of CPUs to use for training and testing.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training and testing.')
    args = parser.parse_args()
    return args

def initialize_csv():
    csv_file = 'training_times.csv'
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["CPUs", "Batch Size", "Training Time (Seconds)"])

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    torch.manual_seed(42)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


# def get_dataloader(rank,world_size,batch_size,train=True):
#     # Define the transformation and datasets
#     augmentation_transforms = transforms.Compose([
#         transforms.Resize((256, 256)),
#         transforms.CenterCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomRotation(10),
#         transforms.RandomGrayscale(),
#         transforms.RandomAffine(translate=(0.05, 0.05), degrees=0),
#         transforms.ToTensor()
#     ])

#     test_transform = transforms.Compose([
#         transforms.Resize((256, 256)),
#         transforms.CenterCrop(224),
#         transforms.ToTensor()
# ])
#     # Load the appropriate dataset
#     if train:
#         dataset = datasets.ImageFolder(root='./Pediatric Chest X-ray Pneumonia/train', transform=augmentation_transforms)
#     else:
#         dataset = datasets.ImageFolder(root='./Pediatric Chest X-ray Pneumonia/test', transform=test_transform)

#     # Create the DistributedSampler
#     sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=train)

#     # Create the DataLoader
#     loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

#     return loader

# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super(SimpleCNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         return self.fc3(x)

def get_dataloader(rank,world_size,batch_size,train=True):
    # Define the transformation and datasets
    augmentation_transforms = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Load the appropriate dataset
    if train:
        dataset = datasets.ImageFolder(root='./Pediatric Chest X-ray Pneumonia/train', transform=augmentation_transforms)
    else:
        dataset = datasets.ImageFolder(root='./Pediatric Chest X-ray Pneumonia/test', transform=augmentation_transforms)

    # Create the DistributedSampler
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=train)

    # Create the DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    return loader

# CNN Model Definition
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Increasing the depth of the convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Adaptive pooling layer to ensure the output size is (6, 6) regardless of the input size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))
        
        # Fully connected layers
        self.fc1 = nn.Linear(512 * 6 * 6, 1024)  # Adjusted for 6x6x512 input features
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 2)
        
        # Dropout layers
        self.dropout = nn.Dropout(0.5)
        
        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = self.pool(self.relu(self.conv5(x)))
        
        # Apply adaptive pooling
        x = self.adaptive_pool(x)
        
        # Flatten the output for the fully connected layer
        x = x.view(-1, 512 * 6 * 6)
        
        # Apply fully connected layers with dropout and ReLU
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)  # No dropout before the final layer
        return x

# def train(rank, world_size, batch_size):
#     setup(rank, world_size)
#     print(f"Starting Training on Rank {rank}.")

   
#     train_loader = get_dataloader(rank, world_size,batch_size, train=True)
#     model = SimpleCNN()
#     ddp_model = DDP(model, device_ids=None)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(ddp_model.parameters(), lr=0.004)

#     if rank == 0:
#         start_time = time.time()

#     for epoch in range(5):
#         ddp_model.train()
#         for i, (inputs, labels) in enumerate(train_loader):
#             optimizer.zero_grad()
#             outputs = ddp_model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             print(f'Rank {rank}, Epoch {epoch}, Loss: {loss.item()}')

#     if rank == 0:
#         training_time = time.time() - start_time
#         print(f"Training completed in {training_time:.2f} seconds.")
#         with open('training_times.csv', mode='a', newline='') as file:
#             writer = csv.writer(file)
#             writer.writerow([world_size, batch_size, training_time])

#     cleanup()

def train(rank, world_size, batch_size):
    setup(rank, world_size)
    if rank == 0:
        print(f"Starting Training on Rank {rank}.")

    train_loader = get_dataloader(rank, world_size, batch_size, train=True)
    model = SimpleCNN()
    ddp_model = DDP(model, device_ids=None)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ddp_model.parameters(), lr=0.004)

    start_time = time.time() if rank == 0 else None  # Timing only on rank 0

    for epoch in range(5):
        ddp_model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = ddp_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if rank == 0:  # Output only from rank 0 to avoid clutter
                print(f'Rank {rank}, Epoch {epoch}, Batch {i}, Loss: {loss.item()}')

        dist.barrier()  # Synchronize all ranks at the end of each epoch

    if rank == 0:
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds.")
        with open('training_times.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([world_size, batch_size, training_time])

    cleanup()


def main():
    args = parse_arguments()
    world_size = args.num_cpus
    batch_size = args.batch_size
    initialize_csv()
    print(f"Using {world_size} CPUs with a batch size of {batch_size}.")
    torch.multiprocessing.spawn(train, args=(world_size, batch_size), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
