import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
from PIL import Image
import os
import yaml

class CIFAR10Classifier:
    def __init__(self, config_path='cifar10_config.yml'):
        self.config = self._load_config(config_path)
        self.transform = self._get_transforms()
        self.batch_size = self.config['batch_size']
        self.num_workers = self.config['num_workers']
        self.device = self._get_device()
        self.trainloader, self.testloader = self._get_dataloaders()
        self.model = self._create_model().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters())
    
    def _load_config(self, config_path):
        with open(config_path, 'r') as ymlfile:
            config = yaml.safe_load(ymlfile)
        return config['model']

    def _get_transforms(self):
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def _get_device(self):
        device_config = self.config['device']
        if device_config == "xla":
            return xm.xla_device()
        elif device_config == "cuda":
            return torch.device("cuda")
        elif device_config == "cpu":
            return torch.device("cpu")
        else:
            raise ValueError("Device is not supported in config file.")
    
    def _get_dataloaders(self):
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=self.transform)
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=self.transform)
        
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        
        return trainloader, testloader

    def _create_model(self):
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=self.config['kernel_size'], padding=self.config['padding']),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=self.config['kernel_size'], padding=self.config['padding']),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    
    def train(self, num_epochs=None):
        if num_epochs is None:
            num_epochs = self.config['num_epochs']
        trainloader = pl.MpDeviceLoader(self.trainloader, self.device)
        self.model.train()
        
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                xm.optimizer_step(self.optimizer)
                running_loss += loss.item()
                if i % 2000 == 1999:
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
            print(f'Epoch {epoch + 1} completed')
        print('Finished Training')
    
    def predict_image(self, image_path):
        image = Image.open(image_path)
        image = self.transform(image).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(image)
            _, predicted = torch.max(outputs, 1)
        classes = tuple(self.config['classes'])
        return classes[predicted.item()]
    
    def save_model(self, path=None):
        if path is None:
            path = self.config['model_path']
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

if __name__ == "__main__":
    classifier = CIFAR10Classifier()
    classifier.train()
    classifier.save_model()
