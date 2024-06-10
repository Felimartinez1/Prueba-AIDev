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

with open ('cifar10_config.yml', 'r') as ymlfile:
    config = yaml.safeload(ymlfile)
    cifar10_config = ['model']

class CIFAR10Classifier:
    def __init__(self, batch_size=cifar10_config['batch_size'], num_workers=cifar10_config['num_workers']):
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),  # CIFAR-10 images are 32x32
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize pixel values
        ])
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=self.transform)
        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=self.transform)
        
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        if cifar10_config['device'] == "xla":
            self.device = xm.xla_device()
        elif cifar10_config["device"] == "cuda":
            self.device = torch.device("cuda")
        elif cifar10_config["device"] == "cpu":
            self.device = torch.device("cpu")
        else:
            raise ValueError("Device is not supported in config file.")
        
        self.model = self._create_model().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters())
        
    def _create_model(self):
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=cifar10_config['kernel_size'], padding=cifar10_config['padding']),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=cifar10_config['kernel_size'], padding=cifar10_config['padding']),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    
    def train(self, num_epochs=cifar10_config['num_epochs']):
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
            print('+ epoch')
        print('Finished Training')
    
    def predict_image(self, image_path):
        image = Image.open(image_path)
        image = self.transform(image).unsqueeze(0)
        image = image.to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(image)
            _, predicted = torch.max(outputs, 1)
        classes = tuple(cifar10_config['classes'])
        return classes[predicted.item()]
    
    def save_model(self, path='model.pth'):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        torch.save(self.model.state_dict(), path)    

if __name__ == "__main__":
    # Ejemplo de uso
    classifier = CIFAR10Classifier()
    classifier.train()
    classifier.save_model('model\cifar10_model.pth')



