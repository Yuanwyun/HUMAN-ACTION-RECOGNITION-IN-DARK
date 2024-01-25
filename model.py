import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import numpy as np
from torchvision import models
import torch.nn as nn
from torchvision.models.video import r3d_18
import os
from glob import glob
from torchvision.io import read_image
from PIL import ImageEnhance
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.optim as optim


class Action(Dataset):
    def __init__(self, annotations_file, train_dir):
        self.annos = pd.read_csv(annotations_file)
        self.train_dir = train_dir
        self.transforms = T.Compose([
            T.Resize((224, 224), antialias=True),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])
        
    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):
        file_name = self.annos.iloc[idx, 0]
        label = self.annos.iloc[idx, 1]
        label = torch.tensor(label)
        train_path = os.path.join(self.train_dir, file_name)
        
        # Use PIL to read the image
        image = Image.open(train_path).convert('RGB')
        # Adjust brightness
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.5) 

        image = self.transforms(image)
        return image, label

train_dataset = Action(
    annotations_file="/Users/yuanweiyun/Desktop/EE6222/output_train.csv",
    train_dir="/Users/yuanweiyun/Desktop/EE6222/train_data",

)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

for image, labels in train_loader:
    print(image.shape, labels.shape)
    break


class Validate(Dataset):
    def __init__(self, anno_file, valid_dir):
        self.annos = pd.read_csv(anno_file)
        self.valid_dir = valid_dir
        self.transforms = T.Compose([
            T.Resize((224, 224), antialias=True),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):
        file_name_v = self.annos.iloc[idx, 0]
        label_v = self.annos.iloc[idx, 1]
        label_v = torch.tensor(label_v)
        valid_path = os.path.join(self.valid_dir, file_name_v)

        # Use PIL to read the image
        image_v = Image.open(valid_path).convert('RGB')
        enhancer = ImageEnhance.Brightness(image_v)
        image_v = enhancer.enhance(1.5)
        image_v = self.transforms(image_v)

        return image_v, label_v


valid_dataset = Validate(
                    anno_file="/Users/yuanweiyun/Desktop/EE6222/output_valid.csv",
                    valid_dir="/Users/yuanweiyun/Desktop/EE6222/valid_new",
                    )

valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)
for image, labels in valid_loader:
    print(image.shape, labels.shape)
    break


##########################################################################################################################################

# Define the ResNet model

class ResNet(nn.Module):
    def __init__(self, num_classes=6, dropout_prob=0.5, weight_decay=1e-4):
        super(ResNet, self).__init__()
        # Load pre-trained ResNet18 model
        self.resnet = models.resnet18(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        # Add dropout layer
        self.dropout = nn.Dropout(p=dropout_prob)
        in_features = self.get_fc_in_features()
        # Add a new fully connected layer
        self.fc = nn.Linear(in_features, num_classes)

    def get_fc_in_features(self):
        dummy_input = torch.randn(8, 3, 224, 224)  
        with torch.no_grad():
            dummy_output = self.resnet(dummy_input)
            print("Feature Dimension:", dummy_output.shape)
        return dummy_output.view(dummy_output.size(0), -1).shape[1]
    
    def forward(self, x):
        x = self.resnet(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# Function to train the model
def train_model(model, train_loader, criterion, optimizer, num_epochs=5):
    model.train()
    train_accuracies = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = outputs.max(1)
            predicted = predicted.to(labels.device)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        train_accuracies.append(epoch_accuracy)
            
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}, Training Accuracy: {epoch_accuracy}%")

def valid_model(model, valid_loader, criterion):
    model.eval()
    correct = 0
    total = 0
    validation_loss = 0 
    with torch.no_grad():
        for inputs, labels in valid_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            validation_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    accuracy = 100 * correct / total
    print(f"Validation Accuracy: {accuracy}%")
    return accuracy

#device = torch.device("cuda")
criterion = nn.CrossEntropyLoss()
# Instantiate the model with dropout and regularization
model = ResNet(num_classes=6, dropout_prob=0.5, weight_decay=1e-4)
#model.to(device)
# Define the loss function and optimizer (using Adam)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # Adjust lr and weight_decay as needed

# Set up learning rate scheduler
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, verbose=True)

train_model(model, train_loader, criterion, optimizer, num_epochs=5)
valid_model(model, valid_loader, criterion)
