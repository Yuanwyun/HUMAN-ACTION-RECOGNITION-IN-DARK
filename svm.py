import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from sklearn.metrics import accuracy_score
from sklearn import svm

class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.labels_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.labels_df.iloc[idx, 0])
   
        image = Image.open(img_name).convert('RGB')

        label = int(self.labels_df.iloc[idx, 1])  

        if self.transform:
            image = self.transform(image)

        return image, label



def extract_features(model, dataloader):
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            #print(inputs.shape)
            outputs = model(inputs)
            features = outputs.squeeze().cpu().numpy()
            features = np.reshape(features, (features.shape[0], -1))  # Flatten features if needed
            print(features.shape)
            all_features.append(features)
            all_labels.extend(labels.cpu().numpy())

    all_features = np.concatenate(all_features, axis=0)

    return all_features, all_labels


# Specify the paths to your CSV files and image folders
train_csv_file = "/Users/yuanweiyun/Desktop/EE6222/output_train.csv"
valid_csv_file = "/Users/yuanweiyun/Desktop/EE6222/output_valid.csv"
train_data_folder = "/Users/yuanweiyun/Desktop/EE6222/train_data"
valid_data_folder = "/Users/yuanweiyun/Desktop/EE6222/valid_new"

# Define the transformations you want to apply
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.07,0.07,0.07], std=[0.1,0.09,0.08]),
])

# Create instances of the custom dataset for training and validation
train_dataset = CustomDataset(csv_file=train_csv_file, root_dir=train_data_folder, transform=transform)
valid_dataset = CustomDataset(csv_file=valid_csv_file, root_dir=valid_data_folder, transform=transform)

# Create DataLoader instances for training and validation
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)


resnet18 = models.resnet18(pretrained=True)

resnet18 = resnet18.eval()

# Extract features for training and validation datasets
train_features, train_labels = extract_features(resnet18, train_loader)
valid_features, valid_labels = extract_features(resnet18, valid_loader)


svm_classifier = svm.SVC(kernel='linear', C=1.0)
svm_classifier.fit(train_features, train_labels)


valid_predictions = svm_classifier.predict(valid_features)


accuracy = accuracy_score(valid_labels, valid_predictions)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")





