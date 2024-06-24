import random
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torch 
from torchsummary import summary

import shutil
import os
from torchvision.utils import save_image


import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Paths to your training and testing data
data_dir = '/home/DLIP_user3/Proyect/asl_alphabet_train_detected/'
images_path = '/home/DLIP_user3/Proyect/asl_alphabet_validation_detected'

# Define transformations
transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Load the dataset using ImageFolder
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# Get class names
class_names = dataset.classes
num_classes = len(class_names)
class_names

# Define the ratio for train/test split
train_ratio = 0.8
train_size = int(train_ratio * len(dataset))
test_size = len(dataset) - train_size

# Split the dataset into training and testing sets
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Get the number of samples in each set
num_train_samples = len(train_dataset)
num_test_samples = len(test_dataset)

print(f'Number of training samples: {num_train_samples}')
print(f'Number of testing samples: {num_test_samples}')

num_classes = len(dataset.classes)
num_classes

# Create DataLoaders for training and testing sets
# Create DataLoaders
print("Initializing Datasets and Dataloaders...")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

# Example of iterating over the training data
for images, labels in train_loader:
    print(f"Shape of images [N, C, H, W]: {images.shape} {images.dtype}")
    print(f"Shape of labels: {labels.shape} {labels.dtype}")
    break
    

# Example of iterating over the testing data
for images, labels in test_loader:
    print(f"Shape of images [N, C, H, W]: {images.shape} {images.dtype}")
    print(f"Shape of labels: {labels.shape} {labels.dtype}")
    break

if os.path.exists(images_path):
    shutil.rmtree(images_path)
os.mkdir(images_path)

def save_test_dataset(dataset, class_names, images_path):
# Crear subdirectorios para cada clase
    for class_name in class_names:
        class_dir = os.path.join(images_path, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
    
    # Guardar cada imagen en su correspondiente subdirectorio de clase
    for i, (img, label) in enumerate(dataset):
        class_name = class_names[label]
        img_path = os.path.join(images_path, class_name, f"{i}.png")
        save_image(img, img_path)

# Llamar a la función para guardar el test_dataset
save_test_dataset(test_dataset, class_names, images_path)

# Select GPU or CPU for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=29):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(128 * 23 * 23, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 23 * 23)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
   
#Nuevo Modelo  
#my_model = SimpleCNN().to(device)
#print(my_model)
#summary(my_model, input_size=(3, 200, 200))
#Cargar Modelo
my_model = torch.load('/home/DLIP_user3/Proyect/ASL_my_model_V1.Hand_detection.pth')

# Train the model
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(my_model.parameters(), lr=0.001)

#Train Module
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    running_loss=0.0
    for batch, (X,y) in enumerate (dataloader):
        X, y = X.to(device),y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
        if batch%100==0:
            running_loss=running_loss/100
            current = batch*len(X)
            print(f"loss: {running_loss:>7f} [{current:>5d}/{size:>5d}]")
            runniing_loss=0
#Test Module
def test(dataloader,model,loss_fn):
    size=len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss,correctN = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            y_pred=pred.argmax(1);
            correctN += (y_pred == y).type(torch.float).sum().item()            
    test_loss /= num_batches
    correctN /= size
    print(f"Test Error: \n Accuracy: {(100*correctN):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    accuracy = (100*correctN)
    return accuracy

run_path='/home/DLIP_user3/Proyect/runs/'
if os.path.exists(run_path):
    shutil.rmtree(run_path)
os.mkdir(run_path)

last_accuracy = 0
counter = 0
#Loop
epoch = 100
model_save_path='/home/DLIP_user3/Proyect/runs/mejorModelo.pth'
for t in range(epoch):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_loader,my_model,loss_fn,optimizer)
    new_accuracy = test(test_loader,my_model,loss_fn)
    if new_accuracy > last_accuracy:
        counter += 1
        print(f"Se guardo la version {counter}% del modelo. \n")
        last_accuracy = new_accuracy
        if os.path.exists(model_save_path):
            os.remove(model_save_path)
            torch.save(my_model, model_save_path)
        else:
            torch.save(my_model, model_save_path)


