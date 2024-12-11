<center>

# Weapon Detection (Classification) - MobileNet V3

</center>

<h1 style="text-align: center;">Transfer Learning</h1>

- Transfer learning reuses pre-trained models for related tasks.
- It accelerates training, requires less data, and improves performance.
- It's effective when tasks share features or data distributions.


### Required Imports

```python
import numpy as np
import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split, KFold
import timm
import time
import pathlib
from torchvision.datasets import ImageFolder
```

### Image Processing

- Since we are dealing with images, they may have different shapes.
- Resizing all the images is necessary to train our model for better accuracy.

```python
IMAGE_SHAPE = (224, 224)  # 224*224 is the standard image size taken here.
```

### Pretrained Model - MobileNet V2

- Developed by TensorFlow

```python
class HubLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(HubLayer, self).__init__(**kwargs)
        self.hub_layer = hub.KerasLayer("https://www.kaggle.com/models/google/mobilenet-v2/TensorFlow2/tf2-preview-classification/2", input_shape=IMAGE_SHAPE+(3,))

    def call(self, inputs):
        return self.hub_layer(inputs)

# Create a Sequential model and add the custom HubLayer
classifier = tf.keras.Sequential([
    HubLayer()
])
```

<center>Example : Humming Bird</center>


```python
hmgbrd = Image.open("./mdimgs/hmgbrd.jpeg")
hmgbrd = np.array(hmgbrd.resize(IMAGE_SHAPE)) / 255.0
hmgbrd[np.newaxis, ...]   # adding a new dimension for conv

result = classifier.predict(hmgbrd[np.newaxis, ...])
predicted_label_ind = np.argmax(result)

image_labels = []
with open("datasets/ImageNetLabels1.txt", "r") as f:
    image_labels = f.read().splitlines()

image_labels[predicted_label_ind]
```

### Training Our Model using MobileNetV2

```python
data_dir = "./datasets/weapons/withoutLabel"
data_dir = pathlib.Path(data_dir)

weapons_images_dict = {
    'Handgun': list(data_dir.glob('Handgun/*')),
    'Shotgun': list(data_dir.glob('Shotgun/*')),
    'Bow and arrow': list(data_dir.glob('Bow and arrow/*')),
    'Knife': list(data_dir.glob('Knife/*')),
    'Sword': list(data_dir.glob('Sword/*')),
    'Rifle': list(data_dir.glob('Rifle/*')),
}

weapons_label_dict = {
    'Handgun': 0,
    'Shotgun': 1,
    'Bow and arrow': 2,
    'Knife': 3,
    'Sword': 4,
    'Rifle': 5,
}

X, y = [], []

for weapnName, imgs in weapons_images_dict.items():
    for img in imgs:
        img1 = cv2.imread(str(img))
        if img1 is None:
            print("Failed to parse : ", img)
            continue
        resImg = cv2.resize(img1, IMAGE_SHAPE)
        X.append(resImg)
        y.append(weapons_label_dict[weapnName])

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
Xtrain_scaled = X_train / 255
Xtest_scaled = X_test / 255

pre_trained_model = "https://www.kaggle.com/models/google/mobilenet-v2/TensorFlow2/tf2-preview-classification/2"
pretrained_model_without_top_layer = hub.KerasLayer(pre_trained_model, input_shape=(224, 224, 3), trainable=False)

model = tf.keras.Sequential([
  pretrained_model_without_top_layer,
  tf.keras.layers.Dense(6)
])

model.summary()
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
model.fit(Xtrain_scaled, y_train, epochs=120)

evaluation_metrics = model.evaluate(Xtest_scaled, y_test)

print("Evaluation Metrics:")
for metric_name, metric_value in zip(model.metrics_names, evaluation_metrics):
    print(f"{metric_name}: {metric_value}")
```

### Predicting with Trained Model

```python
IMAGE_SHAPE = (224, 224)
gun = Image.open("./datasets/download1.jpg")
gun = np.array(gun.resize(IMAGE_SHAPE)) / 255
plt.imshow(gun)

predicted = model.predict(gun[np.newaxis, ...])
predicted_label_ind = np.argmax(predicted)

image_labels = []
with open("datasets/ImageNetLabels.txt", "r") as f:
    image_labels = f.read().splitlines()

image_labels[predicted_label_ind]
```

### Training Our Model using Xception

```python
class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)  # Ensure labels are LongTensors

# Image processing and dataset preparation
IMAGE_SHAPE = (224, 224)
NUM_EPOCHS = 5
NUM_CLASSES = 7

data_dir = pathlib.Path("./datasets/weapons/withoutLabel")

weapons_images_dict = {
    "Handgun": list(data_dir.glob("Handgun/*")),
    "Shotgun": list(data_dir.glob("Shotgun/*")),
    "Bow and arrow": list(data_dir.glob("Bow and arrow/*")),
    "Knife": list(data_dir.glob("Knife/*")),
    "Sword": list(data_dir.glob("Sword/*")),
    "Rifle": list(data_dir.glob("Rifle/*")),
    "No weapons": list(data_dir.glob("No weapons/*")),
}

weapons_label_dict = {
    "Handgun": 0,
    "Shotgun": 1,
    "Bow and arrow": 2,
    "Knife": 3,
    "Sword": 4,
    "Rifle": 5,
    "No weapons": 6,
}

image_paths, labels = [], []

for weapon_name, imgs in weapons_images_dict.items():
    for img in imgs:
        image_paths.append(str(img))
        labels.append(weapons_label_dict[weapon_name])

transform = transforms.Compose([
    transforms.Resize(IMAGE_SHAPE),
    transforms.ToTensor(),
])

dataset = ImageFolder(".\\datasets\\weapons\\withoutLabel", transform=transform)
kf = KFold(n_splits=5, shuffle=True, random_state=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for fold, (train_index, val_index) in enumerate(kf.split(dataset)):
    print(f"Training fold {fold+1}...")

    train_subsampler = torch.utils.data.SubsetRandomSampler(train_index)
    val_subsampler = torch.utils.data.SubsetRandomSampler(val_index)

    train_loader = DataLoader(dataset, batch_size=32, sampler=train_subsampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(dataset, batch_size=32, sampler=val_subsampler, num_workers=4, pin_memory=True)

    model = timm.create_model('xception', pretrained=True, num_classes=len(weapons_label_dict))
    model = model.to(device, non_blocking=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()  # Ensure labels are LongTensors

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).long()  # Ensure labels are LongTensors

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Epoch {epoch+1}, Loss:

 {running_loss/len(train_loader)}, Validation Loss: {val_loss/len(val_loader)}, Accuracy: {100 * correct / total}")

    # Save the model for each fold
    torch.save(model.state_dict(), f"xception_fold{fold+1}.pth")
```

### Loading the Model and Predicting

```python
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import timm

# Define the class labels
weapons_label_dict = {
    0: "Handgun",
    1: "Shotgun",
    2: "Bow and arrow",
    3: "Knife",
    4: "Sword",
    5: "Rifle",
    6: "No weapons",
}

# Function to load the model
def load_model(model_path, num_classes=7):
    model = timm.create_model('xception', pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set model to evaluation mode
    return model

# Function to preprocess the input image
def preprocess_image(image_path, image_shape=(224, 224)):
    transform = transforms.Compose([
        transforms.Resize(image_shape),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Function to predict the class of the image
def predict_image(model, image_tensor, device):
    image_tensor = image_tensor.to(device)
    outputs = model(image_tensor)
    _, predicted = torch.max(outputs.data, 1)
    return predicted.item()

# Load the model
model_path = 'xception_fold4.pth'  # Replace with the path to your saved model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(model_path)
model = model.to(device)

# Preprocess the image
image_path = './testt/test4.jpeg'  # Replace with the path to the image you want to predict
image_tensor = preprocess_image(image_path)

# Predict the image
predicted_class_idx = predict_image(model, image_tensor, device)
predicted_class = weapons_label_dict[predicted_class_idx]

print(f"The predicted class is: {predicted_class}")
```

### References

- [Transfer Learning with Keras](https://keras.io/guides/transfer_learning/)
