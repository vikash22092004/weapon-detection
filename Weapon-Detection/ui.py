import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import timm
import torch
import torchvision.transforms as transforms


# Load MobileNetV2 model
def load_modelMobileNet(model_path, num_classes=7):
    model = timm.create_model(
        "mobilenetv2_100", pretrained=False, num_classes=num_classes
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()  # Set model to evaluation mode
    return model


# Load Xception model
def load_modelXception(model_path, num_classes=7):
    model = timm.create_model("xception", pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()  # Set model to evaluation mode
    return model


def preprocess_imageXception(image_path, image_shape=(224, 224)):
    transform = transforms.Compose(
        [
            transforms.Resize(image_shape),
            transforms.ToTensor(),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image


def predict_imageXception(model, image_tensor, device):
    image_tensor = image_tensor.to(device)
    outputs = model(image_tensor)
    _, predicted = torch.max(outputs.data, 1)
    return predicted.item()


# Load models once at the start
mobilenet_model = load_modelMobileNet("mobilenetv2_fold3.pth")
# mobilenet_model = load_modelMobileNet("xception_fold3.pth")
devicee = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mobilenet_model = mobilenet_model.to(devicee)

xception_model = load_modelXception("xception_fold4.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
xception_model = xception_model.to(device)

# Function to handle model selection and open a new window
current_model_window = None


def select_model(model_name):
    global current_model, current_model_window
    current_model = model_name

    # Close the currently open model window, if any
    if current_model_window is not None:
        current_model_window.destroy()

    model_window = tk.Toplevel(root)
    model_window.title(f"{model_name} Model - Image Uploader")
    model_window.configure(bg="#282c34")

    current_model_window = model_window

    def upload_image():
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.gif;*.bmp")]
        )
        if file_path:
            img = Image.open(file_path)
            img.thumbnail((400, 400))  # Resize the image to fit the display area
            img_display = ImageTk.PhotoImage(img)
            image_label.config(image=img_display)
            image_label.image = (
                img_display  # Keep a reference to avoid garbage collection
            )
            predict_image(file_path)

    def predict_image(file_path):
        IMAGE_SHAPE = (224, 224)
        if current_model == "MobileNetV2":
            image_tensor = preprocess_imageXception(file_path)
            predicted_label_ind = predict_imageXception(
                mobilenet_model, image_tensor, devicee
            )
            image_labels = []
            with open("datasets/ImageNetLabels.txt", "r") as f:
                image_labels = f.read().splitlines()
            predicted_label = mobilenet(file_path)
        elif current_model == "Xception":
            weapons_label_dict = {
                0: "Handgun",
                1: "Shotgun",
                2: "Bow and arrow",
                3: "Knife",
                4: "Sword",
                5: "Rifle",
                6: "No weapons",
            }
            image_tensor = preprocess_imageXception(file_path)
            predicted_label_ind = predict_imageXception(
                xception_model, image_tensor, device
            )
            predicted_label = weapons_label_dict[predicted_label_ind]

        result_label.config(text=f"Predicted: {predicted_label}")

    def mobilenet(file_path):
        weapons_label_dict = {
            0: "Handgun",
            1: "Shotgun",
            2: "Bow and arrow",
            3: "Knife",
            4: "Sword",
            5: "Rifle",
            6: "No weapons",
        }
        image_tensor = preprocess_imageXception(file_path)
        predicted_label_ind = predict_imageXception(
            xception_model, image_tensor, device
        )
        predicted_label = weapons_label_dict[predicted_label_ind]
        return predicted_label

    upload_button = tk.Button(
        model_window,
        text="Upload Image",
        command=upload_image,
        bg="#61afef",
        fg="#282c34",
        font=("Helvetica", 12, "bold"),
    )
    upload_button.pack(pady=10)

    global image_label
    image_label = tk.Label(model_window, bg="#282c34")
    image_label.pack(pady=10)

    global result_label
    result_label = tk.Label(
        model_window,
        text="Predicted: None",
        bg="#abb2bf",
        fg="#282c34",
        font=("Helvetica", 16),
    )
    result_label.pack(pady=10, fill=tk.BOTH, expand=True)


# Create the main window
root = tk.Tk()
root.title("Model Selector")
root.configure(bg="#282c34")

# Create a canvas
canvas = tk.Canvas(root, width=300, height=200, bg="#282c34", highlightthickness=0)
canvas.pack()

# Add buttons to the canvas
button_mobilenet = tk.Button(
    root,
    text="MobileNetV2",
    command=lambda: select_model("MobileNetV2"),
    bg="#98c379",
    fg="#282c34",
    font=("Helvetica", 12, "bold"),
)
button_mobilenet_window = canvas.create_window(
    75, 100, anchor="center", window=button_mobilenet
)

button_xception = tk.Button(
    root,
    text="Xception",
    command=lambda: select_model("Xception"),
    bg="#e06c75",
    fg="#282c34",
    font=("Helvetica", 12, "bold"),
)
button_xception_window = canvas.create_window(
    225, 100, anchor="center", window=button_xception
)

# Run the Tkinter event loop
root.mainloop()
