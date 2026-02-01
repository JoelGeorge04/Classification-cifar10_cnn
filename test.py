import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Model definition (same as training)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16 * 16 * 16, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# CIFAR-10 class labels
classes = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = SimpleCNN().to(device)
model.load_state_dict(
    torch.load("model.pth", map_location=device, weights_only=True)
)
model.eval()

# Image transformations (same as training)
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

# Load image
image_path = "sample.jpg" 
image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0).to(device)

# Prediction
with torch.no_grad():
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)

print("Predicted class:", classes[predicted.item()])
