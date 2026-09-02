import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BATCH_SIZE = 64  # Instead of feeding all 60,000 training images to the neural network simultaneously, we process them in groups of 64
EPOCHS = 3  # Number of times the entire dataset is passed through the network during training (The model has seen the entire training dataset once after 1 epoch, twice after 2 epochs, and so on)
LEARNING_RATE = 0.001  # Step size for the optimizer. This controls how aggressively Adam modifies the model's parameters


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)  # If a CUDA GPU exists, use it. Otherwise use the CPU

print("=" * 60)
print("FedTrust-K8s — Experiment 00: PyTorch Baseline")
print("=" * 60)
print(f"PyTorch version : {torch.__version__}")
print(f"Device          : {device}")
print(f"CPU threads     : {torch.get_num_threads()}")
print("=" * 60)


# --------------------------------------------------
# Dataset
# --------------------------------------------------

transform = transforms.ToTensor()

#  loading datasets (images)
train_dataset = datasets.MNIST(
    root="./datasets",
    train=True,
    download=True,
    transform=transform,
)

test_dataset = datasets.MNIST(
    root="./datasets",
    train=False,
    download=True,
    transform=transform,
)

# handles the training batches
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,  # The training examples are randomized between epochs.
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# --------------------------------------------------
# Model
# We're creating a Convolutional Neural Network.
# CNNs are particularly useful for image data.
# --------------------------------------------------


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # 16 feature maps
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 32 feature maps
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(
                32 * 7 * 7, 128
            ),  # This takes the extracted features and maps them into: 128 neurons
            nn.ReLU(),  # adds non-linearity again.
            nn.Linear(128, 10),  # 10 (0-9) digits
        )

    def forward(self, x):
        return self.network(x)


model = SimpleCNN().to(device)

criterion = nn.CrossEntropyLoss()

# The optimizer changes the neural network's parameters
optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


# --------------------------------------------------
# Training
# We perform three epochs

# --------------------------------------------------

for epoch in range(EPOCHS):
    model.train()

    total_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        # The images go through the neural network

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)

    print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {average_loss:.4f}")


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        total += labels.size(0)

        correct += (predictions == labels).sum().item()


accuracy = 100 * correct / total

print("=" * 60)
print(f"Test accuracy: {accuracy:.2f}%")
print("=" * 60)
