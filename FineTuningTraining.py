import torch
from torch.utils.data import DataLoader
from torch import optim
from torchvision import datasets, transforms, models
data_dir = 'HandsOnlyColoredFinal'

#Prepare data augmentations and transformations
transform = transforms.Compose([
    transforms.RandomGrayscale(p=0.2),  # 20% Probability of grayscaling, useful in reducing focus on colors
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),  # Color transformation to generalize the model for different hand colors and lightning enviroments
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.Resize((224, 224)),  # Resize images to fit ResNet input dimensions
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize images as per ImageNet standards
    transforms.Lambda(lambda x: x + 0.1 * torch.randn_like(x))  # Noise injection
])
dataset = datasets.ImageFolder(data_dir, transform=transform)

# Split dataset into 80% training and 20% validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
# Load a pre-trained ResNet18 model
model = models.resnet18(pretrained=True)

# Modify the final fully connected layer to match the number of classes
features_num = model.fc.in_features
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.2), #Add a dropout of 20% for more generalization of the model
    torch.nn.Linear(features_num, len(dataset.classes))
)
#Cross entropy loss is widely used in categorical classification along with Adam optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epochs = 12
model.to(device)

for epoch in range(epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        # Apply log-softmax to the outputs for a greater punishment on high errors classifications
        log_probs = torch.nn.functional.log_softmax(outputs, dim=1)

        # Calculate loss
        loss = torch.nn.NLLLoss()(log_probs, labels)
        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    #Print current epoch to track model progress
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')

# Set the model to evaluation mode
model.eval()
correct = 0
total = 0
#Test the model and score number of correct answers
with torch.no_grad():
    for inputs, labels in validation_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

#Print the model's performance and save the model
print(f'Accuracy on validation set: {100 * correct / total}%')
print("Process Done!")
torch.save(model, "torchmodel.pth")