
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import copy

# Hyperparameters
NUM_CLIENTS = 5
EPOCHS = 5
BATCH_SIZE = 32
LR = 0.01

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset and DataLoader
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)

# Split dataset among clients
client_datasets = random_split(dataset, [len(dataset) // NUM_CLIENTS] * NUM_CLIENTS)

# Global Model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

global_model = SimpleNN().to(device)

# Federated Training
def train_model(client_model, dataloader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(client_model.parameters(), lr=LR)

    client_model.train()
    for epoch in range(EPOCHS):
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = client_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    return client_model.state_dict()

def federated_averaging(client_states):
    avg_state = copy.deepcopy(client_states[0])
    for key in avg_state.keys():
        for i in range(1, len(client_states)):
            avg_state[key] += client_states[i][key]
        avg_state[key] = avg_state[key] / len(client_states)
    return avg_state

# Training loop
for round in range(1, 6):
    client_states = []

    for client_id, client_dataset in enumerate(client_datasets):
        client_loader = DataLoader(client_dataset, batch_size=BATCH_SIZE, shuffle=True)
        client_model = copy.deepcopy(global_model)
        client_model.load_state_dict(global_model.state_dict())
        client_states.append(train_model(client_model, client_loader))

    # Aggregate client models
    global_model.load_state_dict(federated_averaging(client_states))

# Evaluate global model
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

def evaluate_model(model, dataloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

accuracy = evaluate_model(global_model, test_loader)
print(f"Global Model Accuracy: {accuracy * 100:.2f}%")
