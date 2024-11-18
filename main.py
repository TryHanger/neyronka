import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

<<<<<<< Updated upstream
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create the model
model = SimpleNet()

# Define a loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Dummy data for training (e.g., learning to count)
inputs = torch.tensor([[i] for i in range(10)], dtype=torch.float32)
targets = torch.tensor([[i] for i in range(10)], dtype=torch.float32)

# Training loop
for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Эпоха [{epoch + 1}/1000], Потеря: {loss.item():.4f}')

# Test the model
test_input = torch.tensor([[10]], dtype=torch.float32)
predicted = model(test_input).item()
print(f'Предсказанное значение для входа 10: {predicted}')
=======
# 1. Загрузка и подготовка данных
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Загрузить данные для обучения и тестирования
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 2. Создание модели нейросети
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)

# 3. Создание модели, функции потерь и оптимизатора
model = Net()
criterion = nn.CrossEntropyLoss()  # Для многоклассовой классификации
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Обучение модели
epochs = 5
for epoch in range(epochs):
    model.train()
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
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100 * correct/total:.2f}%')

# 5. Оценка модели
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on test data: {100 * correct / total:.2f}%')

# 6. Предсказания
# Вывод одного примера
sample_image, sample_label = test_dataset[0]
model.eval()
with torch.no_grad():
    output = model(sample_image.unsqueeze(0))
    predicted_label = torch.argmax(output, dim=1).item()

# Отобразим изображение
plt.imshow(sample_image.squeeze(), cmap='gray')
plt.title(f'Predicted: {predicted_label}')
plt.show()
>>>>>>> Stashed changes
