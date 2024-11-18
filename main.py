import torch
import torch.nn as nn

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