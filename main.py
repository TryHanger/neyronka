import torch
import torch.nn as nn

class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 1)
porno = 12
sex = 13
if porno > sex:
    print("penis")
else:
    print("Loh")
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Check if CUDA is available and use it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TestNet().to(device)
# Example of a simple mathematical operation
a = torch.tensor([2.0], device=device)
b = torch.tensor([3.0], device=device)
c = a + b
print(f"The result of adding {a.item()} and {b.item()} is {c.item()}")
# Example input tensor
input_tensor = torch.randn(1, 10).to(device)
output = model(input_tensor)
print(output)

print("Hello World!")
a = 5
for i in range(a):
    print(i) 