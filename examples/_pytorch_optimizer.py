import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from gradient_free_optimizers import (
    HillClimbingOptimizer,
    RandomSearchOptimizer,
    RepulsingHillClimbingOptimizer,
    PowellsConjugateDirectionMethod,
)

# Define a synthetic dataset
# np.random.seed(42)
X = np.random.rand(1000, 20)
true_weights = np.random.rand(20, 1)
y = X @ true_weights + 0.1 * np.random.randn(1000, 1)

X = torch.Tensor(X)
y = torch.Tensor(y)

# Create a DataLoader
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)


num_epochs = 10


# Define a more complex neural network
class ComplexModel(nn.Module):
    def __init__(self):
        super(ComplexModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.network(x)


# Initialize the model
model = ComplexModel()

# Define a loss function
criterion = nn.MSELoss()


# Define the custom optimizer with GFO
class GFOOptimizer(torch.optim.Optimizer):
    def __init__(self, params, model, dataloader, criterion, lr=1e-3):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.lr = lr

        # Flatten the initial model parameters
        self.initial_weights = {}
        self.params = []
        counter = 0
        for param in self.model.parameters():
            self.params.extend(param.data.cpu().numpy().flatten())

            for value in param.data.flatten():
                self.initial_weights[f"x{counter}"] = (
                    value.item()
                )  # Convert tensor value to Python scalar
                counter += 1

        # Define the search space
        self.search_space = {
            f"x{i}": np.arange(-1.0, 1.0, 0.1, dtype=np.float32)
            for i in range(len(self.params))
        }

        # Initialize the GFO optimizer
        self.optimizer = HillClimbingOptimizer(
            self.search_space, initialize={"warm_start": [self.initial_weights]}
        )

        self.optimizer.init_search(
            objective_function=self.objective_function,
            n_iter=num_epochs * len(dataloader),
            max_time=None,
            max_score=None,
            early_stopping=None,
            memory=True,
            memory_warm_start=None,
            verbosity=[],
        )

        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    def objective_function(self, opt_params):
        opt_params_l = list(opt_params.values())

        # Set model parameters
        start = 0
        for param in self.model.parameters():
            param_length = param.numel()
            param.data = torch.tensor(
                opt_params_l[start : start + param_length]
            ).view(param.shape)
            start += param_length

        # Compute the loss
        total_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in self.dataloader:
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()
        return total_loss / len(self.dataloader)

    def step(self, closure=None):
        if closure is not None:
            closure()

        # Use GFO to find the best parameters
        self.optimizer.search_step()
        best_params = self.optimizer.pos_new

        print("self.optimizer.score_new", self.optimizer.score_new)

        # Set the best parameters to the model
        start = 0
        for param in self.model.parameters():
            param_length = param.numel()
            """
            param.data.copy_(
                torch.tensor(
                    best_params[start : start + param_length],
                    dtype=torch.float32,
                ).view(param.shape)
            )
            """
            start += param_length

        self.params = best_params


# Initialize the custom optimizer
optimizer = GFOOptimizer(
    model.parameters(), model, dataloader, criterion, lr=0.01
)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)


# Training loop
for epoch in range(num_epochs):
    for batch_X, batch_y in dataloader:
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        # Backward pass
        loss.backward()

        # Update the weights
        optimizer.step()

    # Print the loss for every epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

print("Training completed!")
