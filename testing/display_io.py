import torch

use_cuda = False
device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

input: torch.Tensor = torch.load("models.Linear-Input", map_location=device)
output: torch.Tensor = torch.load("models.Linear-Output", map_location=device)

print(F"{input.shape=}")
print(F"{output.shape=}")
