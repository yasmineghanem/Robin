import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"My device is: {device}")
