import torch
from homework.low_precision import Linear4Bit

# Set up reproducibility
torch.manual_seed(42)

# Parameters
in_features = 8
out_features = 4

# Create a random input
x = torch.randn(2, in_features)

# Create a standard Linear layer
linear = torch.nn.Linear(in_features, out_features, bias=True)

# Create a Linear4Bit layer
linear4bit = Linear4Bit(in_features, out_features, bias=True, group_size=4)

# Copy weights and bias from linear to linear4bit
with torch.no_grad():
    linear4bit._load_state_dict_pre_hook({'weight': linear.weight.data.clone(), 'bias': linear.bias.data.clone()}, '', None, True, [], [], [])
    if linear4bit.bias is not None:
        linear4bit.bias.copy_(linear.bias.data)

# Forward pass
out_linear = linear(x)
out_4bit = linear4bit(x)

# Compare outputs
print('Standard Linear output:', out_linear)
print('Linear4Bit output:', out_4bit)
print('Mean absolute difference:', (out_linear - out_4bit).abs().mean().item())
