from torch.masked import masked_tensor
import torch
import warnings

# Disable prototype warnings and such
warnings.filterwarnings(action='ignore', category=UserWarning)

values = torch.tensor([[0, 0, 0], [0, 0, 0]])
mask = torch.tensor([[False, False, True], [False, False, True]])
mt = masked_tensor(values, mask)

print(mt)