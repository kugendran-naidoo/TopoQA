# don’t torch.load untrusted files—pickle can execute code on load.

import torch

ckpt = torch.load("model/topoqa.ckpt", map_location="cpu")

# If it’s a raw state_dict:
keys = list(ckpt.keys())

# If it’s a Lightning checkpoint, weights are usually under 'state_dict'
sd = ckpt.get("state_dict", ckpt)

print(len(sd), list(sd)[:10])  # number of params and a peek at first keys
