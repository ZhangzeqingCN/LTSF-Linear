import itertools

import torch
import torch.nn as nn

steps = 128

x = torch.rand(size=[3, 4, 5])
x_max = torch.max(x, dim=-1).values
x_min = torch.min(x, dim=-1).values
for b, c in itertools.product(range(x.shape[-3]), range(x.shape[-2])):
    p: torch.Tensor = x[b, c]
    p_bucketed: torch.Tensor = torch.bucketize(p, torch.linspace(min(p), max(p), steps))
    x[b, c, :] = p_bucketed

x = x.int()

print(F"{x=}")

embeds = nn.Embedding(num_embeddings=steps + 1, embedding_dim=8)
print(F"{embeds(x)=}")
