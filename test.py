import flash_attention
import torch
import torch.nn.functional as F
import math


def pad_last_dim(x: torch.Tensor, target_size: int, value: float = 0.0):
    """Pad the last dimension of `x` to `target_size` with `value`."""
    curr_size = x.size(-1)
    if curr_size >= target_size:
        return x  # no padding needed
    pad_right = target_size - curr_size
    return F.pad(x, (0, pad_right), mode='constant', value=value)

torch.set_printoptions(threshold=float('inf'))

def attention_ref(query, key, value, scale):
    d = query.size(-1)
    scores = torch.einsum('bhqd,bhkd->bhqk', query.to(torch.float32) * scale, key.to(torch.float32))
    attn_weights = torch.softmax(scores, dim=-1).to(value.dtype)
    output = torch.einsum('bhqk,bhkd->bhqd', attn_weights, value)
    return output, scores.to(torch.float32)

query = torch.load("./query.pt", map_location=torch.device('xpu:0'))
key = torch.load("./key.pt", map_location=torch.device('xpu:0'))
value = torch.load("./value.pt", map_location=torch.device('xpu:0'))
scale = torch.load("./scale.pt", map_location=torch.device('xpu:0'))

query = torch.rand_like(query) * 10
key = torch.rand_like(key) * 10
value = torch.rand_like(value) * 10

head_dim = 88
seq_len = 512
scale = 1.0
batch_size = 1
num_heads = 1

query = torch.ones(batch_size, num_heads, seq_len, head_dim, dtype=query.dtype).to('xpu:0')
key =   torch.ones(batch_size, num_heads, seq_len, head_dim, dtype=key.dtype).to('xpu:0')
value = torch.rand(batch_size, num_heads, seq_len, head_dim, dtype=value.dtype).to('xpu:0')

print(f"query shape: {query.shape}, stride: {query.stride()}")
print(f"key shape: {key.shape}, stride: {key.stride()}")
print(f"value shape: {value.shape}, stride: {value.stride()}")

iter_num = 1

query = pad_last_dim(query, 128)
key = pad_last_dim(key, 128)
output_ref, scores_ref = attention_ref(query, key, value, scale)

for i in range(iter_num):
    output = F.scaled_dot_product_attention(query, key, value, scale=scale)

diff_ref = output_ref - output
print(f"ref diff max: {diff_ref.max()}")

for i in range(iter_num):
    ipex_output, debug_output = flash_attention.run(query, key, value, scale)

print(f"output_ref shape: {output_ref.shape}, output shape: {output.shape}, ipex_output shape: {ipex_output.shape}")

s_diff = (scores_ref - debug_output).abs().max()
print(f"scores diff max: {s_diff}")
print(f"ipex output: {debug_output[0, 0, :10, :10]}")
print(f"ref output: {scores_ref[0, 0, :10, :10]}")
print(f"debug output value max and min: {debug_output.max()}, {debug_output.min()}")
exit(0)
# debug_mask = (debug_output == 0)
# pos = torch.nonzero(debug_mask & (scores_ref != 0), as_tuple=False)
# print(f"zero scores pos: {pos}")

query = query.to('cpu')
key = key.to('cpu')
value = value.to('cpu')
cpu_output = F.scaled_dot_product_attention(query, key, value, scale=scale)

diff_torch = cpu_output - output.to('cpu')
diff_ipex = cpu_output - ipex_output.to('cpu')

print(f"query shape: {query.shape}, key shape: {key.shape}, value shape: {value.shape}")
print(f"torch diff max: {diff_torch.max()}, ipex diff max: {diff_ipex.max()}")
