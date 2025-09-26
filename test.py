import flash_attention
import torch
import torch.nn.functional as F
import math

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

head_dim = 128

query =10 * torch.rand(query.shape[0], query.shape[1], query.shape[2], head_dim, dtype=query.dtype).to('xpu:0')
key =  10 * torch.rand(key.shape[0], key.shape[1], key.shape[2], head_dim, dtype=key.dtype).to('xpu:0')
value =10 * torch.rand(value.shape[0], value.shape[1], value.shape[2], head_dim, dtype=value.dtype).to('xpu:0')

print(f"query shape: {query.shape}, stride: {query.stride()}")
print(f"key shape: {key.shape}, stride: {key.stride()}")
print(f"value shape: {value.shape}, stride: {value.stride()}")

iter_num = 1

output_ref, scores_ref = attention_ref(query, key, value, scale)

for i in range(iter_num):
    output = F.scaled_dot_product_attention(query, key, value, scale=scale)

diff_ref = output_ref - output
print(f"ref diff max: {diff_ref.max()}")

for i in range(iter_num):
    ipex_output, debug_output = flash_attention.run(query, key, value, scale)

print(f"scores diff max: {(scores_ref - debug_output).abs().max()}")

query = query.to('cpu')
key = key.to('cpu')
value = value.to('cpu')
cpu_output = F.scaled_dot_product_attention(query, key, value, scale=scale)

diff_torch = cpu_output - output.to('cpu')
diff_ipex = cpu_output - ipex_output.to('cpu')

print(f"query shape: {query.shape}, key shape: {key.shape}, value shape: {value.shape}")
print(f"torch diff max: {diff_torch.max()}, ipex diff max: {diff_ipex.max()}")
