import torch
from lib import paged_attention_loop
from lib import paged_attention_reduce
import pytest

import math

import pytest
import torch.nn.functional as F
from einops import rearrange, repeat

torch.set_printoptions(threshold=float('inf'))


def assert_close_verbose(a, b, rtol=1e-5, atol=1e-8):
    if a.shape != b.shape:
        raise AssertionError(f"Shape mismatch: {a.shape} vs {b.shape}")

    # Compute absolute and relative differences
    diff = torch.abs(a - b)
    tol = atol + rtol * torch.abs(b)

    # Find mismatches
    mismatch_mask = diff > tol
    if mismatch_mask.any():
        idx = mismatch_mask.nonzero(as_tuple=False)
        for i in idx:
            coord = tuple(i.tolist())
            val_a = a[coord].item()
            val_b = b[coord].item()
            d = diff[coord].item()
            t = tol[coord].item()
            print(f"Mismatch at {coord}: a={val_a}, b={val_b}, diff={d}, tol={t}")
        # raise AssertionError(f"Tensors are not close. {mismatch_mask.sum().item()} elements differ.")
    else:
        print("Tensors are close within tolerance.")

def attention_ref_fp8kv(
    q,
    k,
    v,
    s,
    query_padding_mask=None,
    key_padding_mask=None,
    attn_bias=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
    softcap=0.0,
    upcast=True,
    reorder_ops=False,
    key_leftpad=None,
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k: (batch_size, seqlen_k, nheads_k, head_dim)
        v: (batch_size, seqlen_k, nheads_k, head_dim)
        s: (nheads)
        query_padding_mask: (batch_size, seqlen_q)
        key_padding_mask: (batch_size, seqlen_k)
        attn_bias: broadcastable to (batch_size, nheads, seqlen_q, seqlen_k)
        dropout_p: float
        dropout_mask: (batch_size, nheads, seqlen_q, seqlen_k)
        causal: whether to apply causal masking
        window_size: (int, int), left and right window size
        upcast: whether to cast all inputs to fp32, do all computation in fp32, then cast
            output back to fp16/bf16.
        reorder_ops: whether to change the order of operations (scaling k instead of scaling q, etc.)
            without changing the math. This is to estimate the numerical error from operation
            reordering.
    Output:
        output: (batch_size, seqlen_q, nheads, head_dim)
        attention: (batch_size, nheads, seqlen_q, seqlen_k), softmax after dropout
    """
    if causal:
        window_size = (window_size[0], 0)
    dtype_og = q.dtype
    k = k.to(dtype_og)
    v = v.to(dtype_og)
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    d = q.shape[-1]
    if not reorder_ops:
        scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
    else:
        scores = torch.einsum("bthd,bshd->bhts", q, k / math.sqrt(d))
    if softcap > 0:
        scores = scores / softcap
        scores = scores.tanh()
        scores = scores * softcap
    if key_padding_mask is not None:
        scores.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            q.device,
            key_leftpad=key_leftpad,
        )
        scores.masked_fill_(local_mask, float("-inf"))
    if attn_bias is not None:
        scores = scores + attn_bias
    # attention = torch.softmax(scores, dim=-1).to(v.dtype)
    scores_fp32 = scores.to(torch.float32)
    scores_max = torch.max(scores_fp32, dim=-1, keepdim=True)
    if s is not None:
        learnable_sinks = rearrange(s, "h -> 1 h 1 1")
        logits_or_sinks_max = torch.maximum(learnable_sinks, scores_max.values)
        unnormalized_scores = torch.exp(scores_fp32 - logits_or_sinks_max)
        scores_exp_sum = unnormalized_scores.sum(dim=-1, keepdim=True) + torch.exp(learnable_sinks - logits_or_sinks_max)
        attention = (unnormalized_scores / scores_exp_sum).to(v.dtype)
    else:
        scores_exp = torch.exp(scores_fp32 - scores_max.values)
        scores_exp_sum = torch.sum(scores_exp, dim=-1, keepdim=True)
        attention = (scores_exp / scores_exp_sum).to(v.dtype)
    # for debugging
    partition_size = 512
    num_partitions = int((seqlen_k + partition_size - 1) / partition_size)
    old_max = torch.full_like(scores_max.values, float("-inf"))
    old_sum = torch.zeros_like(scores_exp_sum)
    old_output = torch.zeros_like(q).to(dtype=torch.float32)
    out_partitions = torch.zeros((q.shape[0], q.shape[1], q.shape[2], num_partitions * q.shape[3]), dtype=torch.float32, device=q.device)
    # for i in range(num_partitions):
    #     scores_part = scores[:, :, :, i * partition_size : (i + 1) * partition_size]
    #     scores_max_part = torch.max(scores_part, dim=-1, keepdim=True).values
    #     cur_max = torch.max(old_max, scores_max_part)
    #     attn_part = torch.exp(scores_part - cur_max)
    #     rescale_factors = torch.exp(old_max - cur_max)
    #     cur_sum = rescale_factors * old_sum + torch.sum(attn_part, dim=-1, keepdim=True)
    #     old_max = cur_max
    #     old_sum = cur_sum
    #     # print(f"rescale_factors: {rescale_factors} \n cur_sum: {cur_sum}")
    #     out_par = torch.einsum("bhts,bshd->bthd", (attn_part.to(v.dtype)), v[:, i * partition_size : (i + 1) * partition_size])
    #     debug_rescale = torch.ones_like(rescale_factors)
    #     out_par += old_output * rescale_factors.permute(0, 2, 1, 3)
    #     head_size = q.shape[-1]
    #     out_partitions[:, :, :, i * head_size : (i + 1) * head_size] = old_output * rescale_factors.permute(0, 2, 1, 3) # for debugging
    #     old_output = out_par

    # output_p = old_output / old_sum.permute(0, 2, 1, 3)
    # output_p = old_output

    # Some rows might be completely masked out so we fill them with zero instead of NaN
    if window_size[0] >= 0 or window_size[1] >= 0:
        attention = attention.masked_fill(torch.all(local_mask, dim=-1, keepdim=True), 0.0)
    # We want to mask here so that the attention matrix doesn't have any NaNs
    # Otherwise we'll get NaN in dV
    if query_padding_mask is not None:
        attention = attention.masked_fill(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    dropout_scaling = 1.0 / (1 - dropout_p)
    # attention_drop = attention.masked_fill(~dropout_mask, 0.0) * dropout_scaling
    # output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    if dropout_mask is not None:
        attention_drop = attention.masked_fill(~dropout_mask, 0.0)
    else:
        attention_drop = attention
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)
    if query_padding_mask is not None:
        output.masked_fill_(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)
    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og), out_partitions


def _generate_block_fp8kvcache(seqlen_k,
                               paged_kv_block_size,
                               batch_size,
                               nheads_k,
                               d, device,
                               dtype,
                               fp8_dtype):
    # num_blocks = math.ceil(seqlen_k / paged_kv_block_size) * batch_size * 3
    num_blocks = math.ceil(seqlen_k / paged_kv_block_size) * batch_size
    k_cache_paged = torch.randn(
        num_blocks, paged_kv_block_size, nheads_k, d, device=device, dtype=dtype
    )
    v_cache_paged = torch.randn(
        num_blocks, paged_kv_block_size, nheads_k, d, device=device, dtype=dtype
    )
    block_table = rearrange(
        torch.randperm(num_blocks, dtype=torch.int32, device=device),
        "(b nblocks) -> b nblocks",
        b=batch_size,
    )
    k_cache = rearrange(
        k_cache_paged[block_table.to(dtype=torch.long).flatten()],
        "(b nblocks) block_size ... -> b (nblocks block_size) ...",
        b=batch_size,
    )[:, :seqlen_k]
    v_cache = rearrange(
        v_cache_paged[block_table.to(dtype=torch.long).flatten()],
        "(b nblocks) block_size ... -> b (nblocks block_size) ...",
        b=batch_size,
    )[:, :seqlen_k]
    k_cache_paged_fp8 = k_cache_paged.to(fp8_dtype)
    k_cache_fp8 = k_cache.to(fp8_dtype)
    return k_cache.to(fp8_dtype), v_cache.to(fp8_dtype), block_table, k_cache_paged.to(fp8_dtype), v_cache_paged.to(fp8_dtype), num_blocks

def generate_random_padding_mask(max_seqlen, batch_size, device, mode="random"):
    assert mode in ["full", "random", "third"]
    if mode == "full":
        lengths = torch.full((batch_size, 1), max_seqlen, device=device, dtype=torch.int32)
    elif mode == "random":
        lengths = torch.randint(
            max(1, max_seqlen - 20), max_seqlen + 1, (batch_size, 1), device=device
        )
    elif mode == "third":
        lengths = torch.randint(max_seqlen // 3, max_seqlen + 1, (batch_size, 1), device=device)
    padding_mask = (
        repeat(torch.arange(max_seqlen, device=device), "s -> b s", b=batch_size) < lengths
    )
    return padding_mask, lengths

@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("num_splits", [1])
@pytest.mark.parametrize("mha_type", ["gqa"])
@pytest.mark.parametrize("new_kv", [False])
@pytest.mark.parametrize("alibi", [False])
@pytest.mark.parametrize("local", [False])
@pytest.mark.parametrize("causal", [False])
@pytest.mark.parametrize("seqlen_new_eq_seqlen_q", [True])
@pytest.mark.parametrize("has_learnable_sinks", [True, False])
@pytest.mark.parametrize("rotary_interleaved", [False])
@pytest.mark.parametrize("rotary_fraction", [0.0])
@pytest.mark.parametrize("paged_kv_block_size", [64])
@pytest.mark.parametrize("has_leftpad", [False])
@pytest.mark.parametrize("has_batch_idx", [False])
@pytest.mark.parametrize("varlen", [False, True])
@pytest.mark.parametrize("d", [128, 64])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (1, 128),
        (1, 339),
        (1, 1024),
        (1, 800),
        (1, 256),
        (1, 799),
        (1, 2048),
        (1, 20000),
        (1, 40000),
        (1, 50000),
    ],
)
@pytest.mark.parametrize(
    "nheads,nheads_k",
    [
        (8, 1),
        (16, 2),
        (10, 2),
        (14, 2),
        (12, 2),
        (8, 2),
        (6, 2),
        (4, 2),
        (2, 2),
    ],
)
def test_flash_attn_kvcache(
    batch_size,
    nheads,
    nheads_k,
    seqlen_q,
    seqlen_k,
    d,
    has_batch_idx,
    has_leftpad,
    paged_kv_block_size,
    rotary_fraction,
    rotary_interleaved,
    seqlen_new_eq_seqlen_q,
    causal,
    local,
    alibi,
    new_kv,
    has_learnable_sinks,
    mha_type,
    num_splits,
    dtype,
    varlen,
    do_performance=False,
    fp8_dtype=torch.float8_e5m2,
):
    if seqlen_q > seqlen_k and new_kv:
        pytest.skip()
    if not new_kv and rotary_fraction > 0.0:
        pytest.skip()
    if has_batch_idx and paged_kv_block_size is not None:
        pytest.skip()
    if has_leftpad and paged_kv_block_size is not None:
        pytest.skip()
    device = "xpu"
    # set seed
    torch.random.manual_seed(0)
    # batch_size = 2
    batch_size_cache = batch_size if not has_batch_idx else batch_size * 2
    # rotary_dim must be a multiple of 16, and must be <= d
    rotary_dim = math.floor(int(rotary_fraction * d) / 16) * 16
    assert nheads % nheads_k == 0
    window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)
    seqlen_new = seqlen_q if seqlen_new_eq_seqlen_q else torch.randint(1, seqlen_q + 1, (1,)).item()
    if new_kv:
        k = torch.randn(batch_size, seqlen_new, nheads_k, d, device=device, dtype=dtype)
        v = torch.randn(batch_size, seqlen_new, nheads_k, d, device=device, dtype=dtype)
    else:
        k, v = None, None
    if paged_kv_block_size is None:
        k_cache = torch.randn(batch_size_cache, seqlen_k, nheads_k, d, device=device, dtype=dtype)
        v_cache = torch.randn(batch_size_cache, seqlen_k, nheads_k, d, device=device, dtype=dtype)
        block_table = None
    else:
        (
            k_cache,
            v_cache,
            block_table,
            k_cache_paged,
            v_cache_paged,
            num_blocks,
        ) = _generate_block_fp8kvcache(
            seqlen_k, paged_kv_block_size, batch_size, nheads_k, d, device, dtype, fp8_dtype
        )
    cache_seqlens = torch.randint(
        0 if new_kv else 1,
        # If we don't use seqlen_q in the case of causal and rotary, cos/sin won't be long enough
        (
            (seqlen_k - (seqlen_q if (causal or local) and rotary_dim > 1 else seqlen_new) + 1)
            if new_kv
            else (seqlen_k + 1)
        ),
        (batch_size,),
        dtype=torch.int32,
        device=device,
    )
    if has_leftpad:
        cache_leftpad = torch.cat([torch.randint(0, cache_seqlens[i].item(), (1,), dtype=torch.int32, device=device)
                                   if cache_seqlens[i].item() > 0 else torch.zeros(1, dtype=torch.int32, device=device)
                                   for i in range(batch_size)])
    else:
        cache_leftpad = None
    arange = rearrange(torch.arange(seqlen_k, device=device), "s -> 1 s")
    cache_seqlens_expanded = rearrange(cache_seqlens, "b -> b 1")
    # key_padding_mask = arange < cache_seqlens_expanded + (seqlen_new if new_kv else 0)
    key_padding_mask, seq_lens = generate_random_padding_mask(seqlen_k, batch_size, device, mode="random" if varlen else "full")
    if has_leftpad:
        key_padding_mask = torch.logical_and(
            key_padding_mask, arange >= cache_leftpad.unsqueeze(-1).expand(-1, seqlen_k)
        )
    if has_batch_idx:
        cache_batch_idx = torch.randperm(batch_size_cache, dtype=torch.int32, device=device)[
            :batch_size
        ]
    else:
        cache_batch_idx = None
    if alibi:
        alibi_slopes = torch.rand(batch_size, nheads, device=device, dtype=torch.float32) * 0.3
        attn_bias = attn_bias_from_alibi_slopes(
            alibi_slopes, seqlen_q, seqlen_k, None, key_padding_mask, causal=causal, key_leftpad=cache_leftpad
        )
    else:
        alibi_slopes, attn_bias = None, None
    # cache_seqlens = torch.tensor([64], dtype=torch.int32, device=device)
    if rotary_dim > 0:
        angle = (
            torch.rand(
                seqlen_k if paged_kv_block_size is None else num_blocks * paged_kv_block_size,
                rotary_dim // 2,
                device=device,
            )
            * 2
            * math.pi
        )
        cos = torch.cos(angle).to(dtype=dtype)
        sin = torch.sin(angle).to(dtype=dtype)
        if causal or local:
            q_ro = apply_rotary_emb(
                q, cos, sin, seqlen_offsets=cache_seqlens, interleaved=rotary_interleaved
            )
        else:
            q_ro = rearrange(
                apply_rotary_emb(
                    rearrange(q, "b s h d -> b 1 (s h) d"),
                    cos,
                    sin,
                    seqlen_offsets=cache_seqlens,
                    interleaved=rotary_interleaved,
                ),
                "b 1 (s h) d -> b s h d",
                s=seqlen_q,
            )
        # q_ro = q
        k_ro = apply_rotary_emb(
            k, cos, sin, seqlen_offsets=cache_seqlens, interleaved=rotary_interleaved
        )
    else:
        cos, sin = None, None
        q_ro, k_ro = q, k
    # k_cache[:, 64:] = -1
    k_cache_ref = (
        k_cache if not has_batch_idx else k_cache[cache_batch_idx.to(dtype=torch.long)]
    ).clone()
    v_cache_ref = (
        v_cache if not has_batch_idx else v_cache[cache_batch_idx.to(dtype=torch.long)]
    ).clone()
    if new_kv:
        update_mask = torch.logical_and(
            cache_seqlens_expanded <= arange, arange < cache_seqlens_expanded + seqlen_new
        )
        k_cache_ref[update_mask] = rearrange(k_ro, "b s ... -> (b s) ...")
        v_cache_ref[update_mask] = rearrange(v, "b s ... -> (b s) ...")
    k_cache_rep = repeat(k_cache_ref, "b s h d -> b s (h g) d", g=nheads // nheads_k)
    v_cache_rep = repeat(v_cache_ref, "b s h d -> b s (h g) d", g=nheads // nheads_k)

    wg_size = int(d / 16)
    partition_size = int(wg_size * paged_kv_block_size)
    # partition_size = 512
    if has_learnable_sinks:
        learnable_sinks = torch.ones(nheads, dtype=dtype, device=device) * 10
    else:
        learnable_sinks = None
    out_ref, out_attn, out_partitions = attention_ref_fp8kv(
        q_ro,
        k_cache_rep,
        v_cache_rep,
        learnable_sinks,
        None,
        key_padding_mask if varlen else None,
        attn_bias,
        0.0,
        None,
        causal=causal,
        window_size=window_size,
        key_leftpad=cache_leftpad,
    )
    out_pt, out_attn_2, _ = attention_ref_fp8kv(
        q_ro,
        k_cache_rep,
        v_cache_rep,
        learnable_sinks,
        None,
        key_padding_mask if varlen else None,
        attn_bias,
        0.0,
        None,
        causal=causal,
        window_size=window_size,
        upcast=False,
        reorder_ops=True,
        key_leftpad=cache_leftpad,
    )

    num_partitions = int((seqlen_k + partition_size - 1) / partition_size)
    max_logits = torch.empty((batch_size, nheads, num_partitions), dtype=torch.float32, device=device)
    exp_sums = torch.empty_like(max_logits)
    out = torch.zeros((batch_size, nheads, d), dtype=dtype, device=device)
    tem_output = torch.zeros((batch_size, nheads, num_partitions, d), dtype=dtype, device=device)
    debug_output = torch.zeros((batch_size, nheads, num_partitions * d), dtype=torch.float32, device=device)
    query = q.view(batch_size * seqlen_q, nheads, d).contiguous().to(device=device, dtype=dtype)
    alibi_slopes = torch.ones((nheads), dtype=torch.float32, device=device)
    context_lens = torch.tensor([seqlen_k] * batch_size, dtype=torch.int32, device=device)
    context_lens = seq_lens.squeeze(1).to(torch.int32).to(device) if varlen else context_lens
    # print(f"seq_lens: {seq_lens}")

    num_queries_per_token = int(nheads / nheads_k)
    sm_scale = 1. / math.sqrt(d)
    softcat = -1.0

    eclips_times = paged_attention_loop.run(
        max_logits,
        exp_sums,
        tem_output,
        out,
        debug_output,
        query,
        k_cache_paged,
        v_cache_paged,
        block_table,
        context_lens,
        num_queries_per_token,
        sm_scale,
        paged_kv_block_size,
        seqlen_k,
        alibi_slopes,
        learnable_sinks,
        softcat
    )

    if do_performance:
        num_iters = 512
        num_warm = 5
        num_kv = 50
        k_caches = list()
        v_caches = list()
        total_eclips_times = list()
        for i in range(num_kv):
            k_caches.append(k_cache_paged.clone())
            v_caches.append(v_cache_paged.clone())
        for i in range(num_iters + num_warm):
            cur_idx = int(i % num_kv)
            eclips_times = paged_attention_loop.run(
                max_logits,
                exp_sums,
                tem_output,
                out,
                debug_output,
                query,
                k_caches[cur_idx],
                v_caches[cur_idx],
                block_table,
                context_lens,
                num_queries_per_token,
                sm_scale,
                paged_kv_block_size,
                seqlen_k,
                alibi_slopes,
                learnable_sinks,
                softcat
            )
            attn_time = eclips_times[0]
            reduce_time = eclips_times[1] if len(eclips_times) > 1 else 0
            # reduce_time = 0
            total_time = attn_time + reduce_time
            total_kv_size = k_cache_paged.numel() * k_cache_paged.element_size() * 2
            if i >= num_warm:
                total_eclips_times.append(total_time)

        total_kv_size = k_cache_paged.numel() * k_cache_paged.element_size() * 2
        avg_time = sum(total_eclips_times) / len(total_eclips_times)
        print(f"...loop:  batch_size {batch_size}, seqlen_k {seqlen_k}, d {d}, nheads {nheads}, nheads_k {nheads_k} Avg time: {avg_time:.2f} us, Total kv size: {total_kv_size / (1024*1024):.2f} MB, Avg bandwidth: {total_kv_size / avg_time / (1000):.2f} GB/s")

    out_ref = out_ref.squeeze(1)
    out_pt = out_pt.squeeze(1)
    out_attn = out_attn.squeeze(2)
    out_attn_2 = out_attn_2.squeeze(2)
    out_partitions = out_partitions.squeeze(1)
    # print(f"out_ref: {out_ref[0, 0, :10]}")
    # print(f"out: {out[0, 0, :10]}")
    print(f"eclips times: {eclips_times}")
    # assert_close_verbose(out_attn_2, debug_output, rtol=1e-3, atol=1e-3)
    # assert_close_verbose(out, out_ref, rtol=1e-3, atol=1e-3)
    # print(f"debug_output: {debug_output[0, 0, :]}")
    # print(f"out_attn_2: {out_attn_2[0, 0, :]}")
    # print(f"scores max diff: {(out_attn_2 - debug_output[:, :, :seqlen_k]).abs().max().item()}")
    # exit(0)
    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")

    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation.
    if new_kv:
        if paged_kv_block_size is None:
            k_cache_select = (
                k_cache if not has_batch_idx else k_cache[cache_batch_idx.to(dtype=torch.long)]
            )
            v_cache_select = (
                v_cache if not has_batch_idx else v_cache[cache_batch_idx.to(dtype=torch.long)]
            )
        else:
            k_cache_select = rearrange(
                k_cache_paged[block_table.to(dtype=torch.long).flatten()],
                "(b nblocks) block_size ... -> b (nblocks block_size) ...",
                b=batch_size,
            )[:, :seqlen_k]
            v_cache_select = rearrange(
                v_cache_paged[block_table.to(dtype=torch.long).flatten()],
                "(b nblocks) block_size ... -> b (nblocks block_size) ...",
                b=batch_size,
            )[:, :seqlen_k]
        assert torch.allclose(k_cache_select, k_cache_ref, rtol=1e-3, atol=1e-3)
        assert torch.equal(v_cache_select, v_cache_ref)
    mult = 3 if not alibi else 5
    assert (out - out_ref).abs().max().item() <= mult * (out_pt - out_ref).abs().max().item() + 1e-5


if __name__ == "__main__":
    test_flash_attn_kvcache(
        batch_size = 64,
        nheads = 64,
        nheads_k = 8,
        seqlen_q = 1,
        seqlen_k = 1500,
        d = 64,
        has_batch_idx = False,
        has_leftpad = False,
        paged_kv_block_size = 64,
        rotary_fraction = -1,
        rotary_interleaved = False,
        seqlen_new_eq_seqlen_q = True,
        causal = False,
        local = False,
        alibi = False,
        new_kv = False,
        has_learnable_sinks = True,
        mha_type = "gqa",
        num_splits = 1,
        dtype = torch.float16,
        varlen=False,
        do_performance=True,
    )
