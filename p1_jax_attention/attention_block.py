import jax
import jax.numpy as jnp


def conv1d(x, w, b):
    return jnp.dot(x, w) + b


def causal_mask(seq_len):
    mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
    return mask[None, None, :, :]


def split_heads(x, n_head):
    b, s, f = x.shape
    head_dim = f // n_head
    x = x.reshape(b, s, n_head, head_dim)
    return x.transpose(0, 2, 1, 3)


def merge_heads(x):
    b, n, s, d = x.shape
    x = x.transpose(0, 2, 1, 3)
    return x.reshape(b, s, n * d)


def attn_block(x, params, mask, past=None):
    c = conv1d(x, params['c_attn_w'], params['c_attn_b'])
    q, k, v = jnp.split(c, 3, axis=-1)

    n_head = params['n_head']
    q = split_heads(q, n_head)
    k = split_heads(k, n_head)
    v = split_heads(v, n_head)

    if past is not None:
        past_k, past_v = past
        k = jnp.concatenate([past_k, k], axis=2)
        v = jnp.concatenate([past_v, v], axis=2)

    d_k = q.shape[-1]
    scores = jnp.einsum("bhsd,bhtd->bhst", q, k) / jnp.sqrt(d_k)

    scores = jnp.where(mask, scores, -1e10)
    weights = jax.nn.softmax(scores, axis=-1)
    a = jnp.einsum("bhst,bhtd->bhsd", weights, v)

    a = merge_heads(a)
    output = conv1d(a, params['c_proj_w'], params['c_proj_b'])

    present = (k, v)
    return output, present


if __name__ == "__main__":
    # Dummy dimensions and parameters.
    batch = 2
    seq = 16
    embd = 768
    n_head = 12
    head_dim = embd // n_head

    key = jax.random.PRNGKey(0)
    # Dummy input tensor.
    x = jax.random.normal(key, (batch, seq, embd))

    def init_params(key, shape, std=0.02):
        return jax.random.normal(key, shape) * std

    key, k1, k2 = jax.random.split(key, 3)
    params = {
        'c_attn_w': init_params(k1, (embd, 3 * embd)),
        'c_attn_b': jnp.zeros((3 * embd,)),
        'c_proj_w': init_params(k2, (embd, embd)),
        'c_proj_b': jnp.zeros((embd,)),
        'n_head': n_head,
    }

    mask = causal_mask(seq)

    output, present = attn_block(x, params, mask, past=None)
    print("Output shape:", output.shape)