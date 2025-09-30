import numpy as np


def softmax(x):
    exp = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp / np.sum(exp, axis=-1, keepdims=True)


if __name__ == "__main__":
    num_heads, x, wq, wk, wv, wo = input().split(";")
    num_heads = int(num_heads)
    x, wq, wk, wv, wo = map(lambda m: np.array(eval(m)), [x, wq, wk, wv, wo])
    q, k, v = map(lambda m: x @ m, [wq, wk, wv])
    batch_size, seq_len, d_model = x.shape
    d_k = d_model // num_heads
    q_h, k_h, v_h = map(
        lambda m: np.transpose(
            m.reshape(batch_size, seq_len, num_heads, d_k), (0, 2, 1, 3)
        ),
        [q, k, v],
    )
    attention_scores = q_h @ np.transpose(k_h, (0, 1, 3, 2)) / np.sqrt(d_k)
    mask = np.tril(np.ones((seq_len, seq_len))).reshape(1, 1, seq_len, seq_len)
    masked_scores = np.where(mask == 1, attention_scores, -np.inf)
    softmax_scores = softmax(masked_scores)
    attention = np.transpose(softmax_scores @ v_h, (0, 2, 1, 3)).reshape(x.shape)
    print(np.round((attention @ wo), 2).tolist())
