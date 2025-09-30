import numpy as np


def softmax(x):
    exp = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp / np.sum(exp, axis=-1, keepdims=True)


if __name__ == "__main__":
    b, d, r = map(int, input().split())
    x = np.array(list(map(float, input().split())), dtype=np.float64).reshape(b, d)
    wq, wk, wv = [
        np.array(list(map(float, input().split())), dtype=np.float64).reshape(d, d)
        for _ in range(3)
    ]
    a = np.array(list(map(float, input().split())), dtype=np.float64).reshape(r, d)
    b = np.array(list(map(float, input().split())), dtype=np.float64).reshape(d, r)
    wq += b @ a
    q = x @ wq.T
    k = x @ wk.T
    v = x @ wv.T
    lora_attention: np.ndarray = softmax(q @ k.T / np.sqrt(d)) @ v
    tolist = np.round(lora_attention.flatten(), 4).tolist()
    print(" ".join(list(map(str, tolist))))
