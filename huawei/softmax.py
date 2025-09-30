import numpy as np

def softmax(x):
    exp = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp / np.sum(exp, axis=-1, keepdims=True)

if __name__ == "__main__":
    n,m,h = map(int, input().split())
    x = np.ones((n, m))
    w = np.eye(m, h)
    q = k = v = x @ w

    print(int(round(np.sum(softmax(q @ k.T / np.sqrt(h)) @ v), 0)))