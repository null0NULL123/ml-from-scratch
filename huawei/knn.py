import numpy as np

if __name__ == "__main__":
    k, m, n, s = map(int, input().split())
    test = np.array(list(map(float, input().split())))
    sample = [list(map(float, input().split())) for _ in range(m)]
    labels = np.array([int(sample[i].pop()) for i in range(m)])
    sample = np.array(sample)
    d = np.pow(test - sample, 2).sum(axis=1)
    sort = labels[np.argsort(d)][:k]
    cnt = [len(np.where(sort == i)[0]) for i in range(k)]
    arg = np.argmax(cnt)
    print(arg, cnt[arg])
