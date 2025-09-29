import numpy as np

n, k = map(int, input().split())

data = np.array([list(map(int, input().split())) for _ in range(n)], dtype=float)


def kmeans(data, n, k, e=1e-6, it=100):
    centers = data[:k]
    labels = np.zeros(n, dtype=np.int32)

    tmp = np.zeros((k, 2), dtype=float)
    cnt = np.array([0] * k, dtype=np.int32)
    for _ in range(it):
        tmp.fill(0)
        cnt.fill(0)

        for i in range(n):

            l = np.argmin(np.pow(data[i] - centers, 2).sum(axis=1))
            labels[i] = int(l)
            cnt[l] += 1
            tmp[l] += data[i]

        true = cnt > 0
        tmp[true] /= cnt[true].reshape(-1, 1)

        sum = np.sum(np.abs(tmp - centers))
        if sum < e:
            break
        centers = tmp.copy()
    return centers, labels, cnt


def silhouette(centers: np.ndarray, labels, cnt):
    d = np.sqrt(
        np.sum(np.pow(data[:, None, :] - data[None, :, :], 2), axis=2), dtype=float
    )
    clusters = [np.where(labels == c)[0] for c in range(k)]

    s = np.zeros(n, dtype=float)
    for i in range(n):
        if c := cnt[l := labels[i]] - 1:
            a = np.sum(d[i, clusters[l]]) / c
            b = np.inf
            for j in range(k):
                if j == l or not (c := cnt[j]):
                    continue
                b = min(np.mean(d[i, clusters[j]]), b)
            s[i] = (b - a) / max(b, a)
        else:
            s[i] = 1
    sc = np.zeros(k, dtype=float)
    for i in range(k):
        if not cnt[i]:
            continue
        sc[i] = np.mean(s[clusters[i]])
    ret: np.ndarray = centers[np.argmin(sc)]
    return ret.tolist()


centers, labels, cnt = kmeans(data, n, k)
x, y = silhouette(centers, labels, cnt)
x = list(str(x))
y = list(str(y))

for i in [x, y]:
    if int(i[3]) & 1:
        i[3] += 1
print(f'{"".join(x[:4])},{"".join(y[:4])}')
