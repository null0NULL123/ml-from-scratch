import numpy as np


def func():
    n, k, t = map(int, input().split())
    anchor = np.array(
        [list(map(int, input().split())) for _ in range(n)], dtype=np.int64
    )
    center = anchor[:k]
    labels = [0] * n
    cnt = np.array([0] * k, dtype=np.int64)

    for _ in range(t):
        cnt.fill(0)
        s = np.zeros_like(center, dtype=float)
        union = np.zeros((n, k))
        intersection = np.zeros((n, k))
        iou = np.zeros((n, k), dtype=float)

        for i in range(n):
            for j in range(k):
                intersection[i, j] = min(anchor[i, 0], center[j, 0]) * min(
                    anchor[i, 1], center[j, 1]
                )
                union[i, j] = (
                    (anchor[i, 0] * anchor[i, 1])
                    + (center[j, 0] * center[j, 1])
                    - intersection[i, j]
                )

            iou[i] = intersection[i] / (union[i] + 1e-16)
            label = int(np.argmax(iou[i]))
            labels[i] = label
            s[label] += anchor[i]
            cnt[label] += 1

        true = cnt > 0
        s[true] /= cnt[true].reshape(-1, 1)
        diff = np.abs(s - center)

        if np.sum(diff) < 1e-4:
            break
        center = np.array(s.copy(), dtype=np.int64)
    multi = center[:, 0] * center[:, 1]
    print(center[np.argsort(multi)][::-1].tolist())


if __name__ == "__main__":
    func()
