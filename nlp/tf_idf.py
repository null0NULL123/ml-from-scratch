import math
from collections import defaultdict, Counter

def tf_idf(corpus:list[str]):
    idf = Counter()
    tf = Counter()
    for doc in corpus:
        d = doc.split()
        tf.update(d)
        idf.update(set(d))
    total_words = sum(tf.values())
    total_docs = len(corpus)
    for word, count in idf.items():
        idf[word] = math.log(total_docs / (count + 1))

    tf_idf = {word: tf[word] * idf[word] / total_words for word in tf}
    return tf_idf

if __name__ == "__main__":
    print(tf_idf(["this is a sample", "this is another example example example"]))
