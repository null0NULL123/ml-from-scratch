import math
from collections import Counter


def inverse_document_frequency(corpus: list[str]):
    idf = Counter()
    for doc in corpus:
        d = doc.split()
        idf.update(set(d))
    total_docs = len(corpus)
    for word, count in idf.items():
        idf[word] = math.log(total_docs / (count + 1))
    return idf


def term_frequency(corpus: list[str]):
    for doc in corpus:
        d = doc.split()
        cnt = Counter(d)
        doc_len = len(d)
        tf_doc = {word: count / doc_len for word, count in cnt.items()}
        yield tf_doc


def tf_idf(corpus: list[str]):
    idf = inverse_document_frequency(corpus)
    for tf_doc in term_frequency(corpus):
        tfidf = {word: tf * idf[word] for word, tf in tf_doc.items()}
        yield tfidf


if __name__ == "__main__":
    print(
        list(
            tf_idf(
                [
                    "What is the weather like today",
                    "what is for dinner tonight",
                    "this is question worth pondering",
                    "it is a beautiful day today",
                ]
            )
        )
    )
))
