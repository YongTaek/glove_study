from sklearn.feature_extraction.text import CountVectorizer
import _pickle as pickle
import msgpack

import numpy as np
from konlpy.corpus import kolaw

corpus = kolaw.open('constitution.txt').read()

print(corpus.split("\n"))

vectorizer = CountVectorizer(min_df=10, ngram_range=(1,1))
X = vectorizer.fit_transform(corpus.split("\n"))
Xc = X.T * X
Xc.setdiag(0)

result = Xc.toarray()

dic = {}
for idx1, word1 in enumerate(result):
    tmpdic = {}
    for idx2, word2 in enumerate(word1):
        if word2 > 0:
            tmpdic[idx2] = word2
        dic[idx1] = tmpdic

# print(dic)

import operator

vocab = sorted(vectorizer.vocabulary_.items(), key=operator.itemgetter(1))

vocab = [word[0] for word in vocab]
# print(vocab)

import glove

model = glove.Glove(dic, d = 100, alpha=0.75, x_max=100.0)
for epoch in range(25):
    err = model.train(batch_size=200, workers=4)
    print("epoch %d, error %.3f" % (epoch, err), flush=True)

wordvectors = model.W
import pickle
with open('glove','wb') as f:
    pickle.dump([vocab, wordvectors], f)

from scipy.spatial.distance import cosine
def most_similar(word, vocab, vecs, topn=10):
    query = vecs[vocab.index(word)]
    result = []
    for idx, vec in enumerate(vecs):
        if idx is not vocab.index(word):
            result.append((vocab[idx], 1-cosine(query, vec)))
    result = sorted(result, key=lambda x: x[1], reverse=True)
    return result[:topn]


print("most_similar")
print(most_similar(word="영화", vocab=vocab, vecs=wordvectors, topn=5))