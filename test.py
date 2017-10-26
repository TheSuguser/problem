import sys, os
_snlp_book_dir = "../../../../"
sys.path.append(_snlp_book_dir)
import statnlpbook.lm as lm
import statnlpbook.ohhla as ohhla
import math
import numpy as np
import matplotlib.pyplot as plt
import collections
from sklearn import linear_model


_snlp_train_dir = _snlp_book_dir + "/data/ohhla/train"
_snlp_dev_dir = _snlp_book_dir + "/data/ohhla/dev"
_snlp_train_song_words = ohhla.words(ohhla.load_all_songs(_snlp_train_dir))
_snlp_dev_song_words = ohhla.words(ohhla.load_all_songs(_snlp_dev_dir))
assert(len(_snlp_train_song_words)==1041496)



def GoodTuring(train, order):

    counts = collections.defaultdict(float)
    r = collections.defaultdict(float)
    r_star = collections.defaultdict(float)

    for i in range(order, len(train)):
        history = tuple(train[i-order+1: i])
        word = train[i]
        counts[(word,) + history] += 1.0
    print(counts)
    for key in counts:
        r[counts[key]] += 1.0

    return r

oov_train = lm.inject_OOVs(_snlp_train_song_words)
dict_oov = GoodTuring(oov_train, 2)

"""
plt.figure()
plt.scatter(np.log(list(dict_oov.keys())), np.log(list(dict_oov.values())))
plt.show()
"""



def linear_good_turing(dict_oov):

    X = np.array(list(dict_oov.keys()))
    y = np.array(list(dict_oov.values()))

    X = X.reshape([len(X),1])
    y = y.reshape([len(y),1])

    regr = linear_model.LinearRegression()

    regr.fit(np.log(X), np.log(y))

    b = regr.coef_

    print(regr.coef_)

    r_star = collections.defaultdict(float)

    r_max = max(X)

    for i in range(1,int(r_max)+1):

        r_star[i] = i * (1+ 1/i)**(b+1)

    return r_star



dict_r = GoodTuring(oov_train, 2)

r = np.array(list(dict_r.keys()))
Nr = np.array(list(dict_r.values()))

N = np.sum(r*Nr)

print(N)

#c = linear_good_turing(dict_r)









