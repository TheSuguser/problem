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

class GoodTuringLM(lm.LanguageModel):
	"""
	This is a LM based on Good Turing estimation
	"""

	def __init__(self, train, order):

		super().__init__(set(train), order)
		self.train = train
		self.order = order

		self.r_star = collections.defaultdict(float)

		self.counts = collections.defaultdict(float)
		dict_r = collections.defaultdict(float)

		for i in range(self.order, len(self.train)):
			history = tuple(self.train[i-self.order+1 : i])
			word = self.train[i]
			self.counts[(word,) + history] += 1.0

		#print(self.counts)
		for key in self.counts:
			dict_r[self.counts[key]] +=1

		X = np.array(list(dict_r.keys()))
		y = np.array(list(dict_r.values()))

		X = X.reshape([len(X),1])
		y = y.reshape([len(X),1]) 

		regr = linear_model.LinearRegression()

		regr.fit(np.log(X), np.log(y))

		b = regr.coef_

		r_max = max(X)

		for j in range(1 , int(r_max)+1):

			self.r_star[j] = j * (1 + 1 / j)**(b+1)

		r = np.array(list(dict_r.keys()))
		Nr = np.array(list(dict_r.values()))
		self.N = 0
		for k in r:
			print(k)
			self.N += (dict_r[k]) * self.r_star[int(k)]

	
	def probability(self, word, *history):


		sub_history = tuple(history[-(self.order-1):]) if self.order >1 else()
		#print((word,)+ sub_history)

		if self.counts[(word,)+sub_history] == 0.0:
			#print("yes")
			#print(self.r_star[1], self.N)
			return self.r_star[1]/ self.N
		else:
			#print("No")
			return self.r_star[int(self.counts[(word,)+ history])]/ self.N




"""
TEST PART
"""
#! SETUP 2
_snlp_train_dir = _snlp_book_dir + "/data/ohhla/train"
_snlp_dev_dir = _snlp_book_dir + "/data/ohhla/dev"
_snlp_train_song_words = ohhla.words(ohhla.load_all_songs(_snlp_train_dir))
_snlp_dev_song_words = ohhla.words(ohhla.load_all_songs(_snlp_dev_dir))
assert(len(_snlp_train_song_words)==1041496)

oov_train = lm.inject_OOVs(_snlp_train_song_words)
oov_vocab = set(oov_train)

def create_lm(vocab):
    """
    Return an instance of `lm.LanguageModel` defined over the given vocabulary.
    Args:
        vocab: the vocabulary the LM should be defined over. It is the union of the training and test words.
    Returns:
        a language model, instance of `lm.LanguageModel`.
    """
    
    
    return lm.OOVAwareLM(GoodTuringLM(oov_train, 2), vocab - oov_vocab)


#! SETUP 3
_snlp_test_dir = _snlp_book_dir + "/data/ohhla/dev"


#! SETUP 4
_snlp_test_song_words = ohhla.words(ohhla.load_all_songs(_snlp_test_dir))
_snlp_test_vocab = set(_snlp_test_song_words)
_snlp_dev_vocab = set(_snlp_dev_song_words)
_snlp_train_vocab = set(_snlp_train_song_words)
_snlp_vocab = _snlp_test_vocab | _snlp_train_vocab | _snlp_dev_vocab
_snlp_lm = create_lm(_snlp_vocab)


#! ASSESSMENT 1
_snlp_test_token_indices = [100, 1000, 10000]
_eps = 0.000001
for i in _snlp_test_token_indices:
    result = sum([_snlp_lm.probability(word, *_snlp_test_song_words[i-_snlp_lm.order+1:i]) for word in _snlp_vocab])
    print("Sum: {sum}, ~1: {approx_1}, <=1: {leq_1}".format(sum=result, 
                                                            approx_1=abs(result - 1.0) < _eps, 
                                                            leq_1=result - _eps <= 1.0))

print(lm.perplexity(_snlp_lm, _snlp_test_song_words))



















