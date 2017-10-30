import sys, os
_snlp_book_dir = "../../../../"
sys.path.append(_snlp_book_dir) 
import statnlpbook.lm as lm
import statnlpbook.ohhla as ohhla
import math
import numpy as np
import matplotlib.pyplot as plt
import collections

class new_NGramLM(lm.CountLM):
    def __init__(self, train, order):
        """
        Create an NGram language model.
        Args:
            train: list of training tokens.
            order: order of the LM.
        """
        super().__init__(set(train), order)
        self._counts = collections.defaultdict(float)
        self._norm = collections.defaultdict(float)
        self._counts_history = collections.defaultdict(float)
        self._counts_word = collections.defaultdict(float)
        for i in range(self.order, len(train)):
            history = tuple(train[i - self.order + 1 : i])
            word = train[i]
            self._counts[(word,) + history] += 1.0
            self._norm[history] += 1.0
        for j in list(self._counts.keys()):
        	self._counts_word[list(j)[0]] += 1.0
        for k in list(self._counts.keys()):
        	self._counts_history[list(k)[1]] += 1.0
    def counts(self, word_and_history):
    	#print(self._counts)
    	#print(word_and_history, self._counts[word_and_history])
    	return self._counts[word_and_history]
    def norm(self, history):
    	#print(history, self._norm[history])
    	return self._norm[history]
    def counts_history(self, history):
    	"""
    	count = 0
    	for i in list(self._counts.keys()):
    		if history == list(i)[1]:
    			count += 1
    	print(history,count)
    	return count
    	"""
    	return self._counts_history[history]
    def counts_word(self, word):
    	"""
    	count = 0
    	for i in list(self._counts.keys()):
    		if word == list(i)[0]:
    			count += 1
    	print("counts_word:",count)
    	return count
    	"""
    	return self._counts_word[word]
    def counts_all(self):
    	return len(list(self._counts.keys()))

class KneseNeyLM(lm.LanguageModel):
	def __init__(self, main, backoff, discount):
		super().__init__(main.vocab, main.order)
		self.main = main
		self.backoff = backoff
		self.discount = discount
	def probability(self, word, *history):
        sub_history = tuple(history[-1:])
        word_and_history = (word,) + sub_history
        #print(word_and_history)
        history = "".join(sub_history)
        mark += 1
        print(mark)
		if self.main.counts_history(history) == 0.0:
			
			return self.backoff.probability(word)
	
		else:
			
			p1 = (self.main.counts(word_and_history) - self.discount)/ self.backoff.counts((history,))
			lmb = self.discount / self.backoff.counts((history,)) * self.main.counts_history(history)
			pc = self.main.counts_word(word) / self.main.counts_all()

			#print(p1+lmb*pc)

			return p1 + lmb * pc

_snlp_train_dir = _snlp_book_dir + "/data/ohhla/train"
_snlp_dev_dir = _snlp_book_dir + "/data/ohhla/dev"
_snlp_train_song_words = ohhla.words(ohhla.load_all_songs(_snlp_train_dir))
_snlp_dev_song_words = ohhla.words(ohhla.load_all_songs(_snlp_dev_dir))
assert(len(_snlp_train_song_words)==1041496)
_snlp_test_dir = _snlp_book_dir + "/data/ohhla/dev"

oov_train = lm.inject_OOVs(_snlp_train_song_words)
my_lm1 = new_NGramLM(oov_train, 2)
my_lm2 = lm.NGramLM(oov_train,1)
my_lm = KneseNeyLM(my_lm1, my_lm2, 0.75)
oov_vocab = set(oov_train)

## You should improve this cell
def create_lm(vocab):
    """
    Return an instance of `lm.LanguageModel` defined over the given vocabulary.
    Args:
        vocab: the vocabulary the LM should be defined over. It is the union of the training and test words.
    Returns:
        a language model, instance of `lm.LanguageModel`.
    """
    
    
    return lm.OOVAwareLM(my_lm, vocab - oov_vocab)

mark =0
_snlp_test_song_words = ohhla.words(ohhla.load_all_songs(_snlp_test_dir))
_snlp_test_vocab = set(_snlp_test_song_words)
_snlp_dev_vocab = set(_snlp_dev_song_words)
_snlp_train_vocab = set(_snlp_train_song_words)
_snlp_vocab = _snlp_test_vocab | _snlp_train_vocab | _snlp_dev_vocab
_snlp_lm = create_lm(_snlp_vocab)



_snlp_test_token_indices = [100, 1000, 10000]
_eps = 0.000001
for i in _snlp_test_token_indices:
    result = sum([_snlp_lm.probability(word, *_snlp_test_song_words[i-_snlp_lm.order+1:i]) for word in _snlp_vocab])
    print("Sum: {sum}, ~1: {approx_1}, <=1: {leq_1}".format(sum=result, 
                                                            approx_1=abs(result - 1.0) < _eps, 
                                                            leq_1=result - _eps <= 1.0))



#print(lm.perplexity(_snlp_lm, _snlp_test_song_words))











