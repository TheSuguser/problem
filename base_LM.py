#! SETUP 1
import sys, os
_snlp_book_dir = "../../../../"
sys.path.append(_snlp_book_dir) 
import statnlpbook.lm as lm
import statnlpbook.ohhla as ohhla
import math
import numpy as np
import matplotlib.pyplot as plt

#! SETUP 2
_snlp_train_dir = _snlp_book_dir + "/data/ohhla/train"
_snlp_dev_dir = _snlp_book_dir + "/data/ohhla/dev"
_snlp_train_song_words = ohhla.words(ohhla.load_all_songs(_snlp_train_dir))
_snlp_dev_song_words = ohhla.words(ohhla.load_all_songs(_snlp_dev_dir))
assert(len(_snlp_train_song_words)==1041496)

# Define Interpolated language model
class InterpolatedLM(lm.LanguageModel):
    def __init__(self, main, backoff, alpha):
        super().__init__(main.vocab, main.order)
        self.main = main
        self.backoff = backoff
        self.alpha = alpha
    def probability(self, word, *history):
        return self.alpha * self.main.probability(word,*history) + \
               (1.0 - self.alpha) * self.backoff.probability(word,*history)



# Deal with raw data
oov_train = lm.inject_OOVs(_snlp_train_song_words)
bigram = lm.NGramLM(oov_train, 2)
unigram = lm.NGramLM(oov_train,1)
trigram = lm.NGramLM(oov_train,3)
inter_lm = InterpolatedLM(bigram, unigram, 0.7171)
oov_vocab = set(oov_train)


#Create Language model
def create_lm(vocab):
    """
    Return an instance of `lm.LanguageModel` defined over the given vocabulary.
    Args:
        vocab: the vocabulary the LM should be defined over. It is the union of the training and test words.
    Returns:
        a language model, instance of `lm.LanguageModel`.
    """
    
    
    return lm.OOVAwareLM(inter_lm, vocab - oov_vocab)


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

#! ASSESSMENT 2
print("Perlexity:", lm.perplexity(_snlp_lm, _snlp_test_song_words))



# Test parameter
step = 100
alpha_range = np.linspace(0, 1, step)
perp = np.zeros(step)

for i in range(step):

	print(i)

	inter_lm = InterpolatedLM(bigram,unigram, alpha_range[i])
	_snlp_lm = create_lm(_snlp_test_vocab)
	perp[i] = lm.perplexity(_snlp_lm, _snlp_test_song_words)

min_indice = np.argmin(perp)
print(alpha_range[min_indice])

plt.figure()
plt.plot(alpha_range, perp)
plt.show()












