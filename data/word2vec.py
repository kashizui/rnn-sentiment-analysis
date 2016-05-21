# http://rare-technologies.com/word2vec-tutorial/

# import modules & set up logging
import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# train word2vec on the two sentences


class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
 
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()
 
sentences = MySentences('stanfordSentimentTreebank/datasetSentences.txt') # a memory-friendly iterator
model = gensim.models.Word2Vec(sentences)

model = gensim.models.Word2Vec() # an empty model, no training
# model.build_vocab(some_sentences)  # can be a non-repeatable, 1-pass generator
# model.train(other_sentences)  # can be a non-repeatable, 1-pass generator
