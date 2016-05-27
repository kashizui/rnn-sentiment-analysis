from __future__ import division, print_function

import numpy as np
import pickle as pkl
import utils

NUM_EXAMPLES = float('inf')


def split_database(list_to_split, split_fname):
	"""
	Split a list of sentences into train, test, and dev groups.

	Based on the index specified in split_fname, the sentences will
	be grouped according to index into lists.

	Arguments:
	----------
		list_to_split: the list to be split into groups.
		split_fname: name of file with one line for each list element

	Returns:
	----------
		train: Training set as divided by split_fname
		test: Test set as divided by split_fname
		dev: Dev set as divided by split_fname
	"""
	train = []
	test = []
	dev = []
	with open(split_fname, 'rb') as split:
		next(split)
		for line in split:
			index, group = line.split(',')
			group = int(group.strip())
			if index in list_to_split:
				if group == 1:
					train.append(list_to_split[index])
				elif group == 2:
					test.append(list_to_split[index])
				elif group == 3:
					dev.append(list_to_split[index])
	return train, test, dev


def phrases2ints(word_dict, dataset_sentences_fname):
	"""
	Convert a dataset of sentences into numbers corresponding to the words in the sentence.

	Arguments:
	----------
		word_dict: Dict relating words to their indices
		dataset_sentences_fname: File name that holds the sentences in the dataset.

	Returns:
	----------
		int_phrases: Returns phrases with ints substituted for words corresponding to their index
						in dataset_sentences_fname

	"""
	with open(dataset_sentences_fname, 'rb') as dataset_sentences:
		int_phrases = []
		next(dataset_sentences)
		num_lines_read = 0
		for sentence in dataset_sentences:
			if num_lines_read >= NUM_EXAMPLES:
				break

			index, phrase = sentence.split('\t')
			int_phrase = tuple(
				word_dict.get(word.lower(), 0)  # zero doesn't really make sense here
				for word in phrase.split()
			)
			int_phrases.append((int_phrase, index))
			num_lines_read += 1

	return int_phrases


def get_phrases_dict(phrases_fname):
	"""
	Convert a file with phrases into a phrase dict and a word dict.

	Arguments:
	----------
		phrases_fname: Name of file that holds phrases and their corresponding index.

	Returns:
	----------
		phrase_dict: Dict of phrases and their corresponding index
		word_dict: subset of phrase_dict that only contains single words
	"""
	phrase_dict = {}
	with open(phrases_fname, 'rb') as phrases:
		for line in phrases.readlines():
			phrase, index = line.split('|')
			phrase_dict[phrase] = int(index.strip())
	word_dict = {key: val for key, val in phrase_dict.items() if len(key.split())==1}
	return phrase_dict, word_dict


def indices_to_sentiment(int_phrases, sentiment_labels_fname):
	with open(sentiment_labels_fname, 'rb') as sentiment_labels:
		next(sentiment_labels)
		label_dict = dict(tuple(item.split('|')) for item in sentiment_labels)

	return {
		idx: (phrase, utils.get_sentiment(float(label_dict[idx])))  # FIXME
		for phrase, idx in int_phrases
		if idx.isdigit()
	}


def glove_word_indices(glove_fname):
	_, words, _ = utils.build_glove(glove_fname)
	return {word: i for i, word in enumerate(words)}


def main():
	# Get the phrases and their indices
	print("Getting Phrase Dictionary...")
	phrase_dict, _ = get_phrases_dict('stanfordSentimentTreebank/dictionary.txt')

	print("Getting glove word indices...")
	word_dict = glove_word_indices('glove.6B/glove.6B.50d.txt')

	# Convert the phrases to ints with word indices so they can be processed by Neural Network
	print("Converting to Ints...")
	int_phrases = phrases2ints(word_dict, 'stanfordSentimentTreebank/datasetSentences.txt')

	# Convert indices to sentiment values
	print("Converting Indices to Sentiment Values...")
	phrase_sentiments = indices_to_sentiment(int_phrases, 'stanfordSentimentTreebank/sentiment_labels.txt')

	# Split into train, test, and dev groups
	print("Splitting into train, test, and dev groups...")
	train, test, dev = split_database(phrase_sentiments, 'stanfordSentimentTreebank/datasetSplit.txt')

	data = {
			'train': train,
			'test': test,
			'dev': dev
	}

	pkl.dump(data, open( "sst_data.pkl", "wb" ) )

if __name__=="__main__":
	main()