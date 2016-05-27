from __future__ import division, print_function

import numpy as np
import pickle as pkl
import utils

NUM_EXAMPLES = 2500


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
		for line in split.readlines():
			index, group = line.split(',')
			group = group.strip()
			if group.isdigit() and int(index) < len(list_to_split):
				if int(group)==1:
					train.append(list_to_split[int(index)])
				elif int(group)==2:
					test.append(list_to_split[int(index)])
				elif int(group)==3:
					dev.append(list_to_split[int(index)])
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
	dataset = []
	with open(dataset_sentences_fname, 'rb') as dataset_sentences:
		dataset = [sentence.split('\t') for sentence in dataset_sentences.readlines()[2:]]

	int_phrases = {}
	for index, phrase in dataset[:NUM_EXAMPLES]:
		words = phrase.split()
		int_phrase = [0]*len(words)
		for i, word in enumerate(words):
			if word in word_dict.keys():
				int_phrase[i] = word_dict[word]
		int_phrases[tuple(int_phrase)] = index
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
	phrase_sentiments = [0]*(len(int_phrases)+2)
	with open(sentiment_labels_fname, 'rb') as sentiment_labels:
		lines = [item.split('|') for item in sentiment_labels.readlines()]
		label_dict = {index: label for index, label in lines}
		for phrase, idx in int_phrases.items():
			sentiment = utils.get_sentiment(float(label_dict[idx]))
			if idx.isdigit() and int(idx) < len(phrase_sentiments):
				phrase_sentiments[int(idx)] = [phrase, sentiment]
	return phrase_sentiments


def main():
	# Get the phrases and their indices
	print("Getting Phrase Dictionary...")
	phrase_dict, word_dict = get_phrases_dict('stanfordSentimentTreebank/dictionary.txt')

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