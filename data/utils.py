

def get_sentiment(val):
	return int(val >= 0.5)

def format_data(dataset):
	print dataset[:10]
	x = [item[0] for item in dataset]
	y = [item[1] for item in dataset]
	return x, y