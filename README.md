# rnn-sentiment-analysis
CS224u Final Project

Use Recurrent Neural Networks to identify the sentiment of an input text.

Created by Nico Cserepy, Stephen Koo, and Ben Weems

Everything is in the `data` directory at the moment.

Train then evaluate the LSTM model:

    python train.py lstm
    python evaluate.py lstm

To create a new model, create a new module at `data/models/<modelname>.py`.
The module should define function called `build()`, which should
return a network built by tflearn (see `data/models/lstm.py` for an example).

