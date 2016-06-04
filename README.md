# rnn-sentiment-analysis
CS224u Final Project

Use Recurrent Neural Networks to identify the sentiment of an input text.

Created by Nico Cserepy, Stephen Koo, and Ben Weems

Start with the following command to list all the possible options.

    python train.py -h

Example training and evaluation run:

    python train.py --train-embedding --parameters=params.tflearn --epochs=5

To continue training from where you left off (if you previously saved parameters at `params.tflearn`):

    python train.py --train-embedding --parameters=params.tflearn --epochs=5 --continue-training

To try a different set of GLoVE vectors:

    python train.py --glove=YOURVECTORS.txt --data-cache=newdata.pkl --force-preprocess

The `--force-preprocess` flag here may be important here if you previously used a different set of GLoVE vectors with the same data cache location, since the preprocessing steps depend on the GLoVE vectors you are using. If `--force-preprocess` is not specified, the program will attempt to load previously cached preprocessed data from the specified location (or the default `sst_data.pkl`).

To create a new model, create a new module at `data/models/<modelname>.py`.
The module should define function called `build()`, which should
return a network built by tflearn (see `data/models/lstm.py` for an example).

