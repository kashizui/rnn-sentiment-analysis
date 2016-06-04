"""
Models for sentiment analysis.
"""
import importlib

def get_model(name, *args, **kwargs):
    """
    Load the model defined in the module models/<name>.py

    The module should export a single 'build()' function that
    returns a tflearn network.
    """
    module = importlib.import_module('.' + name, 'models')
    return module.build(*args, **kwargs)
