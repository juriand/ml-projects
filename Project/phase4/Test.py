import tensorflow as tf
from tensorflow import keras
from keras.utils.vis_utils import plot_model
from keras.models import load_model
import pydot
import os
import pickle
os.environ["PATH"] += os.pathsep + 'C:/toolkit\graphviz-2.38/release/bin'

test_model = load_model('pre.model')
plot_model(test_model, to_file='pre.png', show_shapes=True, show_layer_names=True)