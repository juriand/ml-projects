import tensorflow as tf
from tensorflow import keras
from keras.utils.vis_utils import plot_model
from keras.models import load_model
import pydot
import os
import pickle
os.environ["PATH"] += os.pathsep + 'C:/toolkit\graphviz-2.38/release/bin'


def main():
    # with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
    # test_model = load_model('williams.model')
    # plot_model(test_model, to_file='williams.png', show_shapes=True, show_layer_names=True)

    f1_train_b = 0.9801699716713881
    f1_test_b = 0.8358208955223879

    f1_train_w = 1.0
    f1_test_w = 0.5217391304347826

    pickle.dump((f1_train_b, f1_test_b),
                open('bush.pkl', 'wb'))
    pickle.dump((f1_train_w, f1_test_w),
                open('williams.pkl', 'wb'))


if __name__ == "__main__":
    main()
