import tensorflow as tf 
import numpy as np 
from tensorflow.models.rnn import rnn
from tensorflow.models.rnn import rnn_cell
# from tensorflow.models.rnn.rnn_cell import BasicLSTMCell

pHolder = tf.placeholder(tf.float32, [None, 9])
Weights = tf.Variable(tf.zeros([6,]))