import os
import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt
from scipy.special import softmax
from tensorflow.keras import Model
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.keras.backend import get_graph
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import Add, Conv1D, Conv2D, Dense, Flatten, Input, LSTM

import memory as mm



# set default precision
tf.keras.backend.set_floatx('float64')

# set thread level
num_threads = 6
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["TF_NUM_INTRAOP_THREADS"] = "6"
os.environ["TF_NUM_INTEROP_THREADS"] = "6"

tf.config.threading.set_inter_op_parallelism_threads(num_threads)
tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.set_soft_device_placement(True)


#### TEST PARAMETERS ####
# choose NTM task to replicate 
task = "copy"
# input size
dim = 8
# temporal length
tlen = 10
# write head specification
write_heads = [["ff", 100]]
# read head specification
read_heads = [["ff"], 10]
# memory shape
memory_shape = [128,20]
# learning rate
learning_rate = 1e-4
# epochs
epoch_num = 10
# whether to train
train = True
###### Generate some test data ########







###### Network setup ######
# leverage Keras sequential model layers
model_input = Input(shape=())



##### Performance analysis




