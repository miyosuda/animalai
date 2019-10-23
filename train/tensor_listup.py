# -*- coding: utf-8 -*-
# @see
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/inspect_checkpoint.py
#
# https://stackoverflow.com/questions/39137597/how-to-restore-variables-using-checkpointreader-in-tensorflow
import tensorflow as tf
import numpy as np

from tensorflow.python import pywrap_tensorflow

load_tensor_names = [
    "main_graph_0_encoder1/conv_1/bias",
    "main_graph_0_encoder1/conv_1/kernel",
    "main_graph_0_encoder1/conv_2/bias",
    "main_graph_0_encoder1/conv_2/kernel",
    "main_graph_0_encoder1/conv_3/bias",
    "main_graph_0_encoder1/conv_3/kernel",
    "main_graph_0_encoder1/flat_encoding/main_graph_0_encoder1/hidden_0/bias",
    "main_graph_0_encoder1/flat_encoding/main_graph_0_encoder1/hidden_0/kernel",
]


def load_tensors_in_checkpoint_file(file_name):
    reader = pywrap_tensorflow.NewCheckpointReader(file_name)
    var_to_shape_map = reader.get_variable_to_shape_map()
    var_to_dtype_map = reader.get_variable_to_dtype_map()
    
    for key, value in sorted(var_to_shape_map.items()):
        print("tensor: %s (%s) %s" % (key, var_to_dtype_map[key].name, value))
        #tensor = reader.get_tensor(key)

checkpoint = tf.train.latest_checkpoint("models/run_109/Learner")
# "models/run_109/Learner/model-1642786.cptk" といったパス

load_tensors_in_checkpoint_file(checkpoint)

