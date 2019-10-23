# -*- coding: utf-8 -*-
# @see
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/inspect_checkpoint.py
#
# https://stackoverflow.com/questions/39137597/how-to-restore-variables-using-checkpointreader-in-tensorflow
import tensorflow as tf
import numpy as np

from tensorflow.python import pywrap_tensorflow


def load_tensors_in_checkpoint_file(file_name):
    reader = pywrap_tensorflow.NewCheckpointReader(file_name)
    var_to_shape_map = reader.get_variable_to_shape_map()
    var_to_dtype_map = reader.get_variable_to_dtype_map()
    
    for key, value in sorted(var_to_shape_map.items()):
        print("tensor: %s (%s) %s" % (key, var_to_dtype_map[key].name, value))
        #tensor = reader.get_tensor(key)

#model_path = "models/run_109/Learner"
model_path = "models/run_034/Learner"

checkpoint = tf.train.latest_checkpoint(model_path)
# "models/run_109/Learner/model-1642786.cptk" といったパス

load_tensors_in_checkpoint_file(checkpoint)

