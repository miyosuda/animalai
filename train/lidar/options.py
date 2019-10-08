# -*- coding: utf-8 -*-
import tensorflow as tf
from datetime import datetime as dt

# Key of the flags to ingore
ignore_keys = set(["h", "help", "helpfull", "helpshort"])


def get_options():
    tf.app.flags.DEFINE_string("save_dir", "saved",
                               "checkpoints,log,options save directory")
    tf.app.flags.DEFINE_float("learning_rate", 1e-4, "learning rate")
    tf.app.flags.DEFINE_integer("steps", 10 * (10**5), "training steps")
    tf.app.flags.DEFINE_integer("save_interval", 5000, "saving interval")
    tf.app.flags.DEFINE_integer("test_interval", 10000, "test interval")
    tf.app.flags.DEFINE_integer("batch_size", 10, "batch size")
    tf.app.flags.DEFINE_boolean("training", True, "whether to train or not")
    tf.app.flags.DEFINE_string("desc", "normal experiment", "experiment description")
    return tf.app.flags.FLAGS


def save_flags(flags):
    dic = flags.__flags

    lines = []

    # Record current time
    time_str = dt.now().strftime('# %Y-%m-%d %H:%M')
    lines.append(time_str + "\n")

    for key in sorted(dic.keys()):
        if key in ignore_keys:
            # Keys like "helpfull" are ignored
            continue
        value = dic[key].value
        line = "{}={}".format(key, value)
        lines.append(line + "\n")

    file_name = flags.save_dir + "/options.txt"
    f = open(file_name, "w")
    f.writelines(lines)
    f.close()
