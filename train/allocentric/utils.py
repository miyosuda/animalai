# -*- coding: utf-8 -*-
import tensorflow as tf
import os


def load_checkpoints(sess, save_dir):
    saver = tf.train.Saver(max_to_keep=2)
    checkpoint_dir = save_dir + "/checkpoints"

    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
    if checkpoint and checkpoint.model_checkpoint_path:
        # load from checkpoint.
        saver.restore(sess, checkpoint.model_checkpoint_path)
        # Retrieve step count from the file name.
        tokens = checkpoint.model_checkpoint_path.split("-")
        step = int(tokens[1])
        print("Loaded checkpoint: {0}, step={1}".format(
            checkpoint.model_checkpoint_path, step))
        return saver, step + 1
    else:
        print("Could not find old checkpoint")
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        return saver, 0


def save_checkponts(sess, saver, global_step, save_dir):
    checkpoint_dir = save_dir + "/checkpoints"
    saver.save(sess, checkpoint_dir + '/' + 'checkpoint', global_step=global_step)
    print("Checkpoint saved")
