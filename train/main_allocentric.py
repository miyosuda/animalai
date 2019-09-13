# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os

from allocentric.data_manager import DataManager
from allocentric.model import AllocentricModel
from allocentric.trainer import Trainer
from allocentric import options

flags = options.get_options()


def load_checkpoints(sess):
    saver = tf.train.Saver(max_to_keep=2)
    checkpoint_dir = flags.save_dir + "/checkpoints"

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


def save_checkponts(sess, saver, global_step):
    checkpoint_dir = flags.save_dir + "/checkpoints"
    saver.save(
        sess, checkpoint_dir + '/' + 'checkpoint', global_step=global_step)
    print("Checkpoint saved")


def train(sess, trainer, saver, summary_writer, start_step):
    # Save command line args
    options.save_flags(flags)

    for i in range(start_step, flags.steps):
        # Train
        trainer.train(
           sess, summary_writer, batch_size=flags.batch_size, step=i)

        if i % flags.save_interval == flags.save_interval - 1:
            # Save
            save_checkponts(sess, saver, i)

        if i % flags.test_interval == flags.test_interval - 1:
            # Test
            trainer.test(sess, summary_writer, batch_size=flags.batch_size, step=i)


def main(argv):
    if not os.path.exists(flags.save_dir):
        os.mkdir(flags.save_dir)

    batch_size = flags.batch_size
    data_manager = DataManager()
    seq_length = data_manager.seq_length

    model = AllocentricModel(seq_length, batch_size)
    model.prepare_loss()
    
    trainer = Trainer(data_manager,
                      model, 
                      flags.learning_rate)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # For Tensorboard log
    log_dir = flags.save_dir + "/log"
    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

    # Load checkpoints
    saver, start_step = load_checkpoints(sess)

    train(sess, trainer, saver, summary_writer, start_step)

if __name__ == '__main__':
    tf.app.run()
