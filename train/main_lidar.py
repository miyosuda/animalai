# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os

from lidar.data_manager import DataManager
from lidar.model import LidarModel
from lidar.trainer import Trainer
from lidar import options
from lidar import utils


flags = options.get_options()


def train(sess, trainer, saver, summary_writer, start_step):
    # Save command line args
    options.save_flags(flags)

    for i in range(start_step, flags.steps):
        # Train
        trainer.train(
           sess, summary_writer, batch_size=flags.batch_size, step=i)

        if i % flags.save_interval == flags.save_interval - 1:
            # Save
            utils.save_checkponts(sess, saver, i, flags.save_dir)

        if i % flags.test_interval == flags.test_interval - 1:
            # Test
            trainer.test(sess, summary_writer, batch_size=flags.batch_size, step=i)


def main(argv):
    if not os.path.exists(flags.save_dir):
        os.mkdir(flags.save_dir)

    batch_size = flags.batch_size
    data_manager = DataManager()
    seq_length = data_manager.seq_length
    weight_decay = flags.weight_decay

    model = LidarModel(seq_length, batch_size, weight_decay=weight_decay)
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
    max_to_keep = 0
    saver, start_step = utils.load_checkpoints(sess,
                                               flags.save_dir,
                                               max_to_keep=max_to_keep)

    train(sess, trainer, saver, summary_writer, start_step)

if __name__ == '__main__':
    tf.app.run()
