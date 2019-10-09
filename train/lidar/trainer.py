# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np


class Trainer(object):
    def __init__(self, data_manager, model, learning_rate):
        self.data_manager = data_manager
        self.model = model
        
        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(
            self.model.loss)

    def train(self, sess, summary_writer, batch_size, step):
        batch_data = self.data_manager.get_next_train_batch(batch_size)
        states, actions, velocities, positions, angles, rewards, target_ids, target_distances = batch_data
        
        # (batch_size, seq_length, 5)
        out = sess.run(
            [
                self.train_op,
                self.model.loss,
                self.model.id_loss,
                self.model.distance_loss
            ],
            feed_dict={
                self.model.state_input: states,
                self.model.action_input: actions,
                self.model.velocity_input: velocities,
                self.model.id_input: target_ids,
                self.model.distance_input: target_distances,
            })
        _, loss, id_loss, distance_loss = out

        if step % 10 == 0:
            self.record_loss(summary_writer, "train/loss", loss, step)
            self.record_loss(summary_writer, "train/id_loss", id_loss, step)
            self.record_loss(summary_writer, "train/distance_loss", distance_loss, step)

    def record_loss(self, summary_writer, tag, value, step):
        summary_str = tf.Summary(
            value=[tf.Summary.Value(tag=tag, simple_value=value)])
        summary_writer.add_summary(summary_str, step)

    def test(self, sess, summary_writer, batch_size, step):
        test_data_size = self.data_manager.test_data_size
        all_losses = []
        all_id_losses = []
        all_distance_losses = []

        # 端数が出た時の対処
        test_data_size = (test_data_size // batch_size) * batch_size

        for i in range(0, test_data_size, batch_size):
            batch_data = self.data_manager.get_test_batch(i, batch_size)
            states, actions, velocities, positions, angles, rewards, target_ids, target_distances = batch_data

            # (batch_size, seq_length, 5)

            # Not updating state because it is initialized with every batch.
            out = sess.run(
                [self.model.loss,
                 self.model.id_loss,
                 self.model.distance_loss],
                feed_dict={
                    self.model.state_input: states,
                    self.model.action_input: actions,
                    self.model.velocity_input: velocities,
                    self.model.id_input: target_ids,
                    self.model.distance_input: target_distances,
                })
            loss, id_loss, distance_loss = out
            all_losses.append(loss)
            all_id_losses.append(id_loss)
            all_distance_losses.append(distance_loss)

        mean_loss = np.mean(all_losses, axis=0)
        mean_id_loss = np.mean(all_id_losses, axis=0)
        mean_distance_loss = np.mean(all_distance_losses, axis=0)
        
        # Record summary
        self.record_loss(summary_writer, "test/loss", mean_loss, step)
        self.record_loss(summary_writer, "test/id_loss", mean_id_loss, step)
        self.record_loss(summary_writer, "test/distance_loss", mean_distance_loss, step)
        print("test loss={0:.2f}".format(mean_loss))
