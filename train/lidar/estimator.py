# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

from lidar.model import LidarModel
from lidar import utils


class LidarEstimator(object):
    def __init__(self, save_dir="saved_lidar"):
        self.model = LidarModel(seq_length=1, batch_size=1)
        
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        
        # Load checkpoints
        _, _ = utils.load_checkpoints(self.sess, save_dir)

        self.reset()
        
    def estimate(self, state, last_action, last_velocity):
        # (84,84,3) (2), (3,)
        state = state.reshape([1,1,84,84,3])
        last_action = last_action.reshape([1,1,2])
        last_velocity = last_velocity.reshape([1,1,3])
        
        feed_dict = {
            self.model.state_input: state,
            self.model.action_input: last_action,
            self.model.velocity_input: last_velocity
        }
        
        if self.lstm_state is not None:
            feed_dict.update({
                self.model.initial_lstm_state[0]:self.lstm_state[0],
                self.model.initial_lstm_state[1]:self.lstm_state[1]
            })
            
        id_probs, distances, lstm_state = self.sess.run(
            [self.model.id_prob_output,
             self.model.distance_output,
             self.model.lstm_state],
            feed_dict=feed_dict,
        )
        # (1,65) (1,5)

        id_probs = id_probs.reshape([-1, LidarModel.LIDAR_RAY_SIZE, LidarModel.TARGET_ID_MAX])
        # (1, 5, 13)
        
        self.lstm_state = lstm_state
        return id_probs[0], distances[0]
        
    def reset(self):
        self.lstm_state = None
