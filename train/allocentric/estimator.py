# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

from .model import AllocentricModel
from . import utils


class AllocentricEstimator(object):
    def __init__(self, save_dir="saved"):
        self.model = AllocentricModel(seq_length=1, batch_size=1)
        
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
        
        output, lstm_state = self.sess.run(
            [self.model.output,
             self.model.lstm_state],
            feed_dict=feed_dict,
        )

        self.lstm_state = lstm_state
        
        output_positions        = output[:,:3]
        output_convreted_angles = output[:,3:]

        output_cos_angles = output_convreted_angles[:,0]
        output_sin_angles = output_convreted_angles[:,1]
        output_angles = np.arctan2(output_sin_angles, output_cos_angles)

        output_positions = utils.denormalie_position(output_positions)
        return output_positions[0], output_angles[0]
        
    def reset(self):
        self.lstm_state = None
