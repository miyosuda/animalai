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


class MultiLidarEstimator(object):
    def __init__(self, save_dir="saved_lidar", n_arenas=1):
        self.batch_size = n_arenas
        self.model = LidarModel(seq_length=1, batch_size=n_arenas)
        
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        
        # Load checkpoints
        _, _ = utils.load_checkpoints(self.sess, save_dir)

        self.reset()

    def estimate(self, brain_info):
        states          = brain_info.visual_observations
        last_actions    = brain_info.previous_vector_actions
        last_velocities = brain_info.vector_observations
        local_dones     = brain_info.local_done

        states          = np.array(states) # (1,n_arenas,84,84,3)
        last_actions    = np.array(last_actions,    np.int32)
        last_velocities = np.array(last_velocities, np.float32)

        states          = states.reshape([self.batch_size,1,84,84,3])
        last_actions    = last_actions.reshape([self.batch_size,1,2])
        last_velocities = last_velocities.reshape([self.batch_size,1,3])

        states_max = np.max(states, axis=(1,2,3,4)) # (n_arenas,)

        # 各Arenaの画面が0でないかどうか
        valids = [state_max > 0.0 for state_max in states_max]
        
        # local doneが立っている時は画像は新しいエピソードのものになっている.
        # last_action, last_velocityは、旧エピソードのものになっているので差し替える.
        for i, local_done in enumerate(local_dones):
            if local_done:
                # 該当のArenaのLSTM stateをゼロクリア
                self.lstm_state.c[i] = np.zeros((256,), dtype=np.float32)
                self.lstm_state.h[i] = np.zeros((256,), dtype=np.float32)
                last_actions[i] = np.zeros((1,2), dtype=np.int32)
                last_velocities[i] = np.zeros((1,3), dtype=np.float32)
                
        feed_dict = {
            self.model.state_input: states,
            self.model.action_input: last_actions,
            self.model.velocity_input: last_velocities,
            self.model.initial_lstm_state[0]:self.lstm_state[0],
            self.model.initial_lstm_state[1]:self.lstm_state[1],
        }
        
        id_probs, distances, lstm_state = self.sess.run(
            [self.model.id_prob_output,
             self.model.distance_output,
             self.model.lstm_state],
            feed_dict=feed_dict,
        )
        # (batch_size, 65) (batch_size, e5)
        
        id_probs = id_probs.reshape([-1, LidarModel.LIDAR_RAY_SIZE, LidarModel.TARGET_ID_MAX])
        # (batch_size, 5, 13)
        
        self.lstm_state = lstm_state
        return id_probs, distances, valids
        
    def reset(self):
        c = np.zeros((self.batch_size, 256), dtype=np.float32)
        h = np.zeros((self.batch_size, 256), dtype=np.float32)
        self.lstm_state = tf.contrib.rnn.LSTMStateTuple(c, h)
