# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import random

DATA_DIR = "data"


class DataManager(object):
    def __init__(self):
        self.train_index_pos = -1

        data_state_path = "{}/states.npz".format(DATA_DIR)
        data_state_all = np.load(data_state_path)        

        data_info_path = "{}/infos.npz".format(DATA_DIR)
        data_info_all = np.load(data_info_path)

        data_states     = data_state_all["states"]    # (1400, 20, 84, 84, 3) uint8
        data_actions    = data_info_all["actions"]    # (1400, 20, 2) float32
        data_velocities = data_info_all["velocities"] # (1400, 20, 3) float32
        data_positions  = data_info_all["positions"]  # (1400, 20, 3) float32
        data_angles     = data_info_all["angles"]     # (1400, 20, 1) float32
        data_rewards    = data_info_all["rewards"]    # (1400, 20, 1) int32

        # Get data dimensions
        total_data_size, self.seq_length, self.w, self.h, self.ch = data_states.shape
        self.train_data_size = 1200
        self.test_data_size  = 200

        # TODO: split_train_test_data()を利用する
        
        # State (uint8)
        self.raw_train_states = data_states[0:self.train_data_size] # (1200, 20, 64, 64, 3)
        self.raw_test_states  = data_states[self.train_data_size:]  # (200, 20, 64, 64, 3)

        # Actions
        self.train_actions    = data_actions[0:self.train_data_size]
        self.test_actions     = data_actions[self.train_data_size:]

        # Velocities.
        self.train_velocities = data_velocities[0:self.train_data_size]
        self.test_velocities  = data_velocities[self.train_data_size:]

        # Positions.
        self.train_positions  = data_positions[0:self.train_data_size]
        self.test_positions   = data_positions[self.train_data_size:]

        # Angles
        self.train_angles     = data_angles[0:self.train_data_size]
        self.test_angles      = data_angles[self.train_data_size:]

        # Rewards
        self.train_rewards    = data_rewards[0:self.train_data_size]
        self.test_rewards     = data_rewards[self.train_data_size:]
        
    def convert_states(self, states):
        return states.astype(np.float32) / 255.0

    def get_next_train_batch(self, batch_size):
        if self.train_index_pos < 0 or \
           self.train_index_pos + batch_size > self.train_data_size:
            self.train_indices = list(range(self.train_data_size))
            random.shuffle(self.train_indices)
            self.train_index_pos = 0
        selected_indices = self.train_indices[self.train_index_pos:
                                              self.train_index_pos+batch_size]
        raw_states = self.raw_train_states[selected_indices, :, :, :, :]
        states     = self.convert_states(raw_states)
        actions    = self.train_actions[selected_indices, :, :]
        velocities = self.train_velocities[selected_indices, :, :]
        positions  = self.train_positions[selected_indices, :, :]
        angles     = self.train_angles[selected_indices, :, :]
        rewards    = self.train_rewards[selected_indices, :, :]
        
        self.train_index_pos += batch_size
        return (states, actions, velocities, positions, angles, rewards)

    def get_test_batch(self, data_index, batch_size):
        indices = list(range(data_index, data_index + batch_size))

        raw_states = self.raw_test_states[indices, :, :, :, :]
        states     = self.convert_states(raw_states)
        actions    = self.test_actions[indices, :, :]
        velocities = self.test_velocities[indices, :, :]
        positions  = self.test_positions[indices, :, :]
        angles     = self.test_angles[indices, :, :]
        rewards    = self.test_rewards[indices, :, :]
        return (states, actions, velocities, positions, angles, rewards)

    def split_train_test_data(self, data, train_data_size):
        # Split data into train/test.
        train_data = data[0:train_data_size]
        test_data = data[train_data_size:]
        return train_data, test_data
