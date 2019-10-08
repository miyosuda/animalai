# -*- coding: utf-8 -*-
import numpy as np
import unittest
import cv2

DATA_DIR = "lidar_data"
DEBUG_SAVE_IMAGE = False


class GenDatasetTest(unittest.TestCase):
    def test_generate(self):
        data_info_path = "{}/infos.npz".format(DATA_DIR)
        data_info_all = np.load(data_info_path)

        data_state_path = "{}/states.npz".format(DATA_DIR)
        data_state_all = np.load(data_state_path)
        
        data_actions          = data_info_all["actions"]
        data_velocities       = data_info_all["velocities"]
        data_positions        = data_info_all["positions"]
        data_angles           = data_info_all["angles"]
        data_rewards          = data_info_all["rewards"]
        data_target_ids       = data_info_all["target_ids"]
        data_target_distances = data_info_all["target_distances"]

        data_states = data_state_all["states"]

        data_size = 1400
                
        self.assertEqual(data_actions.shape,          (data_size, 20, 2))
        self.assertEqual(data_velocities.shape,       (data_size, 20, 3))
        self.assertEqual(data_positions.shape,        (data_size, 20, 3))
        self.assertEqual(data_angles.shape,           (data_size, 20, 1))
        self.assertEqual(data_rewards.shape,          (data_size, 20, 1))
        self.assertEqual(data_target_ids.shape,       (data_size, 20, 5))
        self.assertEqual(data_target_distances.shape, (data_size, 20, 5))
        self.assertEqual(data_states.shape,           (data_size, 20, 84, 84, 3))

        self.assertEqual(data_actions.dtype,          np.int32)
        self.assertEqual(data_velocities.dtype,       np.float32)
        self.assertEqual(data_positions.dtype,        np.float32)
        self.assertEqual(data_angles.dtype,           np.float32)
        self.assertEqual(data_rewards.dtype,          np.float32)
        self.assertEqual(data_target_ids.dtype,       np.int8)
        self.assertEqual(data_target_distances.dtype, np.float32)
        self.assertEqual(data_states.dtype,           np.uint8)
        
        if DEBUG_SAVE_IMAGE:
            seq_id = 10
        
            for i in range(20):
                state = data_states[seq_id,i]
                state = cv2.cvtColor(state, cv2.COLOR_RGB2BGR)
                cv2.imwrite("state_{0:02}_{1:02}.png".format(seq_id, i), state)

if __name__ == '__main__':
    unittest.main()
