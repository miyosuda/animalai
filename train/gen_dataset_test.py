# -*- coding: utf-8 -*-
import numpy as np
import unittest
import cv2

DATA_DIR = "data"
DEBUG_SAVE_IMAGE = True

class GenDatasetTest(unittest.TestCase):
    def test_generate(self):
        data_info_path = "{}/infos.npz".format(DATA_DIR)
        data_info_all = np.load(data_info_path)

        data_state_path = "{}/states.npz".format(DATA_DIR)
        data_state_all = np.load(data_state_path)
        
        data_actions   = data_info_all["actions"]
        data_positions = data_info_all["positions"]
        data_angles    = data_info_all["angles"]
        data_rewards   = data_info_all["rewards"]

        data_states    = data_state_all["states"]
        
        self.assertEqual(data_actions.shape,    (1400, 20, 2))
        self.assertEqual(data_positions.shape,  (1400, 20, 3))
        self.assertEqual(data_angles.shape,     (1400, 20, 1))
        self.assertEqual(data_rewards.shape,    (1400, 20, 1))
        
        self.assertEqual(data_states.shape,     (1400, 20, 84, 84, 3))
        
        if DEBUG_SAVE_IMAGE:
            seq_id = 10
        
            for i in range(20):
                state = data_states[seq_id,i]
                state = cv2.cvtColor(state, cv2.COLOR_RGB2BGR)
                cv2.imwrite("state_{0:02}_{1:02}.png".format(seq_id, i), state)

if __name__ == '__main__':
    unittest.main()

