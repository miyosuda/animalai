# -*- coding: utf-8 -*-
import numpy as np
import unittest

from estimator import LidarEstimator


class LidarEstimatorTest(unittest.TestCase):
    def test_init(self):
        estimator = LidarEstimator("saved_lidar")

        state = np.ones((84, 84, 3), dtype=np.float32)
        last_action = np.array([0,0], dtype=np.int32)
        last_velocity = np.zeros((3,), dtype=np.float32)
        
        id_probs, distances = estimator.estimate(state, last_action, last_velocity)
        self.assertEqual(id_probs.shape, (5, 13))
        self.assertEqual(distances.shape, (5,))
        
        id_probs, distances = estimator.estimate(state, last_action, last_velocity)
        self.assertEqual(id_probs.shape, (5, 13))
        self.assertEqual(distances.shape, (5,))
        
        estimator.reset()
        
        id_probs, distances = estimator.estimate(state, last_action, last_velocity)
        self.assertEqual(id_probs.shape, (5, 13))
        self.assertEqual(distances.shape, (5,))
        
        
if __name__ == '__main__':
    unittest.main()
