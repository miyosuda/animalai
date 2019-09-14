# -*- coding: utf-8 -*-
import numpy as np
import unittest
from estimator import AllocentricEstimator


class AllocentricEstimatorTest(unittest.TestCase):
    def test_init(self):
        estimator = AllocentricEstimator("saved")
        
        state = np.ones((84, 84, 3), dtype=np.float32)
        last_action = np.array([0,0], dtype=np.int32)
        last_velocity = np.zeros((3,), dtype=np.float32)

        position, angle = estimator.estimate(state, last_action, last_velocity)
        self.assertEqual(position.shape, (3,))
        self.assertEqual(angle.shape, ())

        position, angle = estimator.estimate(state, last_action, last_velocity)
        self.assertEqual(position.shape, (3,))
        self.assertEqual(angle.shape, ())

        estimator.reset()

        position, angle = estimator.estimate(state, last_action, last_velocity)
        self.assertEqual(position.shape, (3,))
        self.assertEqual(angle.shape, ())                

if __name__ == '__main__':
    unittest.main()
