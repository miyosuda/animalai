# -*- coding: utf-8 -*-
import numpy as np
import unittest

from estimator import LidarEstimator, MultiLidarEstimator
from animalai.envs.brain import BrainInfo

"""
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
"""


class MultiLidarEstimatorTest(unittest.TestCase):
    def test_init(self):
        n_arenas = 4
        estimator = MultiLidarEstimator("saved_lidar", n_arenas=n_arenas)

        states = np.ones((n_arenas, 84, 84, 3), dtype=np.float32)
        states[0,:,:,:] = 0.0
        last_actions = [np.zeros((2,), dtype=np.int32) for _ in range(n_arenas)]
        last_velocities = [np.ones((3,), dtype=np.float32) for _ in range(n_arenas)]
        
        local_dones = [False, False, False]
        local_dones2 = [False, False, False]

        brain_info = BrainInfo(visual_observation=states,
                               vector_observation=last_velocities,
                               text_observations=None,
                               local_done=local_dones,
                               vector_action=last_actions)
        brain_info2 = BrainInfo(visual_observation=states,
                                vector_observation=last_velocities,
                                text_observations=None,
                                local_done=local_dones2,
                                vector_action=last_actions)

        id_probs, distances, valids = estimator.estimate(brain_info)
        self.assertEqual(id_probs.shape, (n_arenas, 5, 13))
        self.assertEqual(distances.shape, (n_arenas, 5))
        self.assertEqual(len(valids), n_arenas)
        for i in range(n_arenas):
            if i == 0:
                self.assertEqual(valids[i], False)
            else:
                self.assertEqual(valids[i], True)
        
        id_probs, distances, valids = estimator.estimate(brain_info)
        self.assertEqual(id_probs.shape, (n_arenas, 5, 13))
        self.assertEqual(distances.shape, (n_arenas, 5))
        self.assertEqual(len(valids), n_arenas)
        for i in range(n_arenas):
            if i == 0:
                self.assertEqual(valids[i], False)
            else:
                self.assertEqual(valids[i], True)        
        
        id_probs, distances, valids = estimator.estimate(brain_info2)
        self.assertEqual(id_probs.shape, (n_arenas, 5, 13))
        self.assertEqual(distances.shape, (n_arenas, 5))
        self.assertEqual(len(valids), n_arenas)
        for i in range(n_arenas):
            if i == 0:
                self.assertEqual(valids[i], False)
            else:
                self.assertEqual(valids[i], True)
        
if __name__ == '__main__':
    unittest.main()
