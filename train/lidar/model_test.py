# -*- coding: utf-8 -*-
import numpy as np
import unittest
from model import LidarModel


class LidarModelTest(unittest.TestCase):
    def test_init(self):
        seq_length = 20
        batch_size = 6
        
        model = LidarModel(seq_length, batch_size)
        
        model.prepare_loss()
        self.assertEqual(model.loss.shape, ())
        
        model_reused = LidarModel(seq_length, 1, reuse=True)
        
if __name__ == '__main__':
    unittest.main()
