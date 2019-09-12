# -*- coding: utf-8 -*-
import numpy as np
import unittest
from model import AllocentricModel


class AllocentricModelTest(unittest.TestCase):
    def test_init(self):
        seq_length = 20
        batch_size = 5
        model = AllocentricModel(seq_length, batch_size)


if __name__ == '__main__':
    unittest.main()
