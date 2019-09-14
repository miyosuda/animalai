# -*- coding: utf-8 -*-
import numpy as np
import unittest
from utils import normalize_position, denormalie_position


class UtilsTest(unittest.TestCase):
    def test_normalize_position(self):
        positions = np.array([[[0.0, 0.0, 0.0]]], dtype=np.float32)
        normalized_positions = normalize_position(positions)
        positions_ = denormalie_position(normalized_positions)
        np.testing.assert_allclose(positions, positions_)
        
        positions = np.array([[[40.0, 0.0, 40.0]]], dtype=np.float32)
        normalized_positions = normalize_position(positions)
        positions_ = denormalie_position(normalized_positions)
        print(normalized_positions)
        print(positions_)
        np.testing.assert_allclose(positions, positions_)

        positions = np.array([[[20.0, 0.0, 20.0]]], dtype=np.float32)
        normalized_positions = normalize_position(positions)
        positions_ = denormalie_position(normalized_positions)
        print(normalized_positions)
        print(positions_)
        np.testing.assert_allclose(positions, positions_)


if __name__ == '__main__':
    unittest.main()
