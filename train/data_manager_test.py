# -*- coding: utf-8 -*-
import numpy as np
import unittest
from data_manager import DataManager

DEBUG_SAVE_STATE = False

if DEBUG_SAVE_STATE:
    from scipy.misc import imsave


class DataManagerTest(unittest.TestCase):
    def sub_test_get_next_train_batch(self, data_manager, seq_length=20):
        batch_data = data_manager.get_next_train_batch(10)
        (states, actions, positions, angles, rewards) = batch_data

        self.assertEqual(states.shape, (10, seq_length, 84, 84, 3))
        self.assertEqual(states.dtype, "float32")

        self.assertEqual(actions.shape, (10, seq_length, 2))
        self.assertEqual(actions.dtype, "int32")

        self.assertEqual(positions.shape, (10, seq_length, 3))
        self.assertEqual(positions.dtype, "float32")

        self.assertEqual(angles.shape, (10, seq_length, 1))
        self.assertEqual(angles.dtype, "float32")

        self.assertEqual(rewards.shape, (10, seq_length, 1))
        self.assertEqual(rewards.dtype, "float32")


        if DEBUG_SAVE_STATE:
            for j in range(2):
                for i in range(seq_length):
                    img = states[j][i]
                    imsave("train_batch{0:0>2}{1:0>2}.png".format(j, i), img)

    def sub_test_get_test_batch(self, data_manager, seq_length=20):
        batch_data = data_manager.get_test_batch(0, 10)
        (states, actions, positions, angles, rewards) = batch_data

        self.assertEqual(states.shape, (10, seq_length, 84, 84, 3))
        self.assertEqual(states.dtype, "float32")

        self.assertEqual(actions.shape, (10, seq_length, 2))
        self.assertEqual(actions.dtype, "int32")

        self.assertEqual(positions.shape, (10, seq_length, 3))
        self.assertEqual(positions.dtype, "float32")

        self.assertEqual(angles.shape, (10, seq_length, 1))
        self.assertEqual(angles.dtype, "float32")

        self.assertEqual(rewards.shape, (10, seq_length, 1))
        self.assertEqual(rewards.dtype, "float32")

        if DEBUG_SAVE_STATE:
            for j in range(2):
                for i in range(seq_length):
                    img = states[j][i]
                    imsave("test_batch{0:0>2}{1:0>2}.png".format(j, i), img)

    def sub_test_common_properties(self,
                                   data_manager,
                                   seq_length=10,
                                   train_data_size=1200,
                                   test_data_size=200):

        # statesはfloat32になっている
        self.assertEqual(data_manager.raw_train_states.dtype, "uint8")
        self.assertEqual(data_manager.raw_test_states.dtype, "uint8")

        # statesのshape確認
        self.assertEqual(data_manager.raw_train_states.shape,
                         (train_data_size, seq_length, 84, 84, 3))
        self.assertEqual(data_manager.raw_test_states.shape,
                         (test_data_size, seq_length, 84, 84, 3))

        # statesの値の範囲確認
        self.assertLessEqual(np.amax(data_manager.raw_train_states), 255)
        self.assertGreaterEqual(np.amin(data_manager.raw_train_states), 0)
        self.assertLessEqual(np.amax(data_manager.raw_test_states), 255)
        self.assertGreaterEqual(np.amin(data_manager.raw_test_states), 0)

        # フィールドの確認
        self.assertEqual(data_manager.train_data_size, train_data_size)
        self.assertEqual(data_manager.test_data_size, test_data_size)
        self.assertEqual(data_manager.seq_length, seq_length)
        self.assertEqual(data_manager.w, 84)
        self.assertEqual(data_manager.h, 84)

    def test_init(self):
        seq_length = 20
        train_data_size = 1200
        test_data_size = 200

        # 共通のテスト
        data_manager = DataManager()

        self.sub_test_common_properties(data_manager,
                                        seq_length,
                                        train_data_size,
                                        test_data_size)

        if DEBUG_SAVE_STATE:
            for i in range(20):
                img = data_manager.raw_train_states[0][i]
                imsave("out{}.png".format(i), img)

        self.sub_test_get_next_train_batch(data_manager, seq_length)
        self.sub_test_get_test_batch(data_manager, seq_length)


if __name__ == '__main__':
    unittest.main()
