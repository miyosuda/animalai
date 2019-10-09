# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np


class LidarModel(object):
    TARGET_ID_MAX = 13
    LIDAR_RAY_SIZE = 5
    
    def __init__(self, seq_length, batch_size, reuse=False):
        self.seq_length = seq_length
        
        self.step_size = tf.placeholder(tf.float32, [1])

        with tf.variable_scope("lidar_model", reuse=reuse) as scope:
            self.state_input    = tf.placeholder("float", [None, seq_length, 84, 84, 3])
            self.action_input   = tf.placeholder("float", [None, seq_length, 2])
            self.velocity_input = tf.placeholder("float", [None, seq_length, 3])
            
            # batch * seq_lengthをまとめる
            state_input_reshaped    = tf.reshape(self.state_input,  [-1, 84, 84, 3])
            action_input_reshaped   = tf.reshape(self.action_input, [-1, 2])
            velocity_input_reshaped = tf.reshape(self.velocity_input, [-1, 3])
            
            conv1 = tf.layers.conv2d(state_input_reshaped,
                                     filters=16,
                                     kernel_size=[4, 4],
                                     strides=(2, 2),
                                     padding="same",
                                     activation=tf.nn.relu,
                                     name="conv1")
            
            conv2 = tf.layers.conv2d(conv1,
                                     filters=32,
                                     kernel_size=[4, 4],
                                     strides=(2, 2),
                                     padding="same",
                                     activation=tf.nn.relu,
                                     name="conv2")
            # (-1, 21, 21, 32)
            conv3 = tf.layers.conv2d(conv2,
                                     filters=32,
                                     kernel_size=[4, 4],
                                     strides=(2, 2),
                                     padding="same",
                                     activation=tf.nn.relu,
                                     name="conv3")
            # (-1, 11, 11, 32)
            
            conv3_flat = tf.layers.flatten(conv3)
            # (-1, 3872)
            
            fc1 = tf.layers.dense(conv3_flat,
                                  256,
                                  activation=tf.nn.relu,
                                  name="fc1")
                
            lstm_input = tf.concat([fc1, action_input_reshaped, velocity_input_reshaped], 1)
            # (-1, 261)
                
            self.cell = tf.contrib.rnn.BasicLSTMCell(256, state_is_tuple=True)

            self.initial_lstm_state = self.cell.zero_state(batch_size=batch_size,
                                                           dtype=tf.float32)
                
            # TODO: バッチサイズの扱い
            lstm_input_reshaped = tf.reshape(lstm_input, [batch_size, -1, 261])
            # (batch_size, -1, 261)
                
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                self.cell,
                lstm_input_reshaped,
                initial_state = self.initial_lstm_state,
                time_major = False,
                scope = scope) 
            # (batch_size, unroll_step, 256)
            # ((batch_size, 256), (batch_size, 256))
                
            lstm_outputs = tf.reshape(lstm_outputs, [-1,256])
            # (batch_size * unroll_step, 256)

            # (batch_size * unroll_step, 5)
            self.lstm_state = lstm_state
                
            fc_out1 = tf.layers.dense(lstm_outputs,
                                      256,
                                      activation=tf.nn.relu,
                                      name="fc_out1")
            self.id_logits = []
            id_probs = []
            
            for i in range(LidarModel.LIDAR_RAY_SIZE):
                id_logit = tf.layers.dense(fc_out1,
                                           LidarModel.TARGET_ID_MAX,
                                           activation=None,
                                           name="fc_id_logit{}".format(i))
                self.id_logits.append(id_logit)
                id_probs.append(tf.nn.softmax(id_logit))

            self.id_prob_output = tf.concat(id_probs, axis=1, name="all_id_probs")

            self.distance_output = tf.layers.dense(
                fc_out1,
                LidarModel.LIDAR_RAY_SIZE,
                activation=None,
                name="fc_distance")
            
    def prepare_loss(self):
        with tf.variable_scope("lidar_loss") as scope:
            self.id_input       = tf.placeholder(tf.int32, [None,
                                                            self.seq_length,
                                                            LidarModel.LIDAR_RAY_SIZE])
            self.distance_input = tf.placeholder(tf.float32, [None,
                                                              self.seq_length,
                                                              LidarModel.LIDAR_RAY_SIZE])
            
            id_input_reshaped = tf.reshape(self.id_input, [-1, LidarModel.LIDAR_RAY_SIZE])

            id_losses = []

            for i in range(LidarModel.LIDAR_RAY_SIZE):
                # 1本のRay毎にidのcross entropy loss計算
                id_input_oh = tf.one_hot(id_input_reshaped[:, i], LidarModel.TARGET_ID_MAX)
                id_logit = self.id_logits[i]
                id_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=id_input_oh,
                                                                     logits=id_logit)
                # (-1,)
                id_losses.append(id_loss)

            self.id_loss = tf.reduce_sum(tf.add_n(id_losses))
            # スカラーになっている
            
            distance_input_reshaped = tf.reshape(self.distance_input,
                                                 [-1, LidarModel.LIDAR_RAY_SIZE])
            self.distance_loss = tf.nn.l2_loss(distance_input_reshaped - self.distance_output)
            # ここはreduce_sumされてスカラーになっている
            
            self.loss = self.id_loss + self.distance_loss
