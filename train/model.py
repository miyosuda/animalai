# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np


class AllocentricModel(object):
    def __init__(self, seq_length, batch_size, reuse=False):
        self.seq_length = seq_length
        
        self.step_size = tf.placeholder(tf.float32, [1])
        
        with tf.variable_scope("model", reuse=reuse) as scope:
            self.state_input    = tf.placeholder("float", [None, seq_length, 84, 84, 3])
            self.action_input   = tf.placeholder("float", [None, seq_length, 2, 2])
            self.velocity_input = tf.placeholder("float", [None, seq_length, 2, 3])
            
            # batch * seq_lengthをまとめる
            state_input_reshaped    = tf.reshape(self.state_input,  [-1, 84, 84, 3])
            action_input_reshaped   = tf.reshape(self.action_input, [-1, 4]) # 2x2の部分もまとめる
            velocity_input_reshaped = tf.reshape(self.velocity_input, [-1, 3])
            
            with tf.variable_scope("conv", reuse=reuse) as scope:
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
                
                # Store to send to upper layer.
                conv2_flat = tf.layers.flatten(conv2)
                
                fc1 = tf.layers.dense(conv2_flat,
                                      256,
                                      activation=tf.nn.relu,
                                      name="fc1")
                
                lstm_input = tf.concat([fc1, action_input_reshaped, velocity_input_reshaped], 1)
                # (-1, 263)
                
                self.cell = tf.contrib.rnn.BasicLSTMCell(256, state_is_tuple=True)

                # TODO: バッチサイズの扱い
                self.initial_lstm_state0 = tf.placeholder(tf.float32, [batch_size, 256])
                self.initial_lstm_state1 = tf.placeholder(tf.float32, [batch_size, 256])
                
                self.initial_lstm_state = tf.contrib.rnn.LSTMStateTuple(
                    self.initial_lstm_state0,
                    self.initial_lstm_state1)
                
                # TODO: バッチサイズの扱い
                lstm_input_reshaped = tf.reshape(lstm_input, [batch_size, -1, 263])
                # (batch_size, -1, 263)
                
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
                
                fc_out1 = tf.layers.dense(lstm_outputs,
                                          256,
                                          activation=tf.nn.relu,
                                          name="fc_out1")
                self.output = tf.layers.dense(fc_out1,
                                              5,
                                              activation=tf.nn.tanh, # -1~1にする
                                              name="fc_out2")
                # (batch_size * unroll_step, 5)
                
    def prepare_loss(self):
        with tf.variable_scope("loss") as scope:
            # x,y,zを-1~1にnormalizeしたinput
            self.postion_input = tf.placeholder("float", [None, self.seq_length, 3])
            # angleのcos, sinを入力
            self.angle_input   = tf.placeholder("float", [None, self.seq_length, 2])
                
            target_inputs = tf.concat([self.postion_input, self.angle_input], 2)
            # (batch_size, seq_length, 5)
            target_inputs_reshaped = tf.reshape(target_inputs, [-1, 5])
            # (batch_size * seq_length, 5)

            self.loss = tf.nn.l2_loss(target_inputs_reshaped - self.output)
