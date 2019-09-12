# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np


class AllocentricModel(object):
    def __init__(self, seq_length, batch_size, reuse=False):
        self.step_size = tf.placeholder(tf.float32, [1])
        
        with tf.variable_scope("model", reuse=reuse) as scope:
            self.state_input = tf.placeholder("float", [None, seq_length, 84, 84, 3])
            self.action_input = tf.placeholder("float", [None, seq_length, 2, 2])

            state_input_reshaped = tf.reshape(self.state_input, [-1, 84, 84, 3])
            action_input_reshaped = tf.reshape(self.action_input, [-1, 4])

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
                
                lstm_input = tf.concat([fc1, action_input_reshaped], 1)
                # (-1, 260)
                
                self.cell = tf.contrib.rnn.BasicLSTMCell(256, state_is_tuple=True)
                
                self.initial_lstm_state0 = tf.placeholder(tf.float32, [1, 256])
                self.initial_lstm_state1 = tf.placeholder(tf.float32, [1, 256])
                
                self.initial_lstm_state = tf.contrib.rnn.LSTMStateTuple(
                    self.initial_lstm_state0,
                    self.initial_lstm_state1)
                
                # TODO: バッチサイズの扱い
                lstm_input_reshaped = tf.reshape(lstm_input, [batch_size, -1, 260])

                #step_size = tf.shape(conv2_flat)[:1]

                lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                    self.cell,
                    lstm_input_reshaped,
                    initial_state = self.initial_lstm_state,
                    #sequence_length = step_size,
                    sequence_length = seq_length, # TODO: 扱い
                    time_major = False,
                    scope = scope) 
                # (batch_size, unroll_step, 260)
                
                """
                lstm_outputs = tf.reshape(lstm_outputs, [-1,256])
                #(1,unroll_step,256) for back prop, (1,1,256) for forward prop.
                """
                
