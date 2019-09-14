# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

from allocentric.data_manager import DataManager
from allocentric.model import AllocentricModel
from allocentric import options
from allocentric import utils


flags = options.get_options()

def check(sess, model, data_manager, batch_size):
    batch_data = data_manager.get_test_batch(0, batch_size)
    states, actions, velocities, positions, angles, rewards = batch_data

    positions = utils.normalize_position(positions) # -1.0~1.0
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)
    converted_angles = np.concatenate([cos_angles, sin_angles], 2)
    
    output = sess.run(
        model.output,
        feed_dict={
            model.state_input: states,
            model.action_input: actions,
            model.velocity_input: velocities
        })
    output_positions        = output[:,:3]
    output_convreted_angles = output[:,3:]

    output_cos_angles = output_convreted_angles[:,0]
    output_sin_angles = output_convreted_angles[:,1]
    output_angles = np.arctan2(output_sin_angles, output_cos_angles)
    
    print("positions={}".format(positions[0]))
    print("output_positions={}".format(output_positions))
    
    print("angles={}".format(angles[0,:,0]))
    print("output_angles={}".format(output_angles))
    

def main(argv):
    batch_size = 1
    save_dir = flags.save_dir
    
    data_manager = DataManager()
    seq_length = data_manager.seq_length
    
    model = AllocentricModel(seq_length, batch_size)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    # Load checkpoints
    _, _ = utils.load_checkpoints(sess, save_dir)
    
    check(sess, model, data_manager, batch_size)

if __name__ == '__main__':
    tf.app.run()
