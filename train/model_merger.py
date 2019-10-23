# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

from animalai.envs import UnityEnvironment
from animalai.envs.exception import UnityEnvironmentException
from animalai.envs.arena_config import ArenaConfig

from trainers.trainer_controller2 import TrainerController

import os

import random
import yaml
import sys


# ML-agents parameters for training
run_id           = 'run_999'
sub_id           = 1
seed             = 1
run_seed         = 2

keep_checkpoints = 10000
env_path         = '../env/AnimalAIFast'
worker_id        = random.randint(1, 100)
base_port        = 5005
save_freq        = 20000
curriculum_file  = None
load_model       = False
train_model      = True
lesson           = 0

docker_target_name = None
trainer_config_path = 'configs/trainer_config_rec.yaml'
model_path       = './models/{run_id}'.format(run_id=run_id)
summaries_dir    = './summaries'

n_arenas         = 1
arena_config_path= 'configs/1-Food-arena16-t250-r5.yaml'


def load_config(trainer_config_path):
    try:
        with open(trainer_config_path) as data_file:
            trainer_config = yaml.load(data_file)
            return trainer_config
    except IOError:
        raise UnityEnvironmentException('Parameter file could not be found '
                                        'at {}.'
                                        .format(trainer_config_path))
    except UnicodeDecodeError:
        raise UnityEnvironmentException('There was an error decoding '
                                        'Trainer Config from this path : {}'
                                        .format(trainer_config_path))


def init_environment(env_path,
                     docker_target_name,
                     worker_id,
                     seed):
    if env_path is not None:
        # Strip out executable extensions if passed
        env_path = (env_path.strip()
                    .replace('.app', '')
                    .replace('.exe', '')
                    .replace('.x86_64', '')
                    .replace('.x86', ''))
    docker_training = docker_target_name is not None

    return UnityEnvironment(
        n_arenas=n_arenas,  # Change this to train on more arenas
        file_name=env_path,
        worker_id=worker_id,
        seed=seed,
        docker_training=docker_training,
        play=False
    )


arena_config_in = ArenaConfig(arena_config_path)
trainer_config = load_config(trainer_config_path)
env = init_environment(env_path,
                       docker_target_name,
                       worker_id,
                       run_seed)

external_brains = {}
for brain_name in env.external_brain_names:
    external_brains[brain_name] = env.brains[brain_name]

# Create controller and begin training.
tc = TrainerController(model_path,
                       summaries_dir, run_id + '-' + str(sub_id),
                       save_freq,
                       load_model,
                       train_model,
                       keep_checkpoints,
                       lesson,
                       external_brains,
                       run_seed,
                       arena_config_in)

tc._create_model_path(tc.model_path)
tf.reset_default_graph()
tc.initialize_trainers(trainer_config)

trainer = tc.trainers["Learner"]
policy = trainer.policy
graph = policy.graph


# 部分ロードするテンソル名 (":0"は抜いたもの)
partial_load_target_tensor_names = [
    "main_graph_0_encoder1/conv_1/bias",
    "main_graph_0_encoder1/conv_1/kernel",
    "main_graph_0_encoder1/conv_2/bias",
    "main_graph_0_encoder1/conv_2/kernel",
    "main_graph_0_encoder1/conv_3/bias",
    "main_graph_0_encoder1/conv_3/kernel",
    "main_graph_0_encoder1/flat_encoding/main_graph_0_encoder1/hidden_0/bias",
    "main_graph_0_encoder1/flat_encoding/main_graph_0_encoder1/hidden_0/kernel",
]

restore_dict = {}

checkpoint = tf.train.latest_checkpoint("models/run_109/Learner")

with graph.as_default():
    for v in tf.trainable_variables():
        tensor_name = v.name.split(':')[0]
        if tensor_name in partial_load_target_tensor_names:
            print("partial load: {}".format(tensor_name))
            restore_dict[tensor_name] = v

    saver = tf.train.Saver(restore_dict)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, checkpoint)

        save_dir = "merged_model"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        steps = 0
        last_checkpoint = save_dir + '/model-' + str(steps) + '.cptk'
        saver.save(sess, last_checkpoint)
        #tf.train.write_graph(graph, save_dir,
        #                     'raw_graph_def.pb', as_text=False)
