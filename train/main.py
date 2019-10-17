# -*- coding: utf-8 -*-
from animalai.envs import UnityEnvironment
from animalai.envs.exception import UnityEnvironmentException
from animalai.envs.arena_config import ArenaConfig

#from trainers.trainer_controller import TrainerController
from trainers.trainer_controller2 import TrainerController

import random
import yaml
import sys


# ML-agents parameters for training
run_id           = 'run_102'
sub_id           = 1
seed             = 1
run_seed         = 2

keep_checkpoints = 10000
env_path         = '../env/AnimalAI'
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

n_arenas         = 16
arena_config_path= 'configs/3-Obstacles.yaml'


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
tc.start_learning(env, trainer_config)
