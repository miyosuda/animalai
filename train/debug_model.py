# -*- coding: utf-8 -*-
import random
import yaml
import sys
import numpy as np

import copy

from animalai.envs import UnityEnvironment
from animalai.envs.exception import UnityEnvironmentException
from animalai.envs.arena_config import ArenaConfig
from trainers.trainer_controller import TrainerController


run_id           = 'run_005'
sub_id           = 1
keep_checkpoints = 10000
env_path         = '../env/AnimalAIFast'
base_port        = 5005
save_freq        = 20000
load_model       = False
train_model      = True
lesson           = 0
summaries_dir    = './summaries'
model_path       = './models/999'
n_arenas         = 4

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


def init_environment(env_path):
    # Strip out executable extensions if passed
    env_path = (env_path.strip()
                .replace('.app', '')
                .replace('.exe', '')
                .replace('.x86_64', '')
                .replace('.x86', ''))

    return UnityEnvironment(
        n_arenas=n_arenas,  # Change this to train on more arenas
        file_name=env_path,
        worker_id=0,
        seed=0,
        docker_training=False,
        play=False
    )

def get_modified_brain_parameters(brain):
    """ 初期化時にBrainParametersにVisitedMapの画像入力の定義を追加する """
    # 元のものは利用せずコピーを返す
    modified_brain = copy.copy(brain)
    
    # マップ情報用のcamera定義を追加
    modified_brain.number_visual_observations += 1

    # 84x84, 1chのカメラを追加する場合
    extra_camera_parameters = {
        'height': 84,
        'width': 84,
        'blackAndWhite': True
    }
    modified_brain.camera_resolutions.append(extra_camera_parameters)
    return modified_brain


def modify_brain_info(brain_info):
    """ 各stepにおいてBrainInfoにvisited mapを追加する """
    # 全アリーナ分visited mapを用意し、brain_info.visual_observationsに追加する
    visited_map_images = np.zeros((4, 84, 84, 1), dtype=np.float32)
    brain_info.visual_observations.append(visited_map_images)


arena_config_in = ArenaConfig('configs/1-Food.yaml')
trainer_config = load_config('configs/trainer_config.yaml')
env = init_environment(env_path)

from trainers.ppo.policy import PPOPolicy

brain = env.brains["Learner"] # animalai.envs.brain.BrainParameters

# envの中のbrainはそのままにして、
# envのbrainをコピーして追加camera情報を追加したものをtrainer_controllerに与える.
modified_brain = get_modified_brain_parameters(brain)

trainer_params = trainer_config['Learner'].copy()
trainer_params['summary_path']     = "./tmp"
trainer_params['model_path']       = "./tmp"
trainer_params['keep_checkpoints'] = 1

policy = PPOPolicy(seed=0,
                   brain=modified_brain,
                   trainer_params=trainer_params,
                   is_training=True,
                   load=False)

# マイステップのBrainInfoの取得の例として、reset時に得られるBrainInfoを利用
brain_info = env.reset(arenas_configurations=arena_config_in)

modify_brain_info(brain_info['Learner'])

run_out = policy.evaluate(brain_info['Learner'])
print(run_out)
