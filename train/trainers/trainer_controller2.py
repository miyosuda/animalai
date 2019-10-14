# -*- coding: utf-8 -*-
# # Unity ML-Agents Toolkit
# ## ML-Agent Learning
"""Launches trainers for each External Brains in a Unity Environment."""

import os
import logging
import shutil
import sys
import copy
from enum import Enum, auto
if sys.platform.startswith('win'):
    import win32api
    import win32con
from typing import *

import numpy as np
import tensorflow as tf

import cv2

from animalai.envs import BrainInfo
from animalai.envs.exception import UnityEnvironmentException
from animalai.envs.arena_config import ArenaConfig

from trainers.ppo.trainer import PPOTrainer

from lidar.estimator import MultiLidarEstimator
from trainers.visited_map import VisitedMap


ENABLE_VISITED_MAP_IMAGE = True
DEBUG_SHOW_VISITED_MAP = False


class TrainerController(object):
    def __init__(self,
                 model_path: str,
                 summaries_dir: str,
                 run_id: str,
                 save_freq: int,
                 load: bool,
                 train: bool,
                 keep_checkpoints: int,
                 lesson: Optional[int],
                 external_brains: Dict[str, BrainInfo],
                 training_seed: int,
                 config=None):
        """
        Arguments:

        model_path: 
            Path to save the model.
        summaries_dir: 
            Folder to save training summaries.
        run_id: 
            The sub-directory name for model and summary statistics
        save_freq: 
            Frequency at which to save model
        load: 
            Whether to load the model or randomly initialize.
        train: 
            Whether to train model, or only run inference.
        keep_checkpoints: 
            How many model checkpoints to keep.
        lesson: 
            Start learning from this lesson.
        external_brains: 
            dictionary of external brain names to BrainInfo objects.
        training_seed: 
            Seed to use for Numpy and Tensorflow random number generation.
        """        

        self.model_path = model_path
        self.summaries_dir = summaries_dir
        self.external_brains = external_brains
        self.external_brain_names = external_brains.keys()
        self.logger = logging.getLogger('mlagents.envs')
        self.run_id = run_id
        self.save_freq = save_freq
        self.lesson = lesson
        self.load_model = load
        self.train_model = train
        self.keep_checkpoints = keep_checkpoints
        self.trainers = {}
        self.global_step = 0
        self.seed = training_seed
        self.config = config
        self.update_config = True
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)
        
        if ENABLE_VISITED_MAP_IMAGE:
            self.lidar_estimators = None
            self.extra_brain_infos = {}
            for brain_name in self.external_brain_names:
                self.external_brains[brain_name] = add_extra_camera_parameter(
                    self.external_brains[brain_name])
                self.extra_brain_infos[brain_name] = ExtraBrainInfo()

    def _get_measure_vals(self):
        return None

    def _save_model(self, steps=0):
        """
        Saves current model to checkpoint folder.
        :param steps: Current number of steps in training process.
        :param saver: Tensorflow saver for session.
        """
        for brain_name in self.trainers.keys():
            self.trainers[brain_name].save_model()
        self.logger.info('Saved Model')

    def _save_model_when_interrupted(self, steps=0):
        self.logger.info('Learning was interrupted. Please wait '
                         'while the graph is generated.')
        self._save_model(steps)

    def initialize_trainers(self, trainer_config):
        """
        Initialization of the trainers
        :param trainer_config: The configurations of the trainers
        """
        trainer_parameters_dict = {}

        for brain_name in self.external_brains:
            trainer_parameters = trainer_config['default'].copy()
            trainer_parameters['summary_path'] = '{basedir}/{name}'.format(
                basedir=self.summaries_dir,
                name=str(self.run_id) + '_' + brain_name)
            trainer_parameters['model_path'] = '{basedir}/{name}'.format(
                basedir=self.model_path,
                name=brain_name)
            trainer_parameters['keep_checkpoints'] = self.keep_checkpoints
            if brain_name in trainer_config:
                _brain_key = brain_name
                while not isinstance(trainer_config[_brain_key], dict):
                    _brain_key = trainer_config[_brain_key]
                for k in trainer_config[_brain_key]:
                    trainer_parameters[k] = trainer_config[_brain_key][k]
            trainer_parameters_dict[brain_name] = trainer_parameters.copy()
        for brain_name in self.external_brains:
            if trainer_parameters_dict[brain_name]['trainer'] == 'ppo':
                # ここで PPOTrainer 生成
                self.trainers[brain_name] = PPOTrainer(
                    self.external_brains[brain_name],
                    0,
                    trainer_parameters_dict[brain_name], # trainer_configで指定した内容
                    self.train_model,
                    self.load_model,
                    self.seed,
                    self.run_id)
            else:
                raise UnityEnvironmentException('The trainer config contains '
                                                'an unknown trainer type for '
                                                'brain {}'
                                                .format(brain_name))

    @staticmethod
    def _create_model_path(model_path):
        try:
            if not os.path.exists(model_path):
                os.makedirs(model_path)
        except Exception:
            raise UnityEnvironmentException('The folder {} containing the '
                                            'generated model could not be '
                                            'accessed. Please make sure the '
                                            'permissions are set correctly.'
                                            .format(model_path))

    def _reset_env(self, env):
        """Resets the environment.

        Returns:
            A Data structure corresponding to the initial reset state of the
            environment.
        """
        if self.update_config:
            self.update_config = False
            new_info = env.reset(arenas_configurations=self.config)
        else:
            new_info = env.reset()
        
        if ENABLE_VISITED_MAP_IMAGE:
            if self.lidar_estimators is None:
                # Create LIDAR estimator
                self.lidar_estimators = {}
                for brain_name in self.trainers.keys():
                    brain_info = new_info[brain_name]
                    n_arenas = len(brain_info.rewards)
                    self.lidar_estimators[brain_name] = MultiLidarEstimator(
                        save_dir="saved_lidar", # TODO: データパスの指定
                        n_arenas=n_arenas
                    )
            self.extra_brain_infos = {}
            for brain_name in self.trainers.keys():
                self.extra_brain_infos[brain_name] = ExtraBrainInfo() # delete old ExtraBrainInfo
                out = expand_brain_info(new_info[brain_name],
                                        self.extra_brain_infos[brain_name],
                                        self.lidar_estimators[brain_name])
                new_info[brain_name], self.extra_brain_infos[brain_name] = out
                self.lidar_estimators[brain_name].reset()
                
        return new_info

    def start_learning(self, env, trainer_config):
        # TODO: Should be able to start learning at different lesson numbers
        # for each curriculum.
        self._create_model_path(self.model_path)

        tf.reset_default_graph()

        # Prevent a single session from taking all GPU memory.
        self.initialize_trainers(trainer_config)
        
        for _, t in self.trainers.items():
            self.logger.info(t)

        curr_info = self._reset_env(env)

        # Tensorboardにハイパーパラメータを記録
        if self.train_model:
            for brain_name, trainer in self.trainers.items():
                trainer.write_tensorboard_text('Hyperparameters',
                                               trainer.parameters)
                
        try:
            # 学習ループ
            while any([t.get_step <= t.get_max_steps for k, t in self.trainers.items()]) \
                  or not self.train_model:
                # 学習を1ステップ進める
                new_info = self.take_step(env, curr_info)
                
                self.global_step += 1
                
                if self.global_step % self.save_freq == 0 and self.global_step != 0 \
                        and self.train_model:
                    # 学習モデルの保存
                    self._save_model(steps=self.global_step)
                curr_info = new_info
            # Final save Tensorflow model
            if self.global_step != 0 and self.train_model:
                # 最後にモデルを保存
                self._save_model(steps=self.global_step)
        except KeyboardInterrupt:
            if self.train_model:
                self._save_model_when_interrupted(steps=self.global_step)
            pass
        env.close()

    def take_step(self, env, curr_info):
        # If any lessons were incremented or the environment is ready to be reset
        if env.global_done:
            curr_info = self._reset_env(env)
            for brain_name, trainer in self.trainers.items():
                trainer.end_episode()

        # Decide and take an action
        take_action_vector, \
        take_action_memories, \
        take_action_text, \
        take_action_value, \
        take_action_outputs \
            = {}, {}, {}, {}, {}
        
        for brain_name, trainer in self.trainers.items():
            # Actionを決定する. 全arena分の配列になっている.
            (take_action_vector[brain_name],   # 発行するAction
             take_action_memories[brain_name], # None (use_recurrent時に利用)
             take_action_text[brain_name],     # 常にNone
             take_action_value[brain_name],    # 各Arenaに一つの値
             take_action_outputs[brain_name]) = trainer.take_action(curr_info)

        # 選んだActionによって環境を1 step進める
        new_info = env.step(vector_action=take_action_vector,
                            memory=take_action_memories,
                            text_action=take_action_text,
                            value=take_action_value)
        
        if ENABLE_VISITED_MAP_IMAGE:
            for brain_name in self.trainers.keys():
                lidar_estimator = self.lidar_estimators[brain_name]
                out = expand_brain_info(new_info[brain_name],
                                        self.extra_brain_infos[brain_name],
                                        lidar_estimator)
                new_info[brain_name], self.extra_brain_infos[brain_name] = out
                
        for brain_name, trainer in self.trainers.items():
            # ExperienceBufferに貯める
            trainer.add_experiences(curr_info, new_info, take_action_outputs[brain_name])
            trainer.process_experiences(curr_info, new_info)
            
            if trainer.is_ready_update() and self.train_model \
                    and trainer.get_step <= trainer.get_max_steps:
                # ExperienceBuffer に溜まった内容で Policy の学習をSGDで行う.
                trainer.update_policy()
                
            # Write training statistics to Tensorboard.
            trainer.write_summary(self.global_step)
            if self.train_model and trainer.get_step <= trainer.get_max_steps:
                trainer.increment_step_and_update_last_reward()
        return new_info

def expand_brain_info(brain_info, extra_brain_info, lidar_estimator):
    n_arenas = len(brain_info.rewards)
    if n_arenas != len(extra_brain_info.visited_maps):
        extra_brain_info.visited_maps = [ VisitedMap() for _ in range(n_arenas) ]

    # Estimate LIDAR target IDs and distances for all arenas at once.
    all_lidar_id_probs, all_lidar_distances = lidar_estimator.estimate(brain_info)
    # (n_arenas, 5, 13)   (n_arenas, 5)
    
    visited_map_images = []
    
    for reward, local_done, vector_observation, previous_vector_action, \
        visited_map, lidar_id_probs, lidar_distances in zip(
            brain_info.rewards,
            brain_info.local_done,
            brain_info.vector_observations,
            brain_info.previous_vector_actions,
            extra_brain_info.visited_maps,
            all_lidar_id_probs,
            all_lidar_distances):

        visited_map.add_visited_info(local_done,
                                     previous_vector_action,
                                     vector_observation,
                                     lidar_id_probs,
                                     lidar_distances)
        visited_map_image = visited_map.get_image()
        visited_map_images.append(visited_map_image)
        
    brain_info.visual_observations.append(np.array(visited_map_images, dtype=np.float))
    #self.logger.info('{}, {}, {}'.format(brain_info.max_reached[0],
    #                                     brain_info.local_done[0],
    #                                     brain_info.rewards[0]))
    #cv2.imshow('visited map', visited_map_images[0])
    #cv2.waitKey(1)
    return brain_info, extra_brain_info


def add_extra_camera_parameter(brain_info_parameter):
    modified = copy.copy(brain_info_parameter)
    modified.number_visual_observations += 1
    extra_camera_parameters = {
        'height': 84,
        'width': 84,
        'blackAndWhite': True
    }
    modified.camera_resolutions.append(extra_camera_parameters)
    return modified


class ExtraBrainInfo:
    def __init__(self):
        self.visited_maps = []        
