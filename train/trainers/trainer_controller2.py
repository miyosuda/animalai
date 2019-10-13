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
            self.extra_brain_infos = {}
            for brain_name in self.external_brain_names:
                self.external_brains[brain_name] = self._add_extra_camera_parameter(
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
            self.extra_brain_infos = {}
            for brain_name in self.trainers.keys():
                self.extra_brain_infos[brain_name] = ExtraBrainInfo() # delete old ExtraBrainInfo
                out = self._expand_brain_info(new_info[brain_name],
                                              self.extra_brain_infos[brain_name])
                new_info[brain_name], self.extra_brain_infos[brain_name] = out
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
                new_info[brain_name], self.extra_brain_infos[brain_name] = \
                self._expand_brain_info(new_info[brain_name], \
                self.extra_brain_infos[brain_name])
        
        for brain_name in self.trainers.keys():
            # Modify rewards according to the touched objects.
            self._modify_brain_info(new_info[brain_name])
        
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


    class TouchedObject(Enum):
        NONE = auto()
        GREEN = auto()
        YELLOW = auto()
        RED = auto()
        DEATHZONE = auto()
        HOTZONE = auto()
        UNKNOWN = auto()

    def _get_t(self, arena_id=0):
        config = self.config
        return config.arenas[arena_id].t

    def _detect_touched_object(self, reward, local_done):
        """
        Detect touched object.
        According to this page, https://www.mdcrosby.com/blog/animalaieval.html
        maximum episode steps (T) should be 250, 500 or 1000.
        The agent always get -1/T reward for each steps or zero if T=0.
        If the agent doesn't touch anything, he get -1/T reward for each step.
        If the agent is on HOTZONE, he get extra -10/T reward, so total reward should be -11/T for each step.
        If the agent touch DEATHZONE, he get extra -1 reward, so total reward should be -1 + -1/T. And epidode should terminate (local_done is True).
        If the agent touch RED, he get extra [-5, -0.5] reward, so total reward should be [-5, -0.5] + -1/T. And epidode should terminate (local_done is True).
        """
        eps = 1e-6 # for small fluctuation.
        t = self._get_t() # TODO: each arenas could have different t values.
        min_neg_reward = -1.0 / t - eps if t > 0 else -eps # -1/T
        
        if reward <= 0.0 and reward >= min_neg_reward: # [-1/T, 0]
            return self.TouchedObject.NONE
        elif reward <= 0.0 and reward >= min_neg_reward*11.0: # [-11/T, 0]
            return self.TouchedObject.HOTZONE
        elif reward <= 0.0 and not local_done:
            # TODO: HOTZONE reward should not be less than -11/T but sometimes it exceeds. Or it may be an another object.
            # Anyway, I treat this as HOTZONE temporarily.
            #self.logger.info('maybe HOTZONE {}, {}'.format(reward, local_done))
            #return self.TouchedObject.UNKNOWN
            return self.TouchedObject.HOTZONE
        elif reward <= 0.0 and local_done and (reward <= -1.0 and reward >= min_neg_reward-1.0): # [-1 + -1/T, -1]
            # It could be RED with -1 radius.
            return self.TouchedObject.DEATHZONE
        elif reward <= 0.0 and local_done:
            return self.TouchedObject.RED
        elif reward > 0.0 and not local_done:
            return self.TouchedObject.YELLOW
        elif reward > 0.0 and local_done:
            # It could be YELLOW if there's no GREEN in the arena and it's last YELLOW.
            return self.TouchedObject.GREEN
        else:
            return self.TouchedObject.UNKNOWN

    def _modify_brain_info(self, brain_info):
        """
        modify rewards according to this page.
        https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Best-Practices.md#rewards
        """
        assert len(brain_info.rewards) == len(brain_info.local_done)
        t = self._get_t() # TODO: each arenas could have different t values.
        min_pos_reward = 1.0 / t if t > 0 else 0.0 # 1/T
        modified_rewards = []
        
        for reward, local_done, vector_observation, previous_vector_action in zip(brain_info.rewards,
                                                                                  brain_info.local_done,
                                                                                  brain_info.vector_observations,
                                                                                  brain_info.previous_vector_actions):
            new_reward = reward
            touched = self._detect_touched_object(reward, local_done)
            #if touched is not self.TouchedObject.NONE:
            #    self.logger.info('touched: {}, {}, {}'.format(touched, local_done, reward))
            assert touched is not self.TouchedObject.UNKNOWN
            # TODO: tweak amount
            if touched is self.TouchedObject.GREEN:
                new_reward = 1.0
            elif touched is self.TouchedObject.YELLOW:
                new_reward = 2.0
            elif touched is self.TouchedObject.RED:
                new_reward = -0.5
            elif touched is self.TouchedObject.DEATHZONE:
                new_reward = -0.5
            elif touched is self.TouchedObject.HOTZONE:
                new_reward = reward * 0.5
            else:
                new_reward = reward # no change
            # Add small positive extra reward for forward velocity.
            if vector_observation[2] > 0.0:
                new_reward += min_pos_reward
            #if vector_observation[2] == 0.0 and previous_vector_action[0] == 1:
            #    self.logger.info('{}, {}'.format(vector_observation, previous_vector_action))
            modified_rewards.append(new_reward)
        assert len(modified_rewards) == len(brain_info.rewards)
        #if brain_info.rewards != modified_rewards:
        #    self.logger.info('original rewards: {}, modified rewards: {}'.format(brain_info.rewards, modified_rewards))
        brain_info.rewards = modified_rewards

    def _add_extra_camera_parameter(self, brain_info_parameter):
        modified = copy.copy(brain_info_parameter)
        modified.number_visual_observations += 1
        extra_camera_parameters = {
            'height': 84,
            'width': 84,
            'blackAndWhite': True
        }
        modified.camera_resolutions.append(extra_camera_parameters)
        return modified

    VELOCITY_CONSTANT = 0.0595

    def _expand_brain_info(self, brain_info, extra_brain_info):
        n_arenas = len(brain_info.rewards)
        if n_arenas != len(extra_brain_info.local_angles):
            extra_brain_info.local_angles = [ 0 for _ in range(n_arenas) ]
            extra_brain_info.local_positions = [ np.zeros((3), dtype=np.float) for _ in range(n_arenas) ]
            extra_brain_info.local_histories = [ [] for _ in range(n_arenas) ]
        new_local_angles = []
        new_local_positions = []
        new_local_histories = []
        visited_map_images = []
        for reward, local_done, vector_observation, previous_vector_action, local_angle, local_position, local_history in zip(brain_info.rewards,
                                                                                                                              brain_info.local_done,
                                                                                                                              brain_info.vector_observations,
                                                                                                                              brain_info.previous_vector_actions,
                                                                                                                              extra_brain_info.local_angles,
                                                                                                                              extra_brain_info.local_positions,
                                                                                                                              extra_brain_info.local_histories):
            if local_done:
                new_local_angle = 0
                new_local_position = np.zeros((3), dtype=np.float)
                new_local_history = [new_local_position]
            else:
                if previous_vector_action[1] == 1: # turn right
                    new_local_angle = (local_angle + 6) % 360
                elif previous_vector_action[1] == 2: # turn left
                    new_local_angle = (local_angle - 6) % 360
                else:
                    new_local_angle = local_angle % 360
                rot = self._rotate_array(vector_observation, new_local_angle)
                new_local_position = local_position + self.VELOCITY_CONSTANT * np.array(rot, dtype=np.float)
                new_local_history = local_history + [new_local_position] # TODO
            new_local_angles.append(new_local_angle)
            new_local_positions.append(new_local_position)
            new_local_histories.append(new_local_history)
            visited_map_image = self._generate_visited_map_image(new_local_angle,
                                                                 new_local_position,
                                                                 new_local_history)
            visited_map_images.append(visited_map_image)
        extra_brain_info.local_angles = new_local_angles
        extra_brain_info.local_positions = new_local_positions
        extra_brain_info.local_histories = new_local_histories
        brain_info.visual_observations.append(np.array(visited_map_images, dtype=np.float))

        if DEBUG_SHOW_VISITED_MAP:
            self.logger.info('{}, {}, {}, {}, {}, {}'.format(
                brain_info.max_reached[0],
                brain_info.local_done[0],
                brain_info.rewards[0],
                extra_brain_info.local_angles[0],
                extra_brain_info.local_positions[0],
                len(extra_brain_info.local_histories[0])))
            cv2.imshow('visited map', visited_map_images[0])
            cv2.waitKey(1)
        
        return brain_info, extra_brain_info

    def _generate_visited_map_image(self, local_angle, local_position, local_history):
        local_position = np.array(local_position, dtype=np.float)
        local_history = np.array(local_history, dtype=np.float)
        #shifted_local_history = local_history # fixed position for debug
        shifted_local_history = local_history - local_position
        rotated_local_history = []
        for pos in shifted_local_history: # TODO: use numpy
            rot = self._rotate_array(pos, -local_angle) # minus!
            #rotated_local_history.append(pos) # no rotation for debug
            rotated_local_history.append(rot)
        visited_map_image = np.zeros((84, 84, 1), dtype=np.float)
        min_pos = -40.0 * np.sqrt(2.0)
        max_pos =  40.0 * np.sqrt(2.0)
        for pos in rotated_local_history:
            x, y, z = pos
            sx = int(84.0 * (x - min_pos) / (max_pos - min_pos))
            sy = int(84.0 * (-z - min_pos) / (max_pos - min_pos)) # minus z!
            assert 0 <= sx and sx < 84 and 0 <= sy and sy < 84
            #self.logger.info('{}, {}, {}, {}'.format(x, z, sx, sy))
            # TODO: confirm sy,sx or sx,sy
            visited_map_image[sy,sx] = [1.0] # TODO: frequency?
        return visited_map_image

    def _rotate_array(self, pos, angle):
        sin_angle = np.sin(2.0 * np.pi * float(angle) / 360.0)
        cos_angle = np.cos(2.0 * np.pi * float(angle) / 360.0)
        dx = cos_angle * pos[0] + sin_angle * pos[2]
        dy = pos[1]
        dz = -sin_angle * pos[0] + cos_angle * pos[2]
        return [dx, dy, dz]

class ExtraBrainInfo:
    def __init__(self):
        self.local_angles = []
        self.local_positions = []
        self.local_histories = []
