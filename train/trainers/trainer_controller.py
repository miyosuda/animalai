# -*- coding: utf-8 -*-
# # Unity ML-Agents Toolkit
# ## ML-Agent Learning
"""Launches trainers for each External Brains in a Unity Environment."""

import os
import logging
import shutil
import sys
from typing import *

import numpy as np
import tensorflow as tf

from animalai.envs import BrainInfo
from animalai.envs.exception import UnityEnvironmentException

from trainers.ppo.trainer import PPOTrainer


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
        :param model_path: 
            Path to save the model.
        :param summaries_dir: 
            Folder to save training summaries.
        :param run_id: 
            The sub-directory name for model and summary statistics
        :param save_freq: 
            Frequency at which to save model
        :param load: 
            Whether to load the model or randomly initialize.
        :param train: 
            Whether to train model, or only run inference.
        :param keep_checkpoints: 
            How many model checkpoints to keep.
        :param lesson: 
            Start learning from this lesson.
        :param external_brains: 
            dictionary of external brain names to BrainInfo objects.
        :param training_seed: 
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

    def _get_measure_vals(self):
        return None

    def _save_model(self, steps=0):
        """
        Saves current model to checkpoint folder.

        :param steps: 
            Current number of steps in training process.
        :param saver: 
            Tensorflow saver for session.
        """
        for brain_name in self.trainers.keys():
            self.trainers[brain_name].save_model()
        self.logger.info('Saved Model')

    def _save_model_when_interrupted(self, steps=0):
        self.logger.info('Learning was interrupted. Please wait '
                         'while the graph is generated.')
        self._save_model(steps)

    def _win_handler(self, event):
        """
        This function gets triggered after ctrl-c or ctrl-break is pressed
        under Windows platform.
        """
        if event in (win32con.CTRL_C_EVENT, win32con.CTRL_BREAK_EVENT):
            self._save_model_when_interrupted(self.global_step)
            self._export_graph()
            sys.exit()
            return True
        return False

    def _export_graph(self):
        """
        Exports latest saved models to .nn format for Unity embedding.
        """
        for brain_name in self.trainers.keys():
            self.trainers[brain_name].export_model()

    def initialize_trainers(self, trainer_config):
        """
        Initialization of the trainers

        :param trainer_config: 
            The configurations of the trainers
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
                self.trainers[brain_name] = PPOTrainer(
                    self.external_brains[brain_name],
                    0,
                    trainer_parameters_dict[brain_name],
                    self.train_model, self.load_model, self.seed, self.run_id)
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
            return env.reset(arenas_configurations=self.config)
            #self.update_config = False
        else:
            return env.reset()

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
        if self.train_model:
            for brain_name, trainer in self.trainers.items():
                trainer.write_tensorboard_text('Hyperparameters', trainer.parameters)
        try:
            while any([t.get_step <= t.get_max_steps \
                       for k, t in self.trainers.items()]) \
                  or not self.train_model:
                new_info = self.take_step(env, curr_info)
                self.global_step += 1
                if self.global_step % self.save_freq == 0 and self.global_step != 0 \
                        and self.train_model:
                    # Save Tensorflow model
                    self._save_model(steps=self.global_step)
                curr_info = new_info
            # Final save Tensorflow model
            if self.global_step != 0 and self.train_model:
                self._save_model(steps=self.global_step)
        except KeyboardInterrupt:
            if self.train_model:
                self._save_model_when_interrupted(steps=self.global_step)
            pass
        env.close()

        if self.train_model:
            self._export_graph()

    def take_step(self, env, curr_info):
        # If any lessons were incremented or the environment is
        # ready to be reset
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
            (take_action_vector[brain_name],
             take_action_memories[brain_name],
             take_action_text[brain_name],
             take_action_value[brain_name],
             take_action_outputs[brain_name]) = \
                trainer.take_action(curr_info)
            
        new_info = env.step(vector_action=take_action_vector,
                            memory=take_action_memories,
                            text_action=take_action_text,
                            value=take_action_value)
        
        for brain_name, trainer in self.trainers.items():
            trainer.add_experiences(curr_info, new_info, take_action_outputs[brain_name])
            trainer.process_experiences(curr_info, new_info)
            
            if trainer.is_ready_update() and self.train_model \
                    and trainer.get_step <= trainer.get_max_steps:
                # Perform gradient descent with experience buffer
                trainer.update_policy()
                
            # Write training statistics to Tensorboard.
            trainer.write_summary(self.global_step)
            if self.train_model and trainer.get_step <= trainer.get_max_steps:
                trainer.increment_step_and_update_last_reward()
        return new_info
