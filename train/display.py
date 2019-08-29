# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os
from collections import deque
import pygame, sys
from pygame.locals import *
import argparse
from distutils.util import strtobool

import yaml

from animalai.envs.gym.environment import AnimalAIEnv
from animalai.envs.arena_config import ArenaConfig
from animalai.envs.brain import BrainParameters

from trainers.ppo.policy import PPOPolicy


BLUE  = (128, 128, 255)
RED   = (255, 192, 192)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY  = (128, 128, 128)


class MovieWriter(object):
    def __init__(self, file_name, frame_size, fps):
        """
        frame_size is (w, h)
        """
        self._frame_size = frame_size
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.vout = cv2.VideoWriter()
        success = self.vout.open(file_name, fourcc, fps, frame_size, True)
        if not success:
            print("Create movie failed: {0}".format(file_name))

    def add_frame(self, frame):
        """
        frame shape is (h, w, 3), dtype is np.uint8
        """
        self.vout.write(frame)

    def close(self):
        self.vout.release()
        self.vout = None



class ValueHistory(object):
    def __init__(self):
        self._values = deque(maxlen=100)

    def add_value(self, value):
        self._values.append(value)

    @property
    def is_empty(self):
        return len(self._values) == 0

    @property
    def values(self):
        return self._values


class Display(object):
    def __init__(self, display_size, agent, env):
        pygame.init()
        self.surface = pygame.display.set_mode(display_size, 0, 24)
        pygame.display.set_caption('AnimalAI')
        self.action_size = 4

        self.agent = agent
        self.env = env

        self.font = pygame.font.SysFont(None, 20)
        
        self.value_history = ValueHistory()
        self.episode_reward = 0

        # 初回の初期化
        self.obs = None

    def update(self):
        self.surface.fill(BLACK)
        self.process()
        pygame.display.update()

    def choose_action(self, pi_values):
        return np.random.choice(range(len(pi_values)), p=pi_values)

    def scale_image(self, image, scale):
        return image.repeat(scale, axis=0).repeat(scale, axis=1)

    def draw_text(self, str, left, top, color=WHITE):
        text = self.font.render(str, True, color, BLACK)
        text_rect = text.get_rect()
        text_rect.left = left
        text_rect.top = top
        self.surface.blit(text, text_rect)

    def draw_center_text(self, str, center_x, top):
        text = self.font.render(str, True, WHITE, BLACK)
        text_rect = text.get_rect()
        text_rect.centerx = center_x
        text_rect.top = top
        self.surface.blit(text, text_rect)

    def show_policy(self, pi, x=10, y=150, label="PI"):
        """
        Show action probability.
        """
        start_x = x
        for i in range(len(pi)):
            width = pi[i] * 100
            pygame.draw.rect(self.surface, WHITE, (start_x, y, width, 10))
            y += 20
        self.draw_center_text(label, x+40, y)

    def show_image(self, state):
        """
        Show input image
        """
        state_ = state * 255.0
        data = state_.astype(np.uint8)
        image = pygame.image.frombuffer(data, (84, 84), 'RGB')
        self.surface.blit(image, (8, 8))
        self.draw_center_text("input", 50, 100)

    def show_value(self):
        if self.value_history.is_empty:
            return

        min_v = float("inf")
        max_v = float("-inf")

        values = self.value_history.values

        for v in values:
            min_v = min(min_v, v)
            max_v = max(max_v, v)

        top = 150
        left = 150
        width = 100
        height = 100
        bottom = top + width
        right = left + height

        d = max_v - min_v
        last_r = 0.0
        for i, v in enumerate(values):
            r = (v - min_v) / d
            if i > 0:
                x0 = i - 1 + left
                x1 = i + left
                y0 = bottom - last_r * height
                y1 = bottom - r * height
                pygame.draw.line(self.surface, BLUE, (x0, y0), (x1, y1), 1)
            last_r = r

        pygame.draw.line(self.surface, WHITE, (left, top), (left, bottom), 1)
        pygame.draw.line(self.surface, WHITE, (right, top), (right, bottom), 1)
        pygame.draw.line(self.surface, WHITE, (left, top), (right, top), 1)
        pygame.draw.line(self.surface, WHITE, (left, bottom), (right, bottom), 1)

        self.draw_center_text("V", left + width / 2, bottom + 10)

    def show_agent_pos_angle(self, pos_angle):
        """ Draw agent position and angle (for custom environment) """
        top = 8
        left = 150
        width = 100
        height = 100
        bottom = top + width
        right = left + height

        pos = pos_angle[0]
        angle = pos_angle[1] / 360.0 * (2.0 * np.pi)
        
        target_pos_0 = pos[0] + np.sin(angle) * 5
        target_pos_1 = pos[2] + np.cos(angle) * 5
        target_pos = (target_pos_0, target_pos_1)
        
        relateive_pos        = (pos[0]        / 40.0, pos[2]        / 40.0)
        relateive_target_pos = (target_pos[0] / 40.0, target_pos[1] / 40.0)
        screen_pos        = (int(relateive_pos[0]              * width + left),
                             int((1.0-relateive_pos[1])        * height + top))
        target_screen_pos = (int(relateive_target_pos[0]       * width + left),
                             int((1.0-relateive_target_pos[1]) * height + top))

        # draw position
        pygame.draw.circle(self.surface, RED, screen_pos, 5)
        # draw angle target
        pygame.draw.line(self.surface, WHITE,
                         screen_pos, target_screen_pos, 1)

        # draw frame
        pygame.draw.line(self.surface, GRAY, (left, top), (left, bottom), 1)
        pygame.draw.line(self.surface, GRAY, (right, top), (right, bottom), 1)
        pygame.draw.line(self.surface, GRAY, (left, top), (right, top), 1)
        pygame.draw.line(self.surface, GRAY, (left, bottom), (right, bottom), 1)

    def show_reward(self):
        self.draw_text("REWARD: {}".format(self.episode_reward), 310, 10)

    def get_frame(self):
        data = self.surface.get_buffer().raw
        return data

    def softmax(self, a):
        c = np.max(a)
        exp_a = np.exp(a - c)
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a
        return y

    def process(self):
        if self.obs == None:
            self.obs, self.reward, self.done, self.info = self.env.step([0, 0])
            
        action, log_probs, value, entropy, pos_angle = self.agent.step(self.obs,
                                                                       self.reward,
                                                                       self.done,
                                                                       self.info)
        
        self.obs, self.reward, self.done, self.info = self.env.step(action)
        state = self.obs[0] # float64

        self.episode_reward += self.reward
        self.value_history.add_value(value)

        pi0 = self.softmax(log_probs[0,0:3]) # 前後方向のAction
        pi1 = self.softmax(log_probs[0,3:])  # 左右方向のAction

        if self.done:
            self.obs = None
        
        self.show_image(state)
        self.show_policy(pi0, 10, 150, "PI (F<->B)")
        self.show_policy(pi1, 10, 250, "PI (R<->L)")
        self.show_value()
        self.show_reward()

        if pos_angle is not None:
            self.show_agent_pos_angle(pos_angle)


class Agent(object):
    def __init__(self,
                 trainer_config_path,
                 model_path):

        self.brain = BrainParameters(
            brain_name = 'Learner',
            camera_resolutions = [{
                'height': 84,
                'width' : 84,
                'blackAndWhite': False
            }],
            num_stacked_vector_observations = 1,
            vector_action_descriptions    = ['', ''],
            vector_action_space_size      = [3, 3],
            vector_action_space_type      = 0,  # corresponds to discrete
            vector_observation_space_size = 3
        )
        
        self.trainer_params = yaml.load(open(trainer_config_path))['Learner']
        self.trainer_params['keep_checkpoints'] = 0
        self.trainer_params['model_path']       = model_path
        self.trainer_params['use_recurrent']    = False

        self.policy = PPOPolicy(brain=self.brain,
                                seed=0,
                                trainer_params=self.trainer_params,
                                is_training=False,
                                load=True)

    def reset(self, t=250):
        pass

    def fix_brain_info(self, brain_info):
        if brain_info.vector_observations.shape[1] > 3:
            # カスタム環境用にvector_observationsをいじったものだった場合
            extended_infos = brain_info.vector_observations[:,3:]
            # 元の3次元だけのvector_observationsに戻す
            brain_info.vector_observations = brain_info.vector_observations[:,:3]
            agent_pos   = extended_infos[:,:3]
            agent_angle = extended_infos[:,3]
            return (agent_pos[0], agent_angle[0])
        else:
            return None

    def step(self, obs, reward, done, info):
        brain_info = info['brain_info']
        pos_angle = self.fix_brain_info(brain_info) # Custom環境でのみの情報
        
        out = self.policy.evaluate(brain_info=brain_info)
        action    = out['action']
        log_probs = out['log_probs']
        value     = out['value']
        entropy   = out['entropy']
        return action, log_probs, value, entropy, pos_angle


def init_agent(trainer_config_path, model_path):
    agent = Agent(trainer_config_path, model_path)
    return agent


def init_env(env_path, env_seed):
    env = AnimalAIEnv(
        environment_filename=env_path,
        seed=env_seed,
        retro=False,
        n_arenas=1,
        worker_id=1,
        docker_training=False,
        resolution=84
    )
    return env
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--recording", type=strtobool, default="false")
    parser.add_argument("--custom_env", type=strtobool, default="false")
    args = parser.parse_args()
    
    model_path          = './models/run_001/Learner'    
    arena_config_path   = './configs/3-Obstacles.yaml'

    if args.custom_env:
        # Using custom environment for pos/angle visualization
        env_path = '../env/AnimalAICustom'
    else:
        env_path = '../env/AnimalAI'
    
    trainer_config_path = './configs/trainer_config.yaml'
    
    recording = args.recording
    display_size = (440, 400)

    agent = init_agent(trainer_config_path, model_path)
    env = init_env(env_path, args.seed)

    display = Display(display_size, agent, env)
    clock = pygame.time.Clock()

    running = True
    FPS = 15

    if recording:
        writer = MovieWriter("out.mov", display_size, FPS)

    arena_config_in = ArenaConfig(arena_config_path)
    agent.reset(t=arena_config_in.arenas[0].t)

    env.reset(arenas_configurations=arena_config_in)
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        display.update()
        clock.tick(FPS)

        if recording:
            frame_str = display.get_frame()
            d = np.fromstring(frame_str, dtype=np.uint8)
            d = d.reshape((display_size[1], display_size[0], 3))
            writer.add_frame(d)

    if recording:
        writer.close()


if __name__ == '__main__':
    main()
