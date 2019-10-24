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
from trainers.visited_map import VisitedMap


from lidar.utils import convert_target_ids, get_target_names

USE_FIXED_VISITED_MAP_COORDINATE = False
USE_LIDAR_VECTOR_INFO = False


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
    def __init__(self,
                 display_size,
                 agent,
                 env,
                 allo_estimator,
                 ego_integrator):
        pygame.init()
        self.surface = pygame.display.set_mode(display_size, 0, 24)
        pygame.display.set_caption('AnimalAI')
        self.action_size = 4

        self.agent = agent
        self.env = env
        self.allo_estimator = allo_estimator
        self.ego_integrator = ego_integrator

        self.font = pygame.font.SysFont(None, 20)
        
        self.value_history = ValueHistory()
        self.episode_reward = 0
        self.last_episode_reward = 0
        self.total_episode_reward = 0
        self.num_episode = 0

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

    def draw_right_text(self, str, right, top):
        text = self.font.render(str, True, WHITE, BLACK)
        text_rect = text.get_rect()
        text_rect.right = right
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
        image8 = pygame.transform.scale(image, (512, 512))
        self.surface.blit(image8, (900-512-8, 8))

    def show_visited_map(self,
                         visited_map,
                         local_map_image,
                         local_position,
                         local_angle):
        """
        Show visited map
        """
        top = 8
        left = 8+100
        width = 84
        height = 84
        bottom = top + width
        right = left + height
        
        data = (visited_map * 127.0).astype(np.uint8)
        if data.shape[2] != 3:
            data = data.repeat(3, axis=2)
        image = pygame.image.frombuffer(data, (84, 84), 'RGB')
        self.surface.blit(image, (left, top))
        self.draw_center_text("map", left+42, top+92)

        # local mapを何倍して表示するか
        local_map_scale = 2
        local_map_org_w = local_map_image.shape[0]
        local_map_org_h = local_map_image.shape[1]
        
        local_map_w = local_map_org_w * local_map_scale
        local_map_h = local_map_org_h * local_map_scale
        
        local_left = left+84+4
        local_right = local_left + local_map_w
        local_top = top
        local_bottom = top + local_map_h

        local_data = (local_map_image * 127.0).astype(np.uint8)
        local_data = local_data[:,:,np.newaxis]
        local_data = local_data.repeat(3, axis=2)
        image = pygame.image.frombuffer(local_data, (local_map_org_h, local_map_org_w), 'RGB')
        image = pygame.transform.scale(image, (local_map_h, local_map_w))
        
        self.surface.blit(image, (local_left, local_top))

        pygame.draw.line(self.surface, GRAY, (left, top), (left, bottom), 1)
        pygame.draw.line(self.surface, GRAY, (right, top), (right, bottom), 1)
        pygame.draw.line(self.surface, GRAY, (left, top), (right, top), 1)
        pygame.draw.line(self.surface, GRAY, (left, bottom), (right, bottom), 1)

        pygame.draw.line(self.surface, GRAY, 
                         (local_left, local_top), (local_left, local_bottom), 1)
        pygame.draw.line(self.surface, GRAY, 
                         (local_right, local_top), (local_right, local_bottom), 1)
        pygame.draw.line(self.surface, GRAY,
                         (local_left, local_top), (local_right, local_top), 1)
        pygame.draw.line(self.surface, GRAY,
                         (local_left, local_bottom), (local_right, local_bottom), 1)
        
        # local_pos, angleの表示
        converted_local_pos = ((local_position / VisitedMap.RANGE_MAX) + 1.0) * 0.5 * local_map_w
        converted_local_pos_x = int(converted_local_pos[0])
        converted_local_pos_z = int(converted_local_pos[2])

        center = (local_left + converted_local_pos_x,
                  local_top + local_map_h - converted_local_pos_z)

        vx = np.sin(local_angle / 360.0 * np.pi * 2.0) * 8
        vz = np.cos(local_angle / 360.0 * np.pi * 2.0) * 8
        pygame.draw.line(self.surface, RED, center, (center[0]+vx, center[1]-vz), 2)
        

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
            r = (v - min_v) / d if abs(d) > 1e-6 else 1.0
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

    def show_agent_pos_angle(self, pos_angle, top=300, left=150):
        """ Draw agent position and angle (for custom environment) """
        width = 100
        height = 100
        bottom = top + width
        right = left + height

        pos   = pos_angle[0]
        angle = pos_angle[1]
        
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

    def show_target(self, target_ids, target_distances, top=300, left=150):
        """ show lidar target info """
        target_names = get_target_names(target_ids)
        for i, target_name in enumerate(target_names):
            self.draw_text(target_name, left, top + i * 20)
            target_distance = target_distances[i]
            y = top + 15 + 20 * i
            distance_scale = 4.0
            right = left + target_distance * distance_scale
            pygame.draw.line(self.surface, WHITE, (left, y), (right, y), 1)

    def show_reward(self):
        self.draw_right_text("Cur Rwd: ", 900-512-8-8-50, 8)
        self.draw_right_text("{:.5f}".format(self.episode_reward), 900-512-8-8, 8)
        self.draw_right_text("Lst Rwd: ", 900-512-8-8-50, 8+16)
        self.draw_right_text("{:.5f}".format(self.last_episode_reward), 900-512-8-8, 8+16)
        if self.num_episode > 0:
            self.draw_right_text("Avg Rwd: ", 900-512-8-8-50, 8+16+16)
            self.draw_right_text("{:.5f}".format(
                self.total_episode_reward / self.num_episode), 900-512-8-8, 8+16+16)
        self.draw_right_text("Ep Num: ", 900-512-8-8-50, 8+16+16+16)
        self.draw_right_text("{}".format(self.num_episode), 900-512-8-8, 8+16+16+16)

    def show_velocity(self, velocity):
        top = 150 + 10
        left = 250 + 10 + 15
        width = 100 - 20
        height = 100 - 20
        center = (left + width//2, top + height//2)
        pygame.draw.circle(self.surface, GRAY, center, width//2, 1)
        vx = int(velocity[0] / 40.0 * width)
        vz = int(velocity[2] / 40.0 * height)
        pygame.draw.line(self.surface, WHITE, center, (center[0]-vx, center[1]-vz), 2)

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
        # 今回resetが起こったかどうか(デバッグ用なので最終的には消す)
        debug_has_reset = False
        
        if self.obs == None:
            # 初回のstate生成
            self.agent.reset()
            self.last_action = np.array([[0,0]], dtype=np.int32)
            self.obs, self.reward, self.done, self.info = self.env.step(self.last_action)
            if self.allo_estimator is not None:
                self.allo_estimator.reset()
            if self.ego_integrator is not None:
                self.ego_integrator.reset()
            debug_has_reset = True

        last_state = self.obs[0] # dtype=float64

        # obsの状態に対して発行するactionをpolicyが決定.
        # last_actionを取った結果がobs.
        # velocityはobs(=last_state)になる前のlocal velocity.
        # pos_angleはobs(=last_staet)の状態の絶対座標と絶対角度
        out = self.agent.step(self.obs,
                              self.reward,
                              self.done,
                              self.info)
        action, log_probs, value, entropy, velocity, pos_angle, target_ids_distances = out
        
        if debug_has_reset and pos_angle is not None and self.ego_integrator is not None:
            # 今回リセットが起こったので、リセット時の絶対位置角度をデバッグ用に記録
            self.ego_integrator.debug_set_reset_pos_angle(pos_angle[0], pos_angle[1])
            
        if self.ego_integrator is not None:
            if pos_angle is not None:
                self.ego_integrator.debug_confirm(velocity, pos_angle[0], pos_angle[1])
            self.ego_integrator.integrate(self.last_action, velocity)
            # ここでの相対角度はobs(last_state), およびpos_angleの内容に相当.
            if pos_angle is not None:
                debug_integated_pos_angle = self.ego_integrator.debug_integrated_absolute_pos_angle
                self.show_agent_pos_angle(debug_integated_pos_angle,
                                          top=410, left=250+20)

        if self.allo_estimator is not None:
            estimated_pos_angle = self.allo_estimator.estimate(last_state,
                                                               self.last_action,
                                                               velocity)
            self.show_agent_pos_angle(estimated_pos_angle, top=410, left=150)
            
        if self.agent.extra_brain_info is not None:
            estimated_target_id_probs = self.agent.extra_brain_info.debug_lidar_id_probs[0]
            estimated_target_distances = self.agent.extra_brain_info.debug_lidar_distances[0]
            # ArgMaxで最大の確率をとるtarget idを取ってきている
            estimated_target_ids = np.argmax(estimated_target_id_probs, axis=1)
            self.show_target(estimated_target_ids, estimated_target_distances,
                             top=300, left=150)
            
            visited_map = self.agent.extra_brain_info.visited_maps[0]
            map_image = visited_map.get_image()
            local_map_image = visited_map.get_local_map_image()
            self.show_visited_map(map_image, local_map_image,
                                  visited_map.last_local_position,
                                  visited_map.last_local_angle)
            
        # 環境に対してActionを発行して結果を得る
        self.obs, self.reward, self.done, self.info = self.env.step(action)
        # obs:[2], obs[0]:(84,84,3)   obs[1]:(3,)　デバッグ時は(7,)
        # reward: float
        # done: bool

        state = self.obs[0] # float64
        
        self.episode_reward += self.reward
        self.value_history.add_value(value)
        
        pi0 = self.softmax(log_probs[0,0:3]) # 前後方向のAction
        pi1 = self.softmax(log_probs[0,3:])  # 左右方向のAction

        if self.done:
            self.last_episode_reward = self.episode_reward
            self.total_episode_reward += self.episode_reward
            self.num_episode += 1
            self.obs = None
            self.episode_reward = 0.0
        else:
            self.last_action = action
        
        self.show_image(state)
        
        self.show_policy(pi0, 10, 120, "PI (F<->B)")
        self.show_policy(pi1, 10, 220, "PI (R<->L)")
        
        self.show_value()
        self.show_reward()
        self.show_velocity(velocity)

        if pos_angle is not None:
            self.show_agent_pos_angle(pos_angle, top=410, left=30)

        if target_ids_distances is not None:
            self.show_target(target_ids_distances[0], target_ids_distances[1],
                             top=300, left=10)


from trainers.trainer_controller2 import add_extra_camera_parameter, expand_brain_info, ExtraBrainInfo
from lidar.estimator import MultiLidarEstimator
            

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

        self.brain = add_extra_camera_parameter(
            self.brain,
            USE_FIXED_VISITED_MAP_COORDINATE)
        self.extra_brain_info = ExtraBrainInfo()
        
        self.trainer_params = yaml.load(open(trainer_config_path))['Learner']
        self.trainer_params['keep_checkpoints'] = 0
        self.trainer_params['model_path']       = model_path

        self.policy = PPOPolicy(brain=self.brain,
                                seed=0,
                                trainer_params=self.trainer_params,
                                is_training=False,
                                load=True)
        
        self.lidar_estimator = MultiLidarEstimator(
            save_dir="saved_lidar", # データパスの指定
            n_arenas=1
        )

    def reset(self, t=250):
        self.extra_brain_info = ExtraBrainInfo()
        self.lidar_estimator.reset()

    def fix_brain_info(self, brain_info):
        velocity = brain_info.vector_observations[:,:3]
        
        if brain_info.vector_observations.shape[1] > 3:
            # カスタム環境用にvector_observationsをいじったものだった場合
            extended_infos = brain_info.vector_observations[:,3:]
            
            # 元の3次元だけのvector_observationsに戻す
            brain_info.vector_observations = brain_info.vector_observations[:,:3]
            agent_pos   = extended_infos[:,:3]
            agent_angle = extended_infos[:,3] / 360.0 * (2.0 * np.pi) # 0~2pi
            
            raw_target_ids   = extended_infos[:,4:4+5].astype(np.int32)
            target_distances = extended_infos[:,9:9+5]

            # 共通するオブジェクトのIDをまとめる
            target_ids = convert_target_ids(raw_target_ids[0])

            return velocity[0], (agent_pos[0], agent_angle[0]), \
                (target_ids, target_distances[0])
        else:
            return velocity[0], None, None

    def step(self, obs, reward, done, info):
        brain_info = info['brain_info']
        
        out = self.fix_brain_info(brain_info) # Custom環境でのみの情報
        velocity, pos_angle, target_ids_distances = out
        # (3,)        

        brain_info, self.extra_brain_info = expand_brain_info(
            brain_info,
            self.extra_brain_info,
            self.lidar_estimator,
            USE_LIDAR_VECTOR_INFO)
        
        out = self.policy.evaluate(brain_info=brain_info)
        action    = out['action']
        log_probs = out['log_probs']
        value     = out['value']
        entropy   = out['entropy']

        return action, log_probs, value, entropy, velocity, pos_angle, target_ids_distances


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
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--recording", type=strtobool, default="false")
    parser.add_argument("--custom", type=strtobool, default="true")
    parser.add_argument("--allo", type=strtobool, default="false")
    parser.add_argument("--lidar", type=strtobool, default="true")
    
    args = parser.parse_args()
    
    model_path = './models/run_1000/Learner'
    
    #arena_config_path   = './configs/obstacle-curriculum/obstacle-w-t-wt-tt-r1.yaml'
    #arena_config_path   = './configs/lidar/obstacle-w-t-wt-tt-cb-ulo-rm.yaml'
    #arena_config_path   = './configs/lidar/obstacle-w-t-wt-tt-cb-ulo-rm.yaml'
    arena_config_path   = './configs/1-Food-arena16-t250-r5.yaml'

    if args.custom:
        # Using custom environment for pos/angle visualization
        env_path = '../env/AnimalAIScan'
    else:
        env_path = '../env/AnimalAI'
    
    trainer_config_path = './configs/trainer_config_rec.yaml'
    
    recording = args.recording
    display_size = (900, 528)

    agent = init_agent(trainer_config_path, model_path)
    env = init_env(env_path, args.seed)

    if args.allo:
        from allocentric.estimator import AllocentricEstimator
        from allocentric.integrator import EgocentricIntegrator
        allo_model_dir = "saved_allocentric"
        allo_estimator = AllocentricEstimator(allo_model_dir)
        ego_integrator = EgocentricIntegrator()
    else:
        allo_estimator = None
        ego_integrator = None
        
    display = Display(display_size, agent, env, allo_estimator, ego_integrator)
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
