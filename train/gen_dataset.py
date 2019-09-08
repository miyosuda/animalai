# -*- coding: utf-8 -*-
import argparse
import cv2
import os
import time
import numpy as np
from distutils.util import strtobool

import yaml

from animalai.envs.gym.environment import AnimalAIEnv
from animalai.envs.arena_config import ArenaConfig
from animalai.envs.brain import BrainParameters

from trainers.ppo.policy import PPOPolicy


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


def save_state(state, dir_path, i):
    file_name = "{}/image{}.png".format(dir_path, i)
    state = (state*255.0).astype(np.uint8)
    image = cv2.cvtColor(state, cv2.COLOR_RGB2BGR)
    cv2.imwrite(file_name, image)    


def collect(env, agent, step_size):
    obs = None

    file_size_in_dir = 1000

    if not os.path.exists("base_data"):
        os.mkdir("base_data")

    start_time = time.time()        

    for i in range(step_size):
        if i % file_size_in_dir == 0:
            dir_path = "base_data/dir{}".format(i // file_size_in_dir)
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)

        if obs == None:
            obs, reward, done, info = env.step([0, 0])

        actions = []
        positions = []
        angles = []
        rewards = []
        dones = []
        
        action, _, _, _, pos_angle = agent.step(obs, reward, done, info)
        obs, reward, done, info = env.step(action)

        state = obs[0] # dtype=float64
        save_state(state, dir_path, i)

        pos   = pos_angle[0]
        angle = pos_angle[1] / 360.0 * (2.0 * np.pi)

        actions.append(action)
        positions.append(pos)
        angles.append(angle)
        rewards.append(reward)
        dones.append(done)
        
        if done:
            obs = None

        if i % 1000 == 0:
            print("step{}".format(i))
            elapsed_time = time.time() - start_time
            print("fps={}".format(i / elapsed_time))

    np.savez_compressed("base_data/infos",
                        actions=actions,
                        positions=positions,
                        angles=angles,
                        rewards=rewards,
                        dones=dones)
    
    print("collecting finished")


def generate(base_data_dir, frame_size):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--step_size",
                        help="Training step size",
                        type=int,
                        default=100)
    args = parser.parse_args()

    step_size = args.step_size
    
    model_path          = './models/run_005/Learner'
    arena_config_path   = './configs/3-Obstacles.yaml'
    
    trainer_config_path = './configs/trainer_config.yaml'
    env_path            = '../env/AnimalAICustom'

    agent = init_agent(trainer_config_path, model_path)
    env = init_env(env_path, args.seed)

    arena_config_in = ArenaConfig(arena_config_path)
    agent.reset(t=arena_config_in.arenas[0].t)

    env.reset(arenas_configurations=arena_config_in)

    collect(env, agent, step_size)
    
    generate("base_data", step_size)


if __name__ == '__main__':
    main()
