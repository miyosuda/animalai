# -*- coding: utf-8 -*-
# local_done時の挙動確認
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

DEBUG_DATA_DIR = "debug_data"


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
            velocity = brain_info.vector_observations[:,:3]
            
            extended_infos = brain_info.vector_observations[:,3:]
            # 元の3次元だけのvector_observationsに戻す
            brain_info.vector_observations = brain_info.vector_observations[:,:3]
            agent_pos   = extended_infos[:,:3]
            agent_angle = extended_infos[:,3]
            return (velocity[0], agent_pos[0], agent_angle[0])
        else:
            print("There was no extended brain info")
            return None

    def step(self, obs, reward, done, info):
        brain_info = info['brain_info']
        velocity_pos_angle = self.fix_brain_info(brain_info) # Custom環境でのみの情報
        
        out = self.policy.evaluate(brain_info=brain_info)
        action    = out['action']
        log_probs = out['log_probs']
        value     = out['value']
        entropy   = out['entropy']
        return action, log_probs, value, entropy, velocity_pos_angle

    
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


def load_state(index):
    dir_path = "{}/dir{}".format(DEBUG_DATA_DIR, index // FILE_SIZE_IN_DIR)
    file_name = "{}/image{}.png".format(dir_path, index)
    image = cv2.imread(file_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def collect(env, agent, step_size):
    if not os.path.exists(DEBUG_DATA_DIR):
        os.mkdir(DEBUG_DATA_DIR)

    velocities = []
    actions = []
    positions = []
    angles = []
    rewards = []
    dones = []

    # 初回のstate生成
    last_action = np.array([0,0], dtype=np.int32)
    obs, reward, done, info = env.step(last_action)

    next_break = False

    for i in range(step_size):
        # 実際のstateやangle, positionが入っているのはinfo
        last_state = obs[0] # dtype=float64

        # obsの状態に対してpolicyがactionを決定
        action, _, _, _, last_velocity_pos_angle = agent.step(obs, reward, done, info)
        # action=(1,2)
        # last_velocity_pos_angleはaction発行前のvelocity, pos, angle
        # actionは、last_stateにおいて発行したAction

        # Envに対してactionを発行し発行して結果を得る
        obs, reward, done, info = env.step(action)

        new_state = obs[0] # dtype=float64

        # 次stateを保存して、done時の次stateがどうなっているかを確認
        save_state(new_state, DEBUG_DATA_DIR, i)

        last_velocity = last_velocity_pos_angle[0]
        last_pos      = last_velocity_pos_angle[1]
        last_angle    = last_velocity_pos_angle[2] / 360.0 * (2.0 * np.pi)
        
        actions.append(np.squeeze(last_action))
        
        # last_stateの状態になる前に発行したAction (inputになる)
        # (このactionを発行してlast_stateに遷移した)
        # (1,2) -> (2,)
        velocities.append(last_velocity) # Action発酵前のlocal velocity (inputになる)
        positions.append(last_pos)       # Action発行前のpos (output targetになる)
        angles.append(last_angle)        # Action発行前のangle (output targetになる)
        rewards.append(reward)           # Actionを発行して得たreward
        dones.append(done)               # Actionを発行してterminateしたかどうか

        last_action = action

        print("step{}, done={}, action={}, last_velocity={}".format(i, done, action, last_velocity))

        if next_break:
            break

        if done:
            # 次のループ時にbreak
            next_break = True
        
    np.savez_compressed("{}/base_infos".format(DEBUG_DATA_DIR),
                        actions=actions,
                        velocities=velocities,
                        positions=positions,
                        angles=angles,
                        rewards=rewards,
                        dones=dones)
    
    print("collecting finished")


        
def main():
    model_path          = './models/run_005/Learner'
    arena_config_path   = './configs/3-Obstacles-short.yaml'
    
    trainer_config_path = './configs/trainer_config.yaml'
    env_path            = '../env/AnimalAICustom'

    agent = init_agent(trainer_config_path, model_path)
    seed = 0
    env = init_env(env_path, seed)

    arena_config_in = ArenaConfig(arena_config_path)
    agent.reset(t=arena_config_in.arenas[0].t)

    env.reset(arenas_configurations=arena_config_in)

    collect(env, agent, 1000)
    


if __name__ == '__main__':
    main()
