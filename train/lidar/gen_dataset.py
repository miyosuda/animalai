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

from lidar.utils import convert_target_ids

BASE_DATA_DIR = "lidar_base_data"
DATA_DIR = "lidar_data"
FILE_SIZE_IN_DIR = 1000


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

        self.policy = PPOPolicy(brain=self.brain,
                                seed=0,
                                trainer_params=self.trainer_params,
                                is_training=False,
                                load=True)

    def reset(self, t=250):
        pass

    def fix_brain_info(self, brain_info):
        velocity = brain_info.vector_observations[:,:3]
        
        if brain_info.vector_observations.shape[1] > 3:
            # カスタム環境用にvector_observationsをいじったものだった場合
            extended_infos = brain_info.vector_observations[:,3:]
            
            # 元の3次元だけのvector_observationsに戻す
            brain_info.vector_observations = brain_info.vector_observations[:,:3]
            agent_pos   = extended_infos[:,:3]
            agent_angle = extended_infos[:,3] / 360.0 * (2.0 * np.pi) # 0~2pi
            
            target_ids       = extended_infos[:,4:4+5].astype(np.int32)
            target_distances = extended_infos[:,9:9+5]
            
            return velocity[0], (agent_pos[0], agent_angle[0]), \
                (target_ids[0], target_distances[0])
        else:
            return velocity[0], None, None

    def step(self, obs, reward, done, info):
        brain_info = info['brain_info']
        out = self.fix_brain_info(brain_info) # Custom環境でのみの情報
        velocity, pos_angle, target_ids_distances = out
        # (3,) (3,) (), ()
        
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


def save_state(state, dir_path, i):
    file_name = "{}/image{}.png".format(dir_path, i)
    state = (state*255.0).astype(np.uint8)
    image = cv2.cvtColor(state, cv2.COLOR_RGB2BGR)
    cv2.imwrite(file_name, image)


def load_state(index):
    dir_path = "{}/dir{}".format(BASE_DATA_DIR, index // FILE_SIZE_IN_DIR)
    file_name = "{}/image{}.png".format(dir_path, index)
    image = cv2.imread(file_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def collect(env, agent, step_size):
    obs = None

    if not os.path.exists(BASE_DATA_DIR):
        os.mkdir(BASE_DATA_DIR)

    start_time = time.time()

    velocities = []
    actions = []
    positions = []
    angles = []
    rewards = []
    dones = []
    target_ids = []
    target_distances = []

    for i in range(step_size):
        if i % FILE_SIZE_IN_DIR == 0:
            dir_path = "{}/dir{}".format(BASE_DATA_DIR,i // FILE_SIZE_IN_DIR)
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)

        if obs == None:
            # 初回のstate生成
            last_action = np.array([0,0], dtype=np.int32)
            obs, reward, done, info = env.step(last_action)

        # 実際のstateやangle, positionが入っているのはinfo

        last_state = obs[0] # dtype=float64

        # obsの状態に対してpolicyがactionを決定
        out = agent.step(obs, reward, done, info)
        action, _, _, _, last_velocity, last_pos_angle, last_target_ids_distances = out
        # action=(1,2)
        # last_velocity_pos_angleはaction発行前のvelocity, pos, angle
        # actionは、last_stateにおいて発行したAction
        
        
        # Envに対してactionを発行し発行して結果を得る
        obs, reward, done, info = env.step(action)

        # action発行前のstate
        save_state(last_state, dir_path, i)

        last_pos      = last_pos_angle[0]
        last_angle    = last_pos_angle[1] / 360.0 * (2.0 * np.pi)
        
        last_raw_target_ids   = last_target_ids_distances[0]
        last_target_distances = last_target_ids_distances[1]
        last_target_ids = convert_target_ids(last_raw_target_ids)
        
        actions.append(np.squeeze(last_action))
        # last_stateの状態になる前に発行したAction (inputになる)
        # (このactionを発行してlast_stateに遷移した)
        # (1,2) -> (2,)
        
        velocities.append(last_velocity) # Action発酵前のlocal velocity (inputになる)
        positions.append(last_pos)       # Action発行前のpos (output targetになる)
        angles.append(last_angle)        # Action発行前のangle (output targetになる)
        rewards.append(reward)           # Actionを発行して得たreward
        dones.append(done)               # Actionを発行してterminateしたかどうか
        target_ids.append(last_target_ids)             # Action発酵前のターゲットのID列
        target_distances.append(last_target_distances) # Action発酵前のターゲットの距離

        last_action = action

        # 保存内容は通常の強化学習での (S_t,A_t,R_t+1,Term_t+1)の対
        if done:
            obs = None

        if i % 1000 == 0:
            print("step{}".format(i))
            elapsed_time = time.time() - start_time
            print("fps={}".format(i / elapsed_time))

    np.savez_compressed("{}/base_infos".format(BASE_DATA_DIR),
                        actions=actions,
                        velocities=velocities,
                        positions=positions,
                        angles=angles,
                        rewards=rewards,
                        dones=dones,
                        target_ids=target_ids,
                        target_distances=target_distances)
    
    print("collecting finished")


def generate(frame_size):
    data_path = "{}/base_infos.npz".format(BASE_DATA_DIR)
    data_all = np.load(data_path)
    
    data_actions          = data_all["actions"]          # (n, 2)
    data_velocities       = data_all["velocities"]       # (n, 3)
    data_positions        = data_all["positions"]        # (n, 3)
    data_angles           = data_all["angles"]           # (n,)
    data_rewards          = data_all["rewards"]          # (n,)
    data_dones            = data_all["dones"]            # (n,)
    data_target_ids       = data_all["target_ids"]       # (n, 5)
    data_target_distances = data_all["target_distances"] # (n, 5)

    frame_size = len(data_dones)

    seq_start_index = 0
    # 1シーケンスの長さ
    seq_length = 20

    seq_start_indices = []

    while(seq_start_index < frame_size):
        for i in range(seq_length):
            index = seq_start_index + i

            if i == seq_length-1:
                # 最後までシーケンスを流せた
                seq_start_indices.append(seq_start_index)
                seq_start_index = index+1
                break
            elif data_dones[index] == True:
                # このseqは無効なので捨てる
                seq_start_index = index+1
                break
            elif index >= frame_size-1:
                # 最後まで来たので無効として外側のループを抜ける
                seq_start_index = index + 1
                break

    extracted_seq_size = len(seq_start_indices)

    extracted_states     = np.empty((extracted_seq_size, seq_length, 84, 84, 3), dtype=np.uint8)
    extracted_actions    = np.empty((extracted_seq_size, seq_length, 2), dtype=np.int32)
    extracted_velocities = np.empty((extracted_seq_size, seq_length, 3), dtype=np.float32)
    extracted_positions  = np.empty((extracted_seq_size, seq_length, 3), dtype=np.float32)
    extracted_angles     = np.empty((extracted_seq_size, seq_length, 1), dtype=np.float32)
    extracted_rewards    = np.empty((extracted_seq_size, seq_length, 1), dtype=np.float32)
    extracted_target_ids = np.empty((extracted_seq_size, seq_length, 5), dtype=np.int8)
    extracted_target_distances = np.empty((extracted_seq_size, seq_length, 5), dtype=np.float32)

    for seq_id, seq_start_index in enumerate(seq_start_indices):
        for i in range(seq_length):
            index = seq_start_index + i
            
            state = load_state(index)
            
            action           = data_actions[index]
            velocity         = data_velocities[index]
            position         = data_positions[index]
            angle            = data_angles[index]
            reward           = data_rewards[index]
            target_ids       = data_target_ids[index]
            target_distances = data_target_distances[index]

            extracted_states[seq_id, i]              = state
            extracted_actions[seq_id, i, :]          = action
            extracted_velocities[seq_id, i, :]       = velocity
            extracted_positions[seq_id, i, :]        = position
            extracted_angles[seq_id, i, :]           = angle
            extracted_rewards[seq_id, i, :]          = reward
            extracted_target_ids[seq_id, i, :]       = target_ids
            extracted_target_distances[seq_id, i, :] = target_distances

        if seq_id % 100 == 0:
            print("process seq={}".format(seq_id))

    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)

    np.savez_compressed("{}/states".format(DATA_DIR),
                        states=extracted_states)

    np.savez_compressed("{}/infos".format(DATA_DIR),
                        actions=extracted_actions,
                        velocities=extracted_velocities,
                        positions=extracted_positions,
                        angles=extracted_angles,
                        rewards=extracted_rewards,
                        target_ids=extracted_target_ids,
                        target_distances=extracted_target_distances)
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--step_size",
                        help="Training step size",
                        type=int,
                        default=300000) # 以前は150000
    args = parser.parse_args()

    step_size = args.step_size
    
    model_path          = './models/run_lidar0/Leaner/'
    arena_config_path   = './configs/lidar/obstacle-w-t-wt-tt-cb-ulo-rm.yaml'
    trainer_config_path = './configs/lidar/trainer_config_lidar.yaml'
    env_path            = '../env/AnimalAIScan'

    agent = init_agent(trainer_config_path, model_path)
    env = init_env(env_path, args.seed)

    arena_config_in = ArenaConfig(arena_config_path)
    agent.reset(t=arena_config_in.arenas[0].t)

    env.reset(arenas_configurations=arena_config_in)

    collect(env, agent, step_size)
    
    generate(step_size)


if __name__ == '__main__':
    main()
