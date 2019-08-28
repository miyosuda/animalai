# -*- coding: utf-8 -*-
import yaml

from animalai.envs.gym.environment import AnimalAIEnv
from animalai.envs.arena_config import ArenaConfig
from animalai.envs.brain import BrainParameters

from trainers.ppo.policy import PPOPolicy

class Agent(object):
    def __init__(self,
                 trainer_confing_path,
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
        
        self.trainer_params = yaml.load(open(trainer_confing_path))['Learner']
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

    def step(self, obs, reward, done, info):
        brain_info = info['brain_info']
        action = self.policy.evaluate(brain_info=brain_info)['action']
        return action


def main():
    arena_config_path    = './configs/1-Food.yaml'
    env_path             = '../env/AnimalAI'

    trainer_confing_path = './configs/trainer_config.yaml'
    model_path           = './models/run_food1/Learner'
    
    agent = Agent(trainer_confing_path, model_path)
    arena_config_in = ArenaConfig(arena_config_path)

    agent.reset(t=arena_config_in.arenas[0].t)
    
    env = AnimalAIEnv(
        environment_filename=env_path,
        seed=0,
        retro=False,
        n_arenas=1,
        worker_id=1,
        docker_training=False,
        resolution=84
    )    

    for k in range(5):
        env.reset(arenas_configurations=arena_config_in)
        cumulated_reward = 0
        
        obs, reward, done, info = env.step([0, 0])
        
        #for i in range(arena_config_in.arenas[0].t):
        for i in range(250):
            action = agent.step(obs, reward, done, info)
            obs, reward, done, info = env.step(action)
            cumulated_reward += reward
            print(cumulated_reward)
            
            if done:
                break
            
    print('SUCCESS')


if __name__ == '__main__':
    main()
