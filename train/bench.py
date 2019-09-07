# -*- coding: utf-8 -*-
import random
import argparse
import numpy as np
import time

from animalai.envs.environment import UnityEnvironment
from animalai.envs.arena_config import ArenaConfig

env_path = '../env/AnimalAI'
#env_path = '../env/AnimalAICustom'
worker_id = random.randint(1, 100)

seed = 10
docker_target_name = True
resolution = 84

def main(args):
    docker_training = docker_target_name is not None

    env = UnityEnvironment(
        n_arenas=args.n_arenas,
        file_name=env_path,
        worker_id=worker_id,
        seed=seed,
        docker_training=docker_training,
        play=False,
        resolution=resolution
    )

    arena_config_in = ArenaConfig('configs/3-Obstacles.yaml')
    env.reset(arenas_configurations=arena_config_in)

    start_time = time.time()
    for i in range(args.frames):
        res = env.step(np.random.randint(0, 3, size=2 * args.n_arenas))

    elapsed_time = time.time() - start_time
    fps = float(args.frames) / elapsed_time
    print("n_arenas={0}, fps={1:.3f}".format(args.n_arenas, fps))
    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_arenas", type=int, default=4)
    parser.add_argument("--frames", type=int, default=1000)
    args = parser.parse_args()
    
    main(args)
