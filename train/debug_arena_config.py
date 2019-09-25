# -*- coding: utf-8 -*-
import numpy as np
import yaml

from animalai.envs.arena_config import ArenaConfig, Arena, Item, RGB, Vector3


def create_death_zone_arena(size, t):
    arena_config = ArenaConfig()

    # Put green reward
    item_goal = Item(name="GoodGoal",
                     positions=None,
                     rotations=None,
                     sizes=[Vector3(x=1,y=1,z=1)],
                     colors=None)

    # Put Death Zone
    item_death_zone = Item(name="DeathZone",
                           positions=None,
                           rotations=None,
                           sizes=[size],
                           colors=None)
    
    items = []
    items.append(item_goal)
    items.append(item_death_zone)
    arena = Arena(t=t, items=items, blackouts=None)
    return arena


def create_death_zone_arena_config(n_arenas=16, t=250):
    arena_config = ArenaConfig()
    
    for i in range(n_arenas):
        # サイズを指定: オリジナルは(1,0,1) - (40,0,40)
        sx = np.random.randint(1,35) # 1~34まで
        sz = np.random.randint(1,35) # 1~34まで
        size = Vector3(x=sx, y=0, z=sz)
        arena = create_death_zone_arena(size, t)
        arena_config.arenas[i] = arena
    
    return arena_config


arena_config = create_death_zone_arena_config()

f = open("out.yml", "w+")
f.write(yaml.dump(arena_config))
f.close()

arena_config2 = ArenaConfig("out.yml")

def debug_confirm_arena_config(env_path, arena_config):
    env_path = (env_path.strip()
                .replace('.app', '')
                .replace('.exe', '')
                .replace('.x86_64', '')
                .replace('.x86', ''))

    from animalai.envs.environment import UnityEnvironment
        
    env = UnityEnvironment(n_arenas=16,
                           file_name=env_path,
                           worker_id=1,
                           seed=0,
                           docker_training=False,
                           play=True)
    env.reset(arenas_configurations=arena_config)

    try:
        while True:
            continue
    except KeyboardInterrupt:
        env.close()
        
env_path = '../env/AnimalAI'
debug_confirm_arena_config(env_path, arena_config2)

