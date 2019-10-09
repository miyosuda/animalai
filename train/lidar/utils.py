# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import os


def load_checkpoints(sess, save_dir):
    saver = tf.train.Saver(max_to_keep=2)
    checkpoint_dir = save_dir + "/checkpoints"

    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
    if checkpoint and checkpoint.model_checkpoint_path:
        # load from checkpoint.
        saver.restore(sess, checkpoint.model_checkpoint_path)
        # Retrieve step count from the file name.
        tokens = checkpoint.model_checkpoint_path.split("-")
        step = int(tokens[1])
        print("Loaded checkpoint: {0}, step={1}".format(
            checkpoint.model_checkpoint_path, step))
        return saver, step + 1
    else:
        print("Could not find old checkpoint")
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        return saver, 0


def save_checkponts(sess, saver, global_step, save_dir):
    checkpoint_dir = save_dir + "/checkpoints"
    saver.save(sess, checkpoint_dir + '/' + 'checkpoint', global_step=global_step)
    print("Checkpoint saved")


"""
0: CylinderTunnel
1: CylinderTunnelTransparent
2: Ramp
3: Wall
4: WallTransparent
5: Cardbox1
6: Cardbox2
7: LObject
8: LObject2
9: UObject
10: BadGoal
11: BadGoalBounce
12: DeathZone
13: GoodGoal
14: GoodGoalBounce
15: GoodGoalMulti
16: GoodGoalMultiBounce
17: HotZone
18: WallOut1(外壁1)
19: WallOut2(外壁2)
20: WallOut3(外壁3)
21: WallOut4(外壁4)
"""

TARGET_CYLINDER_TUNNEL             = 0
TARGET_CYLINDER_TUNNEL_TRANSPARENT = 1
TARGET_RAMP                        = 2
TARGET_WALL                        = 3
TARGET_WALL_TRANSPARENT            = 4
TARGET_CARDBOX                     = 5
TARGET_LUOBJECT                    = 6
TARGET_BAD_GOAL                    = 7
TARGET_DEATH_ZONE                  = 8
TARGET_GOOD_GOAL                   = 9
TARGET_GOOD_GOAL_MULTI             = 10
TARGET_HOT_ZONE                    = 11
TARGET_WALL_OUT                    = 12

TARGET_ID_MAX                      = 13

target_id_tables = {
    0  : TARGET_CYLINDER_TUNNEL, # CylinderTunnel
    1  : TARGET_CYLINDER_TUNNEL_TRANSPARENT, # CylinderTunnelTransparent
    2  : TARGET_RAMP, # Ramp
    3  : TARGET_WALL, # Wall
    4  : TARGET_WALL_TRANSPARENT, # WallTransparent
    5  : TARGET_CARDBOX, # Cardbox1
    6  : TARGET_CARDBOX, # Cardbox2
    7  : TARGET_LUOBJECT, # LObject
    8  : TARGET_LUOBJECT, # LObject2
    9  : TARGET_LUOBJECT, # UObject
    10 : TARGET_BAD_GOAL, # BadGoal
    11 : TARGET_BAD_GOAL, # BadGoalBounce
    12 : TARGET_DEATH_ZONE, # DeathZone
    13 : TARGET_GOOD_GOAL, # GoodGoal
    14 : TARGET_GOOD_GOAL, # GoodGoalBounce
    15 : TARGET_GOOD_GOAL_MULTI, # GoodGoalMulti
    16 : TARGET_GOOD_GOAL_MULTI, # GoodGoalMultiBounce
    17 : TARGET_HOT_ZONE, # HotZone
    18 : TARGET_WALL_OUT, # WallOut1(外壁1)
    19 : TARGET_WALL_OUT, # WallOut2(外壁2)
    20 : TARGET_WALL_OUT, # WallOut3(外壁3)
    21 : TARGET_WALL_OUT, # WallOut4(外壁4)
}

target_names = [
    "CYL TNL",
    "CYL TNL(TR)",
    "RAMP",
    "WALL",
    "WALL(TR)",
    "CARDBOX",
    "LUOBJECT",
    "BAD GOAL",
    "DEATH ZONE",
    "GOOD GOAL",
    "GOOD GOAL MLT",
    "HOT ZONE",
    "WALL OUT",
]


def convert_target_ids(raw_target_ids):
    target_ids = []
    for raw_target_id in raw_target_ids:
        target_id = target_id_tables[raw_target_id]
        target_ids.append(target_id)
    return np.array(target_ids, dtype=np.int8)


def get_target_names(target_ids):
    names = []
    for target_id in target_ids:
        names.append(target_names[target_id])
    return names
