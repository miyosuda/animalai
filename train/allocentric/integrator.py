# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

VELOCITY_CONSTANT = 0.0595


class EgocentricIntegrator(object):
    def __init__(self):
        pass

    def reset(self):
        # TODO: 複数Arenaの対応がまだ
        self.relative_angle_degree = 0
        self.local_pos = np.zeros([3], dtype=np.float32)
        print("reset") #..

    def debug_set_reset_pos_angle(self, absolute_pos, absolute_angle):
        self.debug_absolute_pos = np.copy(absolute_pos)
        self.debug_absolute_angle = absolute_angle

    def integrate(self, action, local_velocity):
        # TODO: 複数Arenaの対応がまだ
        
        # 位置の更新
        cur_angle_radian = self.angle
        scaled_local_velocity = local_velocity * VELOCITY_CONSTANT
        
        sin_angle = np.sin(cur_angle_radian)
        cos_angle = np.cos(cur_angle_radian)

        dx = scaled_local_velocity[0] * cos_angle + \
             scaled_local_velocity[2] * sin_angle
        dy = scaled_local_velocity[1]
        dz = scaled_local_velocity[0] * (-sin_angle) + \
             scaled_local_velocity[2] * cos_angle
        
        self.local_pos[0] += dx
        self.local_pos[1] += dy
        self.local_pos[2] += dz

        # 角度の更新
        # 前回のstateに対して、取ったactionとその結果のlocal velocity
        if action[0,1] == 1:
            # 右回転
            self.relative_angle_degree += 6
        elif action[0,1] == 2:
            # 左回転
            self.relative_angle_degree += -6
        self.relative_angle_degree = self.relative_angle_degree % 360

    def debug_confirm(self, local_velocity, absolute_pos, absolute_angle):
        """ 絶対角度を利用してデバッグで位置の積分計算を確認 """
        scaled_local_velocity = local_velocity * VELOCITY_CONSTANT

        sin_angle = np.sin(absolute_angle)
        cos_angle = np.cos(absolute_angle)
        
        dx = scaled_local_velocity[0] * cos_angle + \
             scaled_local_velocity[2] * sin_angle
        dy = scaled_local_velocity[1]
        dz = scaled_local_velocity[0] * (-sin_angle) + \
             scaled_local_velocity[2] * cos_angle
        
        self.debug_absolute_pos[0] += dx
        self.debug_absolute_pos[1] += dy
        self.debug_absolute_pos[2] += dz
        pos_diff = absolute_pos - self.debug_absolute_pos
        print("diff={}".format(pos_diff)) #..

    @property
    def angle(self):
        """ 相対角度をradianで返す """
        relative_angle_radian = float(self.relative_angle_degree) / 360.0 * 2.0 * np.pi
        return relative_angle_radian
