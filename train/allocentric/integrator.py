# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np


class EgocentricIntegrator(object):
    def __init__(self):
        pass

    def reset(self):
        # TODO: 複数Arenaの対応がまだ
        self.relative_angle_degree = 0

    def integrate(self, action, local_velocity):
        # TODO: 複数Arenaの対応がまだ
        # 前回のstateに対して、取ったactionとその結果のlocal velocity
        if action[0,1] == 1:
            # 右回転
            self.relative_angle_degree += 6
        elif action[0,1] == 2:
            # 左回転
            self.relative_angle_degree += -6
        self.relative_angle_degree = self.relative_angle_degree % 360

    @property
    def angle(self):
        """ 相対角度をradianで返す """
        relative_angle_radian = float(self.relative_angle_degree) / 360.0 * 2.0 * np.pi
        return relative_angle_radian
