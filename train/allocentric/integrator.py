# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

VELOCITY_CONSTANT = 0.0595


class EgocentricIntegrator(object):
    def __init__(self):
        pass

    def reset(self):
        # TODO: 複数Arenaの対応がまだ
        # Degreeでの積分された相対角度
        self.relative_angle_degree = 0
        # 開始位置、角度からの積分された相対位置
        self.local_pos = np.zeros([3], dtype=np.float32)
        print("reset") #..

    def calculate_dpos(self, angle_radian, local_velocity):
        """ local velocityとangle(localまたはgloal)から位置の変化量を計算 """
        scaled_local_velocity = local_velocity * VELOCITY_CONSTANT
        
        sin_angle = np.sin(angle_radian)
        cos_angle = np.cos(angle_radian)

        dx = scaled_local_velocity[0] * cos_angle + \
             scaled_local_velocity[2] * sin_angle
        dy = scaled_local_velocity[1]
        dz = scaled_local_velocity[0] * (-sin_angle) + \
             scaled_local_velocity[2] * cos_angle
        return np.array([dx, dy, dz], dtype=np.float32)

    def integrate(self, action, local_velocity):
        # 角度の更新
        # 前回のstateに対して、取ったactionとその結果のlocal velocity
        if action[0,1] == 1:
            # 右回転
            self.relative_angle_degree += 6
        elif action[0,1] == 2:
            # 左回転
            self.relative_angle_degree += -6
        self.relative_angle_degree = self.relative_angle_degree % 360

        # 更新後の角度を位置更新に利用している
        updated_angle_radian = self.angle
        dpos = self.calculate_dpos(updated_angle_radian, local_velocity)
        self.local_pos += dpos

    @property
    def angle(self):
        """ 相対角度をradianで返す """
        relative_angle_radian = float(self.relative_angle_degree) / 360.0 * 2.0 * np.pi
        return relative_angle_radian

    def debug_set_reset_pos_angle(self, absolute_pos, absolute_angle):
        """ デバッグ用にreset時の情報を受け取る """
        self.debug_absolute_pos = np.copy(absolute_pos)
        self.debug_absolute_angle = absolute_angle

        # reset時の情報を保存しておく
        self.debug_initial_absolute_pos = np.copy(absolute_pos)
        self.debug_initial_absolute_angle = absolute_angle

    def debug_confirm(self, local_velocity, absolute_pos, absolute_angle):
        """ 絶対角度を利用してデバッグで位置の積分計算を確認 """
        dpos = self.calculate_dpos(absolute_angle, local_velocity)
        self.debug_absolute_pos += dpos

        pos_diff = absolute_pos - self.debug_absolute_pos
        #print("diff={}".format(pos_diff)) #..

        integrated_pos = self.debug_integrated_absolute_pos
        pos_diff2 = absolute_pos - integrated_pos
        print("diff2={}".format(pos_diff2)) #..
        
    @property
    def debug_integrated_absolute_angle(self):
        """ デバッグ用の絶対角度をradianで返す """
        return self.debug_initial_absolute_angle + self.angle

    @property
    def debug_integrated_absolute_pos(self):
        """ デバッグ用の絶対位置を返す """
        # self.local_posをself.debug_initial_absolute_angleで変換し、
        # self.debug_initial_absolute_posを足す.

        # TODO: calculate_dpos()にまとめられる.
        # (その場合はVELOCITY_CONSTANTをかけないようにする必要あり)
        sin_angle = np.sin(self.debug_initial_absolute_angle)
        cos_angle = np.cos(self.debug_initial_absolute_angle)
        dx = self.local_pos[0] * cos_angle + \
             self.local_pos[2] * sin_angle
        dy = self.local_pos[1]
        dz = self.local_pos[0] * (-sin_angle) + \
             self.local_pos[2] * cos_angle
        dpos = np.array([dx, dy, dz], dtype=np.float32)
        #dpos = self.calculate_dpos(self.debug_initial_absolute_angle,
        #                           self.local_pos)
        absolute_pos = self.debug_initial_absolute_pos + dpos
        return absolute_pos
