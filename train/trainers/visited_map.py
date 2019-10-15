# -*- coding: utf-8 -*-
import numpy as np
import math
import cv2


VELOCITY_CONSTANT = 0.0595


class VisitedMap:
    GRID_DIVISION = 40
    #GRID_DIVISION = 30
    TARGET_ID_MAX = 13
    RANGE_MAX =  40.0 * np.sqrt(2.0)
    RANGE_MIN = -40.0 * np.sqrt(2.0)
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.last_local_angle = 0 # Radian
        self.last_local_position = np.zeros((3), dtype=np.float)

        self.target_grid_probs = np.zeros([self.GRID_DIVISION,
                                           self.GRID_DIVISION,
                                           self.TARGET_ID_MAX], dtype=np.float32)
        self.target_grid_counts = np.zeros([self.GRID_DIVISION, self.GRID_DIVISION],
                                           dtype=np.int32)
        self.visited_grid = np.zeros([self.GRID_DIVISION, self.GRID_DIVISION],
                                     dtype=np.int32)
        
    def add_visited_info(self,
                         local_done,
                         previous_vector_action,
                         vector_observation,
                         lidar_id_probs,
                         lidar_distances):
        
        if local_done:
            self.reset()
            
        local_angle, local_position = self.get_local_angle_pos(local_done,
                                                               previous_vector_action,
                                                               vector_observation)
        
        self.last_local_angle = local_angle # Radian
        self.last_local_position = local_position
        
        lidar_angles = [-20, -10, 0, 10, 20]
        for lidar_angle, lidar_id_prob, lidar_distance in zip(lidar_angles,
                                                              lidar_id_probs,
                                                              lidar_distances):
            target_angle = lidar_angle + local_angle
            d_target_pos = self.rotate_array([0.0, 0.0, lidar_distance], target_angle)
            target_pos = local_position + np.array(d_target_pos, dtype=np.float32)
            self.record_target(lidar_id_prob, target_pos)

        self.record_self_pos(local_position)

    def get_local_angle_pos(self,
                            local_done,
                            previous_vector_action,
                            vector_observation):
        if local_done:
            new_local_angle = 0
            new_local_position = np.zeros((3), dtype=np.float)
        else:
            if previous_vector_action[1] == 1: # turn right
                new_local_angle = (self.last_local_angle + 6) % 360
            elif previous_vector_action[1] == 2: # turn left
                new_local_angle = (self.last_local_angle - 6) % 360
            else:
                new_local_angle = self.last_local_angle % 360
            rot = self.rotate_array(vector_observation, new_local_angle)
            new_local_position = self.last_local_position + \
                                 VELOCITY_CONSTANT * np.array(rot, dtype=np.float)
        return new_local_angle, new_local_position

    def record_target(self, id_probs, local_position):
        grid_x, grid_z = self.get_grid_pos(local_position)

        cur_grid_probs = self.target_grid_probs[grid_z, grid_x, :]
        grid_count = self.target_grid_counts[grid_z, grid_x]
        if grid_count == 0:
            new_grid_probs = id_probs
        else:
            new_grid_probs = cur_grid_probs * (grid_count/(grid_count+1)) + \
                             id_probs/(grid_count+1)
        self.target_grid_probs[grid_z, grid_x, :] = new_grid_probs
        self.target_grid_counts[grid_z, grid_x] += 1

    def record_self_pos(self, local_position):
        grid_x, grid_z = self.get_grid_pos(local_position)
        self.visited_grid[grid_z, grid_x] += 1

    def get_grid_pos(self, local_position):
        relative_pos_x = (local_position[0] - self.RANGE_MIN) / (self.RANGE_MAX - self.RANGE_MIN)
        relative_pos_z = (local_position[2] - self.RANGE_MIN) / (self.RANGE_MAX - self.RANGE_MIN)
        
        relative_pos_x = np.clip(relative_pos_x, 0.0, 1.0)
        relative_pos_z = np.clip(relative_pos_z, 0.0, 1.0)
        
        grid_x = int(math.floor(relative_pos_x * self.GRID_DIVISION))
        grid_z = int(math.floor(relative_pos_z * self.GRID_DIVISION))
        
        if grid_x >= self.GRID_DIVISION: grid_x = self.GRID_DIVISION-1
        if grid_z >= self.GRID_DIVISION: grid_z = self.GRID_DIVISION-1

        grid_z = (self.GRID_DIVISION-1) - grid_z
        return grid_x, grid_z

    def rotate_array(self, pos, angle):
        sin_angle = np.sin(2.0 * np.pi * float(angle) / 360.0)
        cos_angle = np.cos(2.0 * np.pi * float(angle) / 360.0)
        dx = cos_angle * pos[0] + sin_angle * pos[2]
        dy = pos[1]
        dz = -sin_angle * pos[0] + cos_angle * pos[2]
        return [dx, dy, dz]

    def get_image(self):
        local_map = self.get_local_map_image()
        local_map = local_map.reshape([self.GRID_DIVISION, self.GRID_DIVISION, 1])
        # (40,40,1)
        # これはstart位置角度座標系での40x40のmap.
        
        # 40x40の画像の1pixelが、Arenaの80xsqrt(2)/40=2.0*sqrt(2)の距離に相当する.
        
        scale = 2.0 # 40x40の画像を何倍するか
        M = cv2.getRotationMatrix2D(center=(self.GRID_DIVISION//2, self.GRID_DIVISION//2),
                                    angle=self.last_local_angle,
                                    scale=scale)
        # 画像の1pixelが、2.0*sqrt(2)/scaleの距離になっている.
        
        # (2,3)
        shift_cc0 = (84/2-self.GRID_DIVISION//2)
        shift_cc1 = (84/2-self.GRID_DIVISION//2)
        
        M[0,2] += shift_cc0
        M[1,2] += shift_cc1
        # これで40x40画像の中心が、84x84の中心にくるシフト分.
        
        pos_rate = (1.0/scale) * (self.RANGE_MAX - self.RANGE_MIN) / self.GRID_DIVISION

        # local_positionを画素のサイズに変換
        local_pos_pix_x = self.last_local_position[0] / pos_rate
        local_pos_pix_z = self.last_local_position[2] / pos_rate

        shift_lp = self.rotate_array([-local_pos_pix_x, 0.0, local_pos_pix_z],
                                     self.last_local_angle)

        #print("last_pos={}".format(self.last_local_position)) #..
        #print("shift_lp={}".format(shift_lp)) #..

        M[0,2] += shift_lp[0]
        M[1,2] += shift_lp[2]

        interpolation = cv2.INTER_NEAREST
        #interpolation = cv2.INTER_LINEAR
        image = cv2.warpAffine(src=local_map,
                               M=M,
                               dsize=(84,84),
                               flags=interpolation)
        image = image.reshape([84,84,1])
        return image

    def get_local_map_image(self):
        """
        Get local map (start-pos-angle coordinate)
        map values:
           0.0:        LIDAR not scanned
           1/13 ~ 1.0: LIDAR scanned
           2.0:        Visited
        """
        grid_on_map = np.clip(self.target_grid_counts, 0, 1) # 1 if count > 0
        grid_id_map = np.argmax(self.target_grid_probs, axis=2) # 0~12
        grid_extended_id_map = grid_on_map + grid_id_map # 0 if no-count, otherwise 1~13.
        grid_extended_id_map = grid_extended_id_map.astype(np.float32) / self.TARGET_ID_MAX
        # 0.0 ~ 1.0
        
        visited_map = np.clip(self.visited_grid, 0, 1) * 2 # 2 if visited, otherwise 0
        local_map = np.clip(grid_extended_id_map + visited_map.astype(np.float32), 0.0, 2.0)
        #local_map = np.clip(visited_map.astype(np.float32), 0.0, 2.0) # 行ったとこのみを表示する場合
        
        # 行った場所: 2.0
        # LIDAR未スキャン範囲: 0.0
        # LIDARスキャン範囲: 1/13~13/13(=1.0)の間を1/13区切りで、ターゲットのID毎に値を入れる.
        return local_map
