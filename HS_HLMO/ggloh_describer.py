import math
import time

import cv2
import numpy as np
import HS_HLMO.norm_coord


# 角度取 [-π/2, π/2)
class GGLOH():
    def __init__(self, NA, NO, R0=5):
        self.NA = NA
        self.NO = NO
        self.R0 = R0
        self.R1 = int(math.sqrt((R0 ** 2) * (NA + 1)))
        self.R2 = int(math.sqrt(NA * (R0 ** 2) + self.R1 ** 2))
        self.delta_A = 2 * math.pi / NA
        self.delta_O = math.pi / NO

        x = np.linspace(-self.R2, self.R2, 2 * self.R2 + 1)
        X, Y = np.meshgrid(x, x)
        self.angle_grid = np.arctan2(Y, X)
        self.distance_grid = np.sqrt(X**2 + Y ** 2)
        self.R0_mask = (self.distance_grid <= self.R0).astype(np.uint8)
        self.R1_mask = (self.R0 < self.distance_grid) & (self.distance_grid <= self.R1)
        self.R2_mask = (self.R1 < self.distance_grid) & (self.distance_grid <= self.R2)

    def __get_sectors_coords(self, center):
        """
        获取[0,R0], (R0,R1], (R1,R2]三个区域内的坐标点集
        :param center: 圆心
        :return: [0,R0], (R0,R1], (R1,R2]
        """
        coords0 = []
        coords0_1 = []
        coords1_2 = []
        cx, cy = center
        for x in range(cx - self.R2, cx + self.R2 + 1):
            for y in range(cy - self.R2, cy + self.R2 + 1):
                d = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                if self.R1 < d <= self.R2:
                    coords1_2.append((x, y))
                elif self.R0 < d <= self.R1:
                    coords0_1.append((x, y))
                elif d <= self.R0:
                    coords0.append((x, y))
        return coords0, coords0_1, coords1_2

    def __compute_orientation_assignment(self, relative_orientation):
        """
        根据相对角计算在直方图上的分配
        :param relative_orientation: 经过regular的相对角
        :return: (index0, val0), (index1, val1)
        """
        index = relative_orientation / self.delta_O  # 所处的位置
        index_down = int(index // 1)  # 向下取整
        index_up = index_down + 1 if index_down < self.NO - 1 else 0

        to_down = index - index_down  # 到下界距离
        to_up = 1.0 - to_down  # 到上界距离
        return (index_down, to_up), (index_up, to_down)

    @staticmethod
    def __regular_orientation(orientation):
        """
        将orientation值规范在[-angle, angle)之内
        """
        if orientation >= math.pi / 2:
            orientation -= math.pi
        elif orientation < -math.pi / 2:
            orientation += math.pi
        return orientation

    def __compute_descriptor_optimized(self, orientation_image, kpt):
        cx, cy = kpt
        descriptor = np.zeros((2 * self.NA + 1, self.NO), dtype=np.float32)
        reference_angle = orientation_image[cy, cx]     # 描述子的参考角度 (as 0°)

        # Divide into NA areas with reference
        index_A_grid = ((self.angle_grid - reference_angle) / self.delta_A // 1).astype(np.int32)
        temp_mask = (index_A_grid < 0).astype(np.int32) * self.NA
        index_A_grid += temp_mask

        #
        local_orientation_image = orientation_image[cy - self.R2: cy + self.R2 + 1, cx - self.R2: cx + self.R2 + 1]
        local_relative_image = local_orientation_image - reference_angle
        temp_mask = (local_relative_image < -math.pi / 2).astype(np.float32) * math.pi
        local_relative_image += temp_mask
        temp_mask = (local_relative_image >= math.pi / 2).astype(np.float32) * math.pi
        local_relative_image -= temp_mask

        # R0
        descriptor[0] = cv2.calcHist([local_relative_image], [0], mask=self.R0_mask, histSize=[self.NO],
                                     ranges=[-math.pi / 2, math.pi / 2]).reshape(-1)
        # R1 R2
        for index_A in range(self.NA):
            mask_A = (index_A_grid == index_A)
            mask = (self.R1_mask & mask_A).astype(np.uint8)
            descriptor[1 + index_A] = cv2.calcHist([local_relative_image], [0], mask=mask,
                                                             histSize=[self.NO], ranges=[-math.pi / 2, math.pi / 2]).reshape(-1)
            mask = (self.R2_mask & mask_A).astype(np.uint8)
            descriptor[1 + self.NA + index_A] = cv2.calcHist([local_relative_image], [0], mask=mask,
                                                             histSize=[self.NO], ranges=[-math.pi / 2, math.pi / 2]).reshape(-1)

        # Avoid angle jump
        D1 = np.zeros((2, int(self.NA / 2), self.NO))
        D2 = np.zeros_like(D1)
        D = np.zeros((4, int(self.NA / 2), self.NO))
        D1[:1] = descriptor[1: int(self.NA / 2 + 1)]
        D1[1:] = descriptor[self.NA + 1: int(self.NA / 2 * 3 + 1)]
        D2[:1] = descriptor[int(self.NA / 2 + 1): self.NA + 1]
        D2[1:] = descriptor[int(self.NA / 2 * 3 + 1): self.NA * 2 + 1]

        D[:2] = D1 + D2
        D[2:] = 1. * np.abs(D1 - D2)
        D = D.reshape((-1, self.NO))
        descriptor[1:] = D
        descriptor = descriptor.reshape(-1)
        return descriptor

    def __compute_descriptor(self, orientation_image, kpt):
        """
        计算关键点kpt的descriptor
        :param orientation_image: pmom特征图
        :param kpt: 关键点坐标
        :return: descriptor
        """
        x, y = kpt
        descriptor = np.zeros((2 * self.NA + 1, self.NO), dtype=np.float32)
        reference_angle = orientation_image[y, x]
        t = time.time()
        coords0, coords0_1, coords1_2 = self.__get_sectors_coords(kpt)
        print("GetSectorCoordCost:{}".format(time.time()-t))

        t = time.time()
        # 0~R0 区域 0
        for coord in coords0:
            relative_orientation = orientation_image[coord[1], coord[0]] - reference_angle
            relative_orientation = self.__regular_orientation(relative_orientation)
            down, up = self.__compute_orientation_assignment(relative_orientation)

            descriptor[0][down[0]] += down[1]
            descriptor[0][up[0]] += up[1]

        # R0~R1 区域 1~NA

        for coord in coords0_1:
            # 计算coord所属的扇区
            angle_A = math.atan2(coord[1] - y, coord[0] - x)  # [-π, π]
            relative_angle_A = angle_A - reference_angle

            index_A = int(relative_angle_A / self.delta_A // 1) # 向下取整

            # 计算其orientation分配
            relative_orientation = orientation_image[coord[1], coord[0]] - reference_angle
            relative_orientation = self.__regular_orientation(relative_orientation)
            down, up = self.__compute_orientation_assignment(relative_orientation)

            descriptor[1 + index_A, down[0]] += down[1]
            descriptor[1 + index_A, up[0]] += up[1]


        # R1~R2 区域 1+NA~2NA
        for coord in coords1_2:
            # 计算coord所属的扇区
            angle_A = math.atan2(coord[1] - y, coord[0] - x)  # [0, 2π]
            relative_angle_A = angle_A - reference_angle

            index_A = int(relative_angle_A / self.delta_A // 1) # 向下取整

            # 计算其orientation分配
            relative_orientation = orientation_image[coord[1], coord[0]] - reference_angle
            relative_orientation = self.__regular_orientation(relative_orientation)
            down, up = self.__compute_orientation_assignment(relative_orientation)

            descriptor[1 + self.NA + index_A, down[0]] += down[1]
            descriptor[1 + self.NA + index_A, up[0]] += up[1]
        print("GetDescriptorCost:{}".format(time.time() - t))

        # Avoid angle jump
        D1 = np.zeros((2, int(self.NA / 2), self.NO))
        D2 = np.zeros_like(D1)
        D = np.zeros((4, int(self.NA / 2), self.NO))
        D1[:1] = descriptor[1 : int(self.NA / 2 + 1)]
        D1[1:] = descriptor[self.NA + 1 : int(self.NA / 2 * 3 + 1)]
        D2[:1] = descriptor[int(self.NA / 2 + 1) : self.NA + 1]
        D2[1:] = descriptor[int(self.NA / 2 * 3 + 1) : self.NA * 2 + 1]

        D[:2] = D1 + D2
        D[2:] = 0.5 * np.abs(D1 - D2)
        D = D.reshape((-1, self.NO))
        descriptor[1:] = D
        descriptor = descriptor.reshape(-1)
        return descriptor

    def generate_descriptors(self, orientation_image: np.ndarray, keypoints: list):
        """
        生成GGLOH描述子
        :param orientation_image:
        :param keypoints: 归一化后的坐标
        :return:
        """
        descriptors = []
        height = orientation_image.shape[0]
        width = orientation_image.shape[1]

        orientation_image = np.pad(orientation_image, pad_width=self.R2, mode='constant')
        keypoints = HS_HLMO.norm_coord.denormalize_coord(keypoints, width, height)
        for kpt in keypoints:
            kpt = int(kpt[0] + self.R2), int(kpt[1] + self.R2)
            descriptors.append(self.__compute_descriptor_optimized(orientation_image, kpt))

        descriptors = np.array(descriptors)
        return descriptors
