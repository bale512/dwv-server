import copy
import itertools
import math
import os
import random
import sys
import time

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import skimage.morphology as sm
from scipy.ndimage import measurements

import GeodisTK
import nibabel as nib


class State():
    def __init__(self, size, click_mode='multi'):
        self.state = np.zeros(size, dtype=np.float32)
        self.act0 = - 0.1
        self.act1 = 0.1
        self.pred_thd = 0.5
        self.click_mode = click_mode
        random.seed(30)
        # self.move_range = move_range
        # self.structure = np.ones((3,3,3))
        self.structure = np.array(
            [[[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 1, 0], [1, 1, 1], [0, 1, 0]], [[0, 0, 0], [0, 1, 0], [0, 0, 0]]])

    def reset(self, img, prob, gt):
        # img: batch x 1 x a x b x h
        # pred: batch x 1 x a x b x h
        # prob: batch x 1 x a x b x h
        # gt: batch x a x b x h
        # hint_map: batch x 2 x a x b x h
        self.img = img
        self.prob = prob
        self.pred = (self.prob > self.pred_thd).astype(np.float)
        self.gt = gt

        self.create_hint_map()
        self.state = np.concatenate((self.img, self.prob, self.hint_map), 1)  # batch x 4 x a x b x h

    def reset_predict(self, img, prob, hints):
        # img: batch x 1 x a x b x h
        # pred: batch x 1 x a x b x h
        # prob: batch x 1 x a x b x h
        # gt: batch x a x b x h
        # hint_map: batch x 2 x a x b x h
        self.hints = hints
        self.img = img
        self.prob = prob
        self.pred = (self.prob > self.pred_thd).astype(np.float)
        self.create_hint_map_predict(hints)
        self.state = np.concatenate((self.img, self.prob, self.hint_map), 1)  # batch x 4 x a x b x h

    def step(self, act):
        # act: 0: -0.4, 1: -0.2, 2: -0.1, 3: 0.1, 4: 0.2, 5: 0.4
        # act: batch x a x b x h
        # pred: batch x 1 x a x b x h
        # prob: batch x 1 x a x b x h

        self.prob += (act == 0) * self.act0 * 4 + \
                     (act == 1) * self.act0 * 2 + \
                     (act == 2) * self.act0 + \
                     (act == 3) * self.act1 + \
                     (act == 4) * self.act1 * 2 + \
                     (act == 5) * self.act1 * 4

        self.prob = np.maximum(np.minimum(self.prob, 1.0), 0.0)
        self.pred = (self.prob > self.pred_thd).astype(np.float)

        self.create_hint_map()
        self.state = np.concatenate((self.img, self.prob, self.hint_map), 1)

    def step_predict(self, act):
        # act: 0: -0.4, 1: -0.2, 2: -0.1, 3: 0.1, 4: 0.2, 5: 0.4
        # act: batch x a x b x h
        # pred: batch x 1 x a x b x h
        # prob: batch x 1 x a x b x h

        self.prob += ((act == 0) * self.act0 * 4 +
                      (act == 1) * self.act0 * 2 +
                      (act == 2) * self.act0 +
                      (act == 3) * self.act1 +
                      (act == 4) * self.act1 * 2 +
                      (act == 5) * self.act1 * 4).astype(np.float64)

        self.prob = np.maximum(np.minimum(self.prob, 1.0), 0.0)
        self.pred = (self.prob > self.pred_thd).astype(np.float)


    def create_hint_map_predict(self, hints):
        self.hints = hints
        hint_map = make_gt_geodesic(self.img[0, 0], hints)
        self.hint_map = hint_map[np.newaxis]

    def create_hint_map(self):
        batch_size = self.pred.shape[0]
        assert batch_size == 1
        self.produce_largest_error_regions()  # 先根据pred和gt，生成 error_regions（每个label有各自的 error_region_map）

        if self.click_mode == 'multi':
            # max number of click
            self.produce_hint_map_multi()  # 再根据error_regions，生成 hint点和hint map：
        elif self.click_mode == 'force':
            # exact number of click
            self.produce_hint_map_multi_force()
        else:
            raise ValueError("No such click mode!")

    def produce_largest_error_regions(self, class_num=2):
        s = 5  # side length of the square we consider for removing thin error regions
        r = (s - 1) // 2
        thd = 0.2  # percentage threshold we use for removing thin error regions. If lower, remove

        a, b, h = self.pred[0, 0].shape
        diff_all = (self.pred[0, 0] != self.gt[0]) * 1
        labeled_ccs = []
        for i in range(class_num):
            diff = (diff_all & (self.gt[0] == i)) * 1

            if np.sum(diff) != 0:
                labeled_cc, num_cc = measurements.label(diff, structure=self.structure)
                record_pad = np.zeros([a + 2 * r, b + 2 * r, h + 2 * r])
                labeled_cc_pad = np.zeros([a + 2 * r, b + 2 * r, h + 2 * r])
                labeled_cc_pad[r:r + a, r:r + b, r:r + h] = labeled_cc
                for ir in range(s):
                    for jr in range(s):
                        for kr in range(s):
                            record_pad[ir:ir + a, jr:jr + b, kr:kr + h] += (
                                    (labeled_cc_pad[ir:ir + a, jr:jr + b, kr:kr + h] == labeled_cc) & (
                                    labeled_cc != 0))
                record = (record_pad[r:r + a, r:r + b, r:r + h] < s * s * s * thd) * 1
                labeled_cc[record == 1] = 0

                new_diff = copy.deepcopy(labeled_cc)
                new_diff[new_diff != 0] = 1
                if np.sum(new_diff) != 0:
                    labeled_cc, num_cc = measurements.label(new_diff, structure=self.structure)
                else:
                    labeled_cc = np.zeros_like(new_diff)
            else:
                labeled_cc = np.zeros_like(diff)

            labeled_ccs.append(labeled_cc)
        labeled_ccs = np.stack(labeled_ccs, axis=0)  # class_num x a x b x h

        self.error_region = labeled_ccs[np.newaxis]

    def produce_hint_map_multi(self, cc_thd=30, max_hint_num=5, hint_range=5, shape='geodesic'):
        # error_regions按面积从大到小排序，每个 error_region生成一个hint点（在error region的中心，
        # 即与所有boundary point的最小距离最小的那个点，在此基础上再加六个方向随机的 [0,3]扰动），直到达到self.max_hint_num或没有满足条件的 error_region
        # cc_thd: 能接受的error region最小面积，小于这个面积的直接忽略
        # shape: 默认用geodesic
        class_num, a, b, h = self.error_region[0].shape
        hints = []
        large_ccs = []  # list of [num, idx, label]
        for c in range(class_num):
            hints.append([])
            if np.sum(self.error_region[0][c, ...]) != 0:
                # large_nums = []
                labeled_cc = self.error_region[0][c, ...]
                labeled_flat = np.reshape(labeled_cc, -1)
                if len(np.bincount(labeled_flat)) > 1:
                    lcount = np.bincount(labeled_flat)[1:]
                    sort_lcount = np.sort(lcount)[::-1]
                    # print(sort_lcount)
                    argsort_lcount = np.argsort(lcount)[::-1]
                    for arg in range(len(lcount)):
                        if sort_lcount[arg] < cc_thd:
                            break
                        large_ccs.append([sort_lcount[arg], argsort_lcount[arg] + 1, c])
                        # large_nums.append(sort_lcount[arg])

        sorted_large_ccs = sorted(large_ccs, key=lambda x: x[0], reverse=True)
        for i in range(min(len(sorted_large_ccs), max_hint_num)):
            largest_id = sorted_large_ccs[i][1]
            label = sorted_large_ccs[i][2]
            labeled_cc = self.error_region[0][label, ...]
            points = np.where(labeled_cc == largest_id)
            num_point = len(points[0])
            # print("points",points)
            # print("num_point",num_point)
            boundary_points = []
            inner_points = []
            for kk in range(0, num_point, 8):
                # print("kk",kk)
                xx = points[0][kk]
                yy = points[1][kk]
                zz = points[2][kk]
                if xx == 0 or xx == a - 1 or yy == 0 or yy == b - 1 or zz == 0 or zz == h - 1 \
                        or labeled_cc[xx - 1][yy][zz] != largest_id or labeled_cc[xx + 1][yy][zz] != largest_id \
                        or labeled_cc[xx][yy - 1][zz] != largest_id or labeled_cc[xx][yy + 1][zz] != largest_id \
                        or labeled_cc[xx][yy][zz - 1] != largest_id or labeled_cc[xx][yy][zz + 1] != largest_id:
                    boundary_points.append([xx, yy, zz])
                else:
                    inner_points.append([xx, yy, zz])
            num_bd_point = len(boundary_points)
            num_in_point = len(inner_points)
            # print("%d boundary points and %d inner points" % (num_bd_point, num_in_point))

            boundary_points = np.asarray(boundary_points).reshape(1, -1, 3)
            inner_points = np.asarray(inner_points).reshape(-1, 1, 3)
            dis_map = np.amin(np.sum(np.abs(boundary_points - inner_points), axis=2), axis=1)
            if dis_map.size:
                chosen_id = np.argmax(dis_map)

                rand = np.random.randint(7, size=3) - 3
                x = min(max(inner_points[chosen_id, 0, 0] + rand[0], 0), a - 1)
                y = min(max(inner_points[chosen_id, 0, 1] + rand[1], 0), b - 1)
                z = min(max(inner_points[chosen_id, 0, 2] + rand[2], 0), h - 1)
                if labeled_cc[x][y][z] != largest_id:
                    x = inner_points[chosen_id, 0, 0]
                    y = inner_points[chosen_id, 0, 1]
                    z = inner_points[chosen_id, 0, 2]

                hints[label].append([x, y, z])
                # print("pred", self.pred[0,0,x,y,z])
                # print("gt", self.gt[0,x,y,z])

        if shape == 'gaussian':
            hint_map = make_gt_gaussian(a, b, h, hints, sigma=hint_range)
        elif shape == 'euclidean':
            hint_map = make_gt_euclidean(a, b, h, hints)
        elif shape == 'geodesic':
            hint_map = make_gt_geodesic(self.img[0, 0], hints)

        self.hint_map = hint_map[np.newaxis]
        self.hints = hints

    def produce_hint_map_multi_force(self, cc_thd=30, max_hint_num=5, hint_range=5, shape='geodesic'):
        class_num, a, b, h = self.error_region[0].shape
        hints = []
        large_ccs = []  # list of [num, idx, label]
        for c in range(class_num):
            hints.append([])
            if np.sum(self.error_region[0][c, ...]) != 0:
                # large_nums = []
                labeled_cc = self.error_region[0][c, ...]
                labeled_flat = np.reshape(labeled_cc, -1)
                if len(np.bincount(labeled_flat)) > 1:
                    lcount = np.bincount(labeled_flat)[1:]
                    sort_lcount = np.sort(lcount)[::-1]
                    # print(sort_lcount)
                    argsort_lcount = np.argsort(lcount)[::-1]
                    for arg in range(len(lcount)):
                        if sort_lcount[arg] < cc_thd:
                            break
                        large_ccs.append([sort_lcount[arg], argsort_lcount[arg] + 1, c])
                        # large_nums.append(sort_lcount[arg])

        sorted_large_ccs = sorted(large_ccs, key=lambda x: x[0], reverse=True)
        click_tot = 0
        for i in range(min(len(sorted_large_ccs), max_hint_num)):
            # print("int(max_hint_num/len(sorted_large_ccs))",int(max_hint_num/len(sorted_large_ccs)))
            # print("int((max_hint_num%len(sorted_large_ccs) > i))", int((max_hint_num%len(sorted_large_ccs) > i)))
            click_num = max(1,
                            int(max_hint_num / len(sorted_large_ccs)) + int((max_hint_num % len(sorted_large_ccs) > i)))
            # print("click_num",click_num)
            click_tot += click_num
            largest_id = sorted_large_ccs[i][1]
            label = sorted_large_ccs[i][2]
            labeled_cc = self.error_region[0][label, ...]
            points = np.where(labeled_cc == largest_id)
            num_point = len(points[0])
            # print("points",points)
            # print("num_point",num_point)
            _boundary_points = []
            _inner_points = []

            for kk in range(0, num_point, 8):
                # print("kk",kk)
                xx = points[0][kk]
                yy = points[1][kk]
                zz = points[2][kk]
                if xx == 0 or xx == a - 1 or yy == 0 or yy == b - 1 or zz == 0 or zz == h - 1 \
                        or labeled_cc[xx - 1][yy][zz] != largest_id or labeled_cc[xx + 1][yy][zz] != largest_id \
                        or labeled_cc[xx][yy - 1][zz] != largest_id or labeled_cc[xx][yy + 1][zz] != largest_id \
                        or labeled_cc[xx][yy][zz - 1] != largest_id or labeled_cc[xx][yy][zz + 1] != largest_id:
                    _boundary_points.append([xx, yy, zz])
                else:
                    _inner_points.append([xx, yy, zz])
            num_bd_point = len(_boundary_points)
            num_in_point = len(_inner_points)
            # print("%d boundary points and %d inner points" % (num_bd_point, num_in_point))

            for _ in range(click_num):
                boundary_points = np.asarray(_boundary_points).reshape(1, -1, 3)
                inner_points = np.asarray(_inner_points).reshape(-1, 1, 3)
                dis_map = np.amin(np.sum(np.abs(boundary_points - inner_points), axis=2), axis=1)
                if dis_map.size:
                    chosen_id = np.argmax(dis_map)

                    rand = np.random.randint(7, size=3) - 3
                    x = min(max(inner_points[chosen_id, 0, 0] + rand[0], 0), a - 1)
                    y = min(max(inner_points[chosen_id, 0, 1] + rand[1], 0), b - 1)
                    z = min(max(inner_points[chosen_id, 0, 2] + rand[2], 0), h - 1)
                    if labeled_cc[x][y][z] != largest_id:
                        x = inner_points[chosen_id, 0, 0]
                        y = inner_points[chosen_id, 0, 1]
                        z = inner_points[chosen_id, 0, 2]

                    hints[label].append([x, y, z])
                    # if [x, y, z] not in _boundary_points:
                    #     _inner_points.remove([x, y, z])
                    #     _boundary_points.append([x, y, z])
                    if [x, y, z] not in _boundary_points:
                        if [x, y, z] in _inner_points:
                            _inner_points.remove([x, y, z])
                        _boundary_points.append([x, y, z])
                    # print("pred", self.pred[0,0,x,y,z])
                    # print("gt", self.gt[0,x,y,z])

        # print("click_tot",click_tot)

        if shape == 'gaussian':
            hint_map = make_gt_gaussian(a, b, h, hints, sigma=hint_range)
        elif shape == 'euclidean':
            hint_map = make_gt_euclidean(a, b, h, hints)
        elif shape == 'geodesic':
            hint_map = make_gt_geodesic(self.img[0, 0], hints)

        # plt.subplot(3,2,1)
        # plt.imshow(pred[:, :, 14])
        # plt.title('image')
        # plt.subplot(3,2,2)
        # plt.imshow(gt[:, :, 14])
        # plt.title('gt')
        # plt.subplot(3,2,3)
        # plt.imshow(error_region[0, :, :, 14])
        # plt.title('error0')
        # plt.subplot(3,2,4)
        # plt.imshow(error_region[1, :, :, 14])
        # plt.title('error1')
        # plt.subplot(3,2,5)
        # plt.imshow(hint_map[0, :, :, 14])
        # plt.title('hint0')
        # plt.subplot(3,2,6)
        # plt.imshow(hint_map[1, :, :, 14])
        # plt.title('hint1')

        # plt.show()
        self.hint_map = hint_map[np.newaxis]
        self.hints = hints


def make_gt_geodesic(I, hints):
    I = np.float32(I)
    hint_map = []
    for i in range(len(hints)):
        S = np.zeros_like(I, np.uint8)
        if len(hints[i]) == 0:
            D = np.random.rand(*I.shape) * 5
        else:
            for x, y, z in hints[i]:
                S[x, y, z] = 1
            D = GeodisTK.geodesic3d_raster_scan(I, S, [55, 55, 30], 0.5, 4)
            D = D / np.amax(D)
            # print("D", np.amax(D), np.amin(D), np.mean(D))
        hint_map.append(D)
    hint_map = np.stack(hint_map, axis=0)
    return hint_map


def make_gaussian(size, sigma=None, center=None, d_type=np.float64):
    """ Make a square gaussian kernel.
    size: is the dimensions of the output gaussian
    sigma: is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size[0], 1, float)
    x = np.reshape(x, (-1, 1, 1))
    y = np.arange(0, size[1], 1, float)
    y = np.reshape(y, (-1, 1))
    z = np.arange(0, size[2], 1, float)

    if center is None:
        x0 = size[0] // 2
        y0 = size[1] // 2
        z0 = size[2] // 2
    else:
        x0 = center[0]
        y0 = center[1]
        z0 = center[2]

    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2) / sigma ** 2).astype(d_type)


def make_gt_gaussian(a, b, h, hints, sigma=10):
    """ Make the ground-truth for  landmark.
    img: the original color image
    labels: label with the Gaussian center(s) [[x0, y0],[x1, y1],...]
    sigma: sigma of the Gaussian.
    one_mask_per_point: masks for each point in different channels?
    """

    # a, b, h = img.shape
    maps = []

    for c in range(len(hints)):
        hint = hints[c]
        if len(hint) == 0:
            m = np.zeros(shape=(a, b, h), dtype=np.float64)
        else:
            hint = np.array(hint)
            if hint.ndim == 1:
                hint = hint[np.newaxis]
            m = np.zeros(shape=(a, b, h), dtype=np.float64)
            for i in range(hint.shape[0]):
                m = np.maximum(m, make_gaussian((a, b, h), center=hint[i, :], sigma=sigma))

        maps.append(m)

    maps = np.stack(maps, 0)  # 2 x a x b x h
    return maps


def make_euclidean(size, center=None, d_type=np.float64):
    x = np.arange(0, size[0], 1, float)
    x = np.reshape(x, (-1, 1, 1))
    y = np.arange(0, size[1], 1, float)
    y = np.reshape(y, (-1, 1))
    z = np.arange(0, size[2], 1, float)

    if center is None:
        x0 = size[0] // 2
        y0 = size[1] // 2
        z0 = size[2] // 2
    else:
        x0 = center[0]
        y0 = center[1]
        z0 = center[2]

    euc = np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2).astype(d_type)
    return np.max(euc) - euc


def make_gt_euclidean(a, b, h, hints):
    # a, b, h = img.shape
    maps = []

    for c in range(len(hints)):
        hint = hints[c]
        if len(hint) == 0:
            m = np.zeros(shape=(a, b, h), dtype=np.float64)
        else:
            hint = np.array(hint)
            if hint.ndim == 1:
                hint = hint[np.newaxis]
            m = np.zeros(shape=(a, b, h), dtype=np.float64)
            for i in range(hint.shape[0]):
                m = np.maximum(m, make_euclidean((a, b, h), center=hint[i, :]))
            m = m / np.max(m)

        maps.append(m)

    maps = np.stack(maps, 0)  # 2 x a x b x h
    return maps
