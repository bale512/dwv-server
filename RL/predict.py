from model.pnet import PNet
from model.rnet import RNet
from model.unet import UNet3D
from transform import *
from torch.nn.functional import interpolate
from pixelwise_a3c import PixelWiseA3C
import torch
import torch.nn.functional as F
import numpy as np
import chainerrl
import chainer
import os
import SimpleITK as sitk
import pydicom
import json
import State
import cv2
import time
import SimpleITK as sitk

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Predict(object):
    """
    参数说明：
    dcm_list_path: dcm序列的文件夹路径,dir
    output_path: 输出的dcm文件夹路径,dir
    information: 粗分割概率及分割框的路径,dir
    hints_position: 输入hints的三维坐标,list,eg:[[[1,2,3],[4,5,6]],[[4,5,7],[5,5,5]]]
    first_model_path: 初分割模型路径
    second_model_path: 二阶段模型路径
    """

    def __init__(self,
                 first_model="PNet",
                 second_model="RNet",
                 n_classes=2,
                 first_model_path=r"/home/jiayoutao/mailrlseg/segmention/rlseg/hsresult/pnet_global_hs_flair/pnet_global_hs_flair_4137.pth",
                 second_model_path=r"/home/jiayoutao/mailrlseg/segmention/rlseg/hsresult/r6prob_hs_train13test1/r6prob_hs_train13test1_1963/",
                 crop_model="PNet",
                 model_with_crop_path=r"/home/jiayoutao/mailrlseg/segmention/rlseg/hsresult/dl_pnet_hs/dl_pnet_hs_2099.pth"):
        self.n_classes = n_classes  # 最终目标是几分类
        self.size = (55, 55, 30)  # 模型接受图像大小
        self.dcm_list_path = None  # 原始dcm序列的路径
        self.output_path = None  # 输出dcm序列的路径
        self.information_path = None  # crop和prob信息存储位置
        self.hints_position = None  # 交互点的位置
        self.dslist = []  # dcm序列
        self.image = None  # 原始图像 eg:240*240*176
        self.crop = {}  # 裁剪的六个坐标
        self.prob = None  # 初分割概率 eg:1*1*240
        self.img = None  # 预处理完后的图像 eg:1*1*55*55*30
        # RL参数
        self.EPISODE_LEN = 5
        self.GAMMA = 0.95  # reward discount factor
        self.LEARNING_RATE = 1e-3
        self.agent = None
        self.extend = 15

        # 加载模型
        self.model1 = self.get_model(first_model, first_model_path)
        self.model2 = self.get_model(second_model, second_model_path)
        self.model_with_crop = self.get_model(crop_model, model_with_crop_path)

    # 清空数据缓存
    def clear(self):
        self.dcm_list_path = None
        self.output_path = None
        self.information_path = None
        self.hints_position = None
        self.dslist = []
        self.image = None
        self.crop = {}
        self.prob = None
        self.img = None

    # 根据模型名加载模型
    def get_model(self, model_name, model_path):
        if model_name == "PNet":
            model = PNet(output_chn=self.n_classes)
            model = model.cuda()
            # cudnn.benchmark = True

            # load model
            model.load_state_dict(
                torch.load(model_path))
            print("PNet has been loaded")

        elif model_name == "RNet":
            # load model
            model = RNet(pretrained=False)
            # _/_/_/ setup _/_/_/
            optimizer = chainer.optimizers.Adam(alpha=self.LEARNING_RATE)
            optimizer.setup(model)
            agent = PixelWiseA3C(
                model, optimizer, self.EPISODE_LEN, self.GAMMA)
            agent.model.to_gpu()
            agent.act_deterministically = True
            agent.load(model_path)
            self.agent = agent
            print("RNet has been loaded")
        else:
            model = UNet3D(1, [32, 48, 64, 96, 128], 2).cuda()
            model.load_state_dict(
                torch.load(model_path))
            print("PNet has been loaded")
        return model

    # 根据路径加载原始图像列表self.dslist以及图像self.image
    def get_image(self, path):
        file_name_list = os.listdir(path)
        self.dslist = []
        sample = pydicom.dcmread(os.path.join(
            path, file_name_list[0])).pixel_array
        w = sample.shape[0]
        h = sample.shape[1]
        d = len(file_name_list)

        '''
        for file_name in file_name_list:
            ds = pydicom.dcmread(os.path.join(path, file_name))
            w, h = ds.pixel_array.shape
            self.dslist.append(ds)
        '''
        threads = []
        for i, file_name in enumerate(file_name_list):
            t = ReadDcmThread(i, "ReadDcm%d" %
                              i, i, os.path.join(path, file_name))
            t.start()
            threads.append(t)

        self.dslist = [t.get_result() for t in threads]

        self.dslist = sorted(
            self.dslist, key=lambda x: float(x[0x0020, 0x1041].value))
        self.image = np.zeros((w, h, d))
        for i, ds in enumerate(self.dslist):
            self.image[:, :, i] = ds.pixel_array

    # mode 为第一阶段或者第二阶段的数据预处理
    def transform_img(self, img):
        img = interpolate(torch.from_numpy(img[np.newaxis, np.newaxis]), self.size, mode="trilinear",
                          align_corners=True).numpy()[0, 0]
        img = Zscore_Normal(img)
        img = ToTensor(img)
        return img

    # hints是原始的交互坐标，size是裁剪框的坐标
    def hintsolution(self, hints, size, img_shape):
        a, b, depth = img_shape
        hints_new = []
        if (len(hints[0]) != 0):
            hint_false = np.array(hints[0])
            hint_item = hint_false[:, 0].copy()
            hint_false[:, 0] = hint_false[:, 1]
            hint_false[:, 1] = hint_item
            hint_false[:, 0] = np.minimum(np.maximum((hint_false[:, 0] - size[0]), 0),
                                          size[1] - size[0] - 1) * self.size[0] / a
            hint_false[:, 1] = np.minimum(np.maximum((hint_false[:, 1] - size[2]), 0),
                                          size[3] - size[2] - 1) * self.size[1] / b
            hint_false[:, 2] = np.minimum(np.maximum((hint_false[:, 2] - size[4]), 0),
                                          size[5] - size[4] - 1) * self.size[2] / depth
            hints_new.append(hint_false.tolist())
        else:
            hints_new.append([])
        if (len(hints[1]) != 0):
            hint_true = np.array(hints[1])
            hint_item = hint_true[:, 0].copy()
            hint_true[:, 0] = hint_true[:, 1]
            hint_true[:, 1] = hint_item
            hint_true[:, 0] = np.minimum(np.maximum((hint_true[:, 0] - size[0]), 0),
                                         size[1] - size[0] - 1) * self.size[0] / a
            hint_true[:, 1] = np.minimum(np.maximum((hint_true[:, 1] - size[2]), 0),
                                         size[3] - size[2] - 1) * self.size[1] / b
            hint_true[:, 2] = np.minimum(np.maximum((hint_true[:, 2] - size[4]), 0),
                                         size[5] - size[4] - 1) * self.size[2] / depth
            hints_new.append(hint_true.tolist())
        else:
            hints_new.append([])
        return hints_new

    # 第一次全图分割，获得概率及边界框
    def first_model_inference(self, dcm_list_path, information_path):
        self.dcm_list_path = dcm_list_path
        self.information_path = information_path
        self.get_image(dcm_list_path)

        self.crop_path = os.path.join(self.information_path, "crop.json")
        self.prob_path = os.path.join(self.information_path, "prob.npy")
        # 根据路径确定是否之前对其进行过分割，如果分割过了则有crop记录和粗分割概率prob
        if not os.path.exists(information_path):
            os.mkdir(information_path)
        if os.path.exists(self.crop_path) and os.path.exists(self.prob_path):
            with open(self.crop_path, "r") as f:
                crop = json.load(f)
            prob = np.load(self.prob_path)
            self.crop = crop
            self.img = self.image[crop["xmin"]:crop["xmax"],
                                  crop["ymin"]:crop["ymax"], crop["zmin"]:crop["zmax"]]
            prob = prob[self.crop["xmin"]:self.crop["xmax"], self.crop["ymin"]:self.crop["ymax"],
                        self.crop["zmin"]:self.crop["zmax"]]
            self.prob = prob[np.newaxis, np.newaxis, :, :, :]
            self.prob = interpolate(torch.from_numpy(self.prob), self.size, mode="trilinear",
                                    align_corners=True).numpy()
        else:
            img = self.transform_img(self.image)
            with torch.no_grad():
                output = self.model1(img.to(torch.float).cuda())
                output = interpolate(output, size=self.image.shape, mode='trilinear',
                                     align_corners=True)
                output = output[:, 1, :, :, :]
                prob = output.cpu().numpy()[0]
                posx, posy, posz = np.where(prob > 0)
                w, h, d = self.image.shape

                self.crop["xmin"] = (max(np.min(posx) - self.extend, 0))
                self.crop["xmax"] = (min(np.max(posx) + self.extend, w - 1))
                self.crop["ymin"] = (max(np.min(posy) - self.extend, 0))
                self.crop["ymax"] = (min(np.max(posy) + self.extend, h - 1))
                self.crop["zmin"] = (max(np.min(posz) - self.extend, 0))
                self.crop["zmax"] = (min(np.max(posz) + self.extend, d - 1))
                with open(self.crop_path, "w") as f:
                    json.dump(self.crop, f)

                prob = prob[self.crop["xmin"]:self.crop["xmax"], self.crop["ymin"]:self.crop["ymax"],
                            self.crop["zmin"]:self.crop["zmax"]]
                self.prob = prob[np.newaxis, np.newaxis, :, :, :]
                self.prob = interpolate(torch.from_numpy(self.prob), self.size, mode="trilinear",
                                        align_corners=True).numpy()
                self.img = self.image[self.crop["xmin"]:self.crop["xmax"], self.crop["ymin"]:self.crop["ymax"],
                                      self.crop["zmin"]:self.crop["zmax"]]

    # 有crop情况下的推断
    def predict_with_crop(self, dcm_list_path, information_path, output_path):
        # 获取crop信息
        self.dcm_list_path = dcm_list_path
        self.information_path = information_path
        self.get_image(dcm_list_path)

        self.crop_path = os.path.join(self.information_path, "crop.json")
        self.prob_path = os.path.join(self.information_path, "prob.npy")
        with open(self.crop_path, "r") as f:
            crop = json.load(f)
        self.crop = crop
        crop_pos = [self.crop['xmin'], self.crop['xmax'], self.crop['ymin'], self.crop['ymax'], self.crop['zmin'],
                    self.crop['zmax']]
        # 获取crop后的图像
        self.img = self.image[crop["xmin"]:crop["xmax"],
                              crop["ymin"]:crop["ymax"], crop["zmin"]:crop["zmax"]]
        input = Zscore_Normal(self.img)
        input = interpolate(torch.from_numpy(input[np.newaxis, np.newaxis, :, :, :]), self.size, mode="trilinear",
                            align_corners=True).float().cuda()
        # 模型推断
        with torch.no_grad():
            output = self.model_with_crop(input)
        # 图像复原
        output = interpolate(output, self.img.shape,
                             mode='trilinear', align_corners=True).cpu()
        pred = Padding(output.numpy()[0, 1], crop_pos, self.image.shape)
        # 0.85概率
        pred_high = (pred > 0.85).astype(np.float)
        # 0.5概率
        pred = (pred > 0.5).astype(np.float)
        # 多线程保存
        k = len(self.dslist)
        save_thread_list = []
        for i in range(k):
            save_path = os.path.join(
                output_path, self.dslist[i].filename.split('/')[-1])
            t = SaveDcmThread(
                i, self.dslist[i], save_path, pred[:, :, i], pred_high[:, :, i])
            t.start()
            save_thread_list.append(t)
        # 等待所有线程执行完毕
        for t in save_thread_list:
            t.join()
        print("ok")

    # 带交互的局部分割
    def second_model_inference(self, dcm_list_path, output_path, information_path, hints_position, load=False):
        '''
        :param dcm_list_path: path of dicom files
        :param output_path: path of output dicom files
        :param information_path: path of informations files,crop.json,prob.npy
        :param hints_position: given hints ,like [[[1,1,3],[4,5,6]],[[4,5,5]]]
        :param load: whether the class need to reload the image data
        :return:
        '''
        self.output_path = output_path
        self.hints_position = hints_position
        # 是否需要加载数据
        if load:
            self.dcm_list_path = dcm_list_path
            self.information_path = information_path
            self.get_image(dcm_list_path)
        # 如果第一步没有执行完，则返回超时
        while self.prob is None:
            if time.time() - t1 > 5:
                return "please wait for the first stage complete!"

        crop_pos = [self.crop['xmin'], self.crop['xmax'], self.crop['ymin'], self.crop['ymax'], self.crop['zmin'],
                    self.crop['zmax']]
        hints = self.hintsolution(hints_position, crop_pos, self.img.shape)

        input = torch.from_numpy(self.img[np.newaxis, np.newaxis])
        input = interpolate(input, self.size, mode="trilinear",
                            align_corners=True)  # 0.04s
        input = input.numpy()

        current_state = State.State(input.shape, click_mode='force')
        current_state.reset_predict(input, self.prob, hints)

        action = self.agent.act(current_state.state)
        current_state.step_predict(action)
        # 概率映射回原始尺寸
        prob = torch.from_numpy(current_state.prob)
        prob = interpolate(prob, self.img.shape,
                           mode='trilinear', align_corners=True)
        prob = Padding(prob.numpy()[0, 0], crop_pos, self.image.shape)
        # 保存概率结果
        np.save(self.prob_path, prob)
        # 生成预测
        pred = torch.from_numpy(current_state.pred)
        pred = interpolate(pred, self.img.shape,
                           mode='trilinear', align_corners=True)
        pred = Padding(pred.numpy()[0, 0], crop_pos, self.image.shape)
        pred = (pred > 0.5).astype(np.float)
        return_list = []
        k = len(self.dslist)
        save_thread_list = []
        for i in range(k):
            save_path = os.path.join(
                output_path, self.dslist[i].filename.split('/')[-1])
            t = SaveDcmThread(
                i, self.dslist[i], save_path, pred[:, :, k - 1 - i])
            t.start()
            save_thread_list.append(t)
            return_list.append(save_path)
        # 等待所有线程执行完毕
        for t in save_thread_list:
            t.join()
        return return_list


ppp = Predict(model_with_crop_path=r"/home/jiayoutao/mailrlseg/segmention/rlseg/hsresult/pnet_high_level_enhanced/pnet_high_level_enhanced_499.pth")
# ppp.clear()
# ppp.first_model_inference(dcm_list_path="3d flair", information_path="information")
# ppp.second_model_inference("3d flair", "output", "information", [[[4, 4, 6]], []])
# ppp.predict_with_crop(dcm_list_path="liyulong_12277731_flair_t1+c", information_path="liyulong_t1c_information",
#                       output_path="output")
