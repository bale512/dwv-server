import numpy as np
import torch
from scipy.ndimage import zoom
import cv2
import threading
import pydicom


def Box_Crop(img, crop):
    a, b, h = img.shape
    xmin = crop['xmin'] - extend
    xmax = crop['xmax'] + extend
    ymin = crop['ymin'] - extend
    ymax = crop['ymax'] + extend
    zmin = crop['zmin'] - extend
    zmax = crop['zmax'] + extend
    cropped_img = img[max(xmin, 0): min(xmax, a),
                      max(ymin, 0): min(ymax, b),
                      max(zmin, 0): min(zmax, h)]
    return cropped_img


def Zscore_Normal(img):
    norm_img = (img - np.mean(img)) / np.std(img)
    return norm_img


def Resize(img, size):
    a, b, h = img.shape
    scale = (size[0] / a, size[1] / b, size[2] / h)
    resize_img = zoom(img, scale, order=3)
    return resize_img


def ToTensor(img):
    if len(img.shape) == 3:
        image = img[np.newaxis, np.newaxis, :, :, :]
    elif len(img.shape) == 4:
        image = img[np.newaxis, :, :, :]
    else:
        image = img
    image = torch.from_numpy(image)
    return image


def Padding(img, size, img_org_size):
    img = np.pad(img, ((max(size[0], 0), img_org_size[0] - size[1]),
                       (max(size[2], 0), img_org_size[1] - size[3]),
                       (max(size[4], 0), img_org_size[2] - size[5])))
    return img


def Draw(ds, pred, color=1):
    contours, _ = cv2.findContours(np.asarray(pred * 255).astype(np.uint8), cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_NONE)
    img = ds.pixel_array
    maxpixel = np.max(img)
    for contour in contours:
        pos = contour[:, 0, :]
        pos_x = pos[:, 1]
        pos_y = pos[:, 0]
        img[(pos_x, pos_y)] = maxpixel * color
    ds.PixelData = img.tobytes()
    return ds

# 多线程读图


class ReadDcmThread(threading.Thread):
    def __init__(self, threadID, name, counter, path):
        threading.Thread.__init__(self)
        self.path = path
        self.counter = counter
        self.name = name
        self.threadID = threadID

    def run(self):
        self.ds = pydicom.dcmread(self.path)

    def get_result(self):
        threading.Thread.join(self)
        try:
            return self.ds
        except:
            return None


# 多线程保存结果
class SaveDcmThread(threading.Thread):
    def __init__(self, threadID, ds, path, pred, pred_high=None, pred_middle=None):
        threading.Thread.__init__(self)
        self.ds = ds
        self.path = path
        self.pred = pred
        self.threadID = threadID
        self.pred_high = pred_high
        self.pred_middle = pred_middle

    def run(self):
        self.ds = Draw(self.ds, self.pred)
        if self.pred_high is not None:
            self.ds = Draw(self.ds, self.pred_high, 0)
        if self.pred_middle is not None:
            self.ds = Draw(self.ds, self.pred_middle, 0.5)
        self.ds.save_as(self.path)
