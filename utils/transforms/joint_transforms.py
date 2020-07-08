# @Time    : 2020/7/8
# @Author  : Lart Pang
# @Email   : lartpang@163.com
# @File    : joint_transforms.py
# @Project : HDFNet
# @GitHub  : https://github.com/lartpang
import random

import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision.transforms import transforms


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        assert img.size == mask.size
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class RandomHorizontallyFlip(object):
    def __call__(self, img, mask):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask


class JointResize(object):
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        elif isinstance(size, tuple):
            self.size = size
        else:
            raise RuntimeError("size参数请设置为int或者tuple")

    def __call__(self, img, mask):
        img = img.resize(self.size)
        mask = mask.resize(self.size)
        return img, mask


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return img.rotate(rotate_degree, Image.BILINEAR), mask.rotate(rotate_degree, Image.NEAREST)


class RandomScaleCrop(object):
    def __init__(self, input_size, scale_factor):
        """
        处理的是长宽相同的图像。这里会进行扩张到原图的随机倍数（1～scale_factor），
        之后进行随机裁剪，得到输入图像大小。
        """
        self.input_size = input_size
        self.scale_factor = scale_factor

    def __call__(self, img, mask):
        # random scale (short edge)
        assert img.size[0] == self.input_size

        o_size = random.randint(int(self.input_size * 1), int(self.input_size * self.scale_factor))
        img = img.resize((o_size, o_size), Image.BILINEAR)
        mask = mask.resize((o_size, o_size), Image.NEAREST)  # mask的放缩使用的是近邻差值

        # random crop input_size
        x1 = random.randint(0, o_size - self.input_size)
        y1 = random.randint(0, o_size - self.input_size)
        img = img.crop((x1, y1, x1 + self.input_size, y1 + self.input_size))
        mask = mask.crop((x1, y1, x1 + self.input_size, y1 + self.input_size))

        return img, mask


class ScaleCenterCrop(object):
    def __init__(self, input_size):
        self.input_size = input_size

    def __call__(self, img, mask):
        w, h = img.size
        # 让短边等于剪裁的尺寸
        if w > h:
            oh = self.input_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.input_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        # 从放缩后的结果中进行中心剪裁
        w, h = img.size
        x1 = int(round((w - self.input_size) / 2.0))
        y1 = int(round((h - self.input_size) / 2.0))
        img = img.crop((x1, y1, x1 + self.input_size, y1 + self.input_size))
        mask = mask.crop((x1, y1, x1 + self.input_size, y1 + self.input_size))

        return img, mask


class RandomGaussianBlur(object):
    def __call__(self, img, mask):
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))

        return img, mask


class PaddingSquare(object):
    """
    先通过在原图的四边进行padding变成正方形，再利用resize放缩到网络需要的输入尺寸（方形）
    """

    def __init__(self, pad_size):
        assert isinstance(pad_size, int)
        self.pad_size = (pad_size, pad_size)

    def __call__(self, x, y):
        # array和asarray都可以将结构数据转化为ndarray
        # 但是主要区别就是当数据源是ndarray时，array仍然会copy出一个副本，占用新的内存，但asarray不会，仍会受原始的数据的影响。
        x = np.asarray(x)
        y = np.asarray(y)

        h, w, c = x.shape
        size = max(h, w)
        pad_h = (size - h) // 2
        pad_w = (size - w) // 2
        temp_x = np.zeros((size, size, c))
        temp_y = np.zeros((size, size))
        temp_x[pad_h : h + pad_h, pad_w : w + pad_w, :] = x
        temp_y[pad_h : h + pad_h, pad_w : w + pad_w] = y

        temp_x = Image.fromarray(temp_x.astype(np.uint8)).resize(self.pad_size)
        temp_y = Image.fromarray(temp_y.astype(np.uint8)).resize(self.pad_size)
        return temp_x, temp_y


if __name__ == "__main__":
    a = torch.randn((3, 320, 320))
    b = torch.randn((1, 320, 320))
    to_pil = transforms.ToPILImage()

    a = to_pil(a)
    b = to_pil(b)
    lu_x, lu_y, rb_x, rb_y = (320, 320, 544, 544)
    b.crop((lu_x, lu_y, rb_x, rb_y))
    print(np.asarray(b.crop((lu_x, lu_y, rb_x, rb_y))))
