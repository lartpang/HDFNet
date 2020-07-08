# @Time    : 2020/7/8
# @Author  : Lart Pang
# @Email   : lartpang@163.com
# @File    : triple_transforms.py
# @Project : HDFNet
# @GitHub  : https://github.com/lartpang
import random

from PIL import Image


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask, depth):
        assert img.size == mask.size
        assert img.size == depth.size
        for t in self.transforms:
            img, mask, depth = t(img, mask, depth)
        return img, mask, depth


class RandomHorizontallyFlip(object):
    def __call__(self, img, mask, depth):
        if random.random() < 0.5:
            return (
                img.transpose(Image.FLIP_LEFT_RIGHT),
                mask.transpose(Image.FLIP_LEFT_RIGHT),
                depth.transpose(Image.FLIP_LEFT_RIGHT),
            )
        return img, mask, depth


class JointResize(object):
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        elif isinstance(size, tuple):
            self.size = size
        else:
            raise RuntimeError("size参数请设置为int或者tuple")

    def __call__(self, img, mask, depth):
        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)
        depth = depth.resize(self.size, Image.BILINEAR)
        return img, mask, depth


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask, depth):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return (
            img.rotate(rotate_degree, Image.BILINEAR),
            mask.rotate(rotate_degree, Image.NEAREST),
            depth.rotate(rotate_degree, Image.BILINEAR),
        )
