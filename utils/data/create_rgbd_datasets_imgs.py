# @Time    : 2020/7/8
# @Author  : Lart Pang
# @Email   : lartpang@163.com
# @File    : create_rgbd_datasets_imgs.py
# @Project : HDFNet
# @GitHub  : https://github.com/lartpang
import os

from PIL import Image
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from utils.transforms.triple_transforms import Compose, JointResize, RandomHorizontallyFlip, RandomRotate


def _read_list_from_file(list_filepath):
    img_list = []
    with open(list_filepath, mode="r", encoding="utf-8") as openedfile:
        line = openedfile.readline()
        while line:
            img_list.append(line.split()[0])
            line = openedfile.readline()
    return img_list


def _make_test_dataset(root, prefix=(".jpg", ".png")):
    img_path = os.path.join(root, "Image")
    mask_path = os.path.join(root, "Mask")
    depth_path = os.path.join(root, "Depth")
    img_list = [os.path.splitext(f)[0] for f in os.listdir(mask_path) if f.endswith(prefix[1])]
    return [
        (
            os.path.join(img_path, img_name + prefix[0]),
            os.path.join(mask_path, img_name + prefix[1]),
            os.path.join(depth_path, img_name + prefix[1]),
        )
        for img_name in img_list
    ]


def _make_test_dataset_from_list(list_filepath, prefix=(".jpg", ".png")):
    img_list = _read_list_from_file(list_filepath)
    return [
        (
            os.path.join(os.path.join(os.path.dirname(img_path), "Image"), os.path.basename(img_path) + prefix[0]),
            os.path.join(os.path.join(os.path.dirname(img_path), "Mask"), os.path.basename(img_path) + prefix[1]),
            os.path.join(os.path.join(os.path.dirname(img_path), "Depth"), os.path.basename(img_path) + prefix[1]),
        )
        for img_path in img_list
    ]


class TestImageFolder(Dataset):
    def __init__(self, root, in_size, prefix):
        if os.path.isdir(root):
            print(f" ==>> {root}是图片文件夹, 将会遍历其中的图片进行测试 <<==")
            self.imgs = _make_test_dataset(root, prefix=prefix)
        elif os.path.isfile(root):
            print(f" ==>> {root}是图片地址列表, 将会遍历对应的图片进行测试 <<==")
            self.imgs = _make_test_dataset_from_list(root, prefix=prefix)
        else:
            raise NotImplementedError
        self.test_img_trainsform = transforms.Compose(
            [
                # 输入的如果是一个tuple，则按照数据缩放，但是如果是一个数字，则按比例缩放到短边等于该值
                transforms.Resize((in_size, in_size), interpolation=Image.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.test_depth_transform = transforms.Compose(
            [transforms.Resize((in_size, in_size), interpolation=Image.BILINEAR), transforms.ToTensor(),]
        )

    def __getitem__(self, index):
        img_path, mask_path, depth_path = self.imgs[index]

        img = Image.open(img_path).convert("RGB")
        depth = Image.open(depth_path).convert("L")
        if img.size != depth.size:
            depth = depth.resize(img.size, resample=Image.BILINEAR)
        img_name = (img_path.split(os.sep)[-1]).split(".")[0]

        img = self.test_img_trainsform(img)
        depth = self.test_depth_transform(depth)
        return img, img_name, mask_path, depth

    def __len__(self):
        return len(self.imgs)


def _make_train_dataset(root, prefix=(".jpg", ".png")):
    img_path = os.path.join(root, "Image")
    mask_path = os.path.join(root, "Mask")
    depth_path = os.path.join(root, "Depth")
    img_list = [os.path.splitext(f)[0] for f in os.listdir(mask_path) if f.endswith(prefix[1])]
    return [
        (
            os.path.join(img_path, img_name + prefix[0]),
            os.path.join(mask_path, img_name + prefix[1]),
            os.path.join(depth_path, img_name + prefix[1]),
        )
        for img_name in img_list
    ]


def _make_train_dataset_from_list(list_filepath, prefix=(".jpg", ".png")):
    # list_filepath = '/home/lart/Datasets/RGBDSaliency/FinalSet/rgbd_train_jw.lst'
    img_list = _read_list_from_file(list_filepath)
    return [
        (
            os.path.join(os.path.join(os.path.dirname(img_path), "Image"), os.path.basename(img_path) + prefix[0]),
            os.path.join(os.path.join(os.path.dirname(img_path), "Mask"), os.path.basename(img_path) + prefix[1]),
            os.path.join(os.path.join(os.path.dirname(img_path), "Depth"), os.path.basename(img_path) + prefix[1]),
        )
        for img_path in img_list
    ]


class TrainImageFolder(Dataset):
    def __init__(self, root, in_size, prefix):
        if os.path.isdir(root):
            print(f" ==>> {root}是图片文件夹, 将会遍历其中的图片进行训练 <<==")
            self.imgs = _make_train_dataset(root, prefix=prefix)
        elif os.path.isfile(root):
            print(f" ==>> {root}是图片地址列表, 将会遍历对应的图片进行训练 <<==")
            self.imgs = _make_train_dataset_from_list(root, prefix=prefix)
        else:
            raise NotImplementedError
        self.train_triple_transform = Compose([JointResize(in_size), RandomHorizontallyFlip(), RandomRotate(10)])
        self.train_img_transform = transforms.Compose(
            [
                transforms.ColorJitter(0.1, 0.1, 0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # 处理的是Tensor
            ]
        )
        self.train_mask_transform = transforms.ToTensor()
        self.train_depth_transform = transforms.ToTensor()

    def __getitem__(self, index):
        img_path, mask_path, depth_path = self.imgs[index]

        img = Image.open(img_path)
        mask = Image.open(mask_path)
        depth = Image.open(depth_path)
        if len(img.split()) != 3:
            img = img.convert("RGB")
        if len(mask.split()) == 3:
            mask = mask.convert("L")
        if len(depth.split()) == 3:
            depth = depth.convert("L")

        if img.size != depth.size:
            depth = depth.resize(img.size, resample=Image.BILINEAR)

        img, mask, depth = self.train_triple_transform(img, mask, depth)
        mask = self.train_mask_transform(mask)
        depth = self.train_depth_transform(depth)
        img = self.train_img_transform(img)

        img_name = (img_path.split(os.sep)[-1]).split(".")[0]

        return img, mask, img_name, depth

    def __len__(self):
        return len(self.imgs)


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super(DataLoaderX, self).__iter__())


if __name__ == "__main__":
    img_list = _make_train_dataset_from_list()
    print(len(img_list))
