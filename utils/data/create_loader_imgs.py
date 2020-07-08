# @Time    : 2020/7/8
# @Author  : Lart Pang
# @Email   : lartpang@163.com
# @File    : create_loader_imgs.py
# @Project : HDFNet
# @GitHub  : https://github.com/lartpang
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader

from config import arg_config


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super(DataLoaderX, self).__iter__())


def _make_loader(dataset, shuffle=True, drop_last=False):
    return DataLoaderX(
        dataset=dataset,
        batch_size=arg_config["batch_size"],
        num_workers=arg_config["num_workers"],
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=True,
    )


def create_loader(data_path, mode, get_length=False, data_mode="RGBD", prefix=(".jpg", ".png")):
    if data_mode == "RGB":
        from utils.data.create_rgb_datasets_imgs import TestImageFolder, TrainImageFolder
    elif data_mode == "RGBD":
        from utils.data.create_rgbd_datasets_imgs import TestImageFolder, TrainImageFolder
    else:
        raise NotImplementedError

    if mode == "train":
        print(f" ==>> 使用训练集{data_path}训练 <<== ")
        train_set = TrainImageFolder(data_path, in_size=arg_config["input_size"], prefix=prefix)
        loader = _make_loader(train_set, shuffle=True, drop_last=True)
        length_of_dataset = len(train_set)
    elif mode == "test":
        print(f" ==>> 使用测试集{data_path}测试 <<== ")
        test_set = TestImageFolder(data_path, in_size=arg_config["input_size"], prefix=prefix)
        loader = _make_loader(test_set, shuffle=False, drop_last=False)
        length_of_dataset = len(test_set)
    else:
        raise NotImplementedError

    if get_length:
        return loader, length_of_dataset
    else:
        return loader
