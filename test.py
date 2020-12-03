# -*- coding: utf-8 -*-
# @Time    : 2020/7/5
# @Author  : Lart Pang
# @Email   : lartpang@163.com
# @File    : test.py
# @Project : HDFNet
# @GitHub  : https://github.com/lartpang
import argparse
import os
import os.path as osp
from datetime import datetime
from distutils.util import strtobool

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

import network
from utils.metric import CalTotalMetric
from utils.misc import check_dir_path_valid

my_parser = argparse.ArgumentParser(
    prog="main script",
    description="The code is created by lartpang (Youwei Pang).",
    epilog="Enjoy the program! :)",
    allow_abbrev=False,
)
my_parser.add_argument("--param_path", required=True, type=str, help="自定义参数文件路径")
my_parser.add_argument("--model", required=True, type=str, help="选择使用的模型的名字，请把对应的模型类导入到network文件夹中的`__init__.py`文件中")
my_parser.add_argument("--testset", required=True, type=str, help="测试集路径，该路径下至少包含两个文件夹: Image, Depth")
# https://stackoverflow.com/a/46951029
my_parser.add_argument(
    "--has_masks",
    default=False,
    type=lambda x: bool(strtobool(str(x))),
    help="是否存在对应的Mask数据，即`--testset`指定的路径下是否包含存放有mask文件的Mask文件夹",
)
my_parser.add_argument("--save_pre", default=False, type=lambda x: bool(strtobool(str(x))), help="是否保存测试生成的结果")
my_parser.add_argument("--save_path", default="", type=str, help="保存测试结果的路径")
my_parser.add_argument(
    "--data_mode", default="RGBD", choices=["RGB", "RGBD"], type=str, help="测试的是RGB数据还是RGBD数据，注意请选择使用对应任务的模型"
)
my_parser.add_argument("--use_gpu", default=True, type=lambda x: bool(strtobool(str(x))), help="测试是否使用GPU")
my_args = my_parser.parse_args()


class Tester:
    def __init__(self, args):
        if args.use_gpu and torch.cuda.is_available():
            self.dev = torch.device("cuda:0")
        else:
            self.dev = torch.device("cpu")

        self.to_pil = transforms.ToPILImage()
        self.data_mode = args.data_mode
        self.model_name = args.model

        self.te_data_path = args.testset
        self.image_dir = os.path.join(self.te_data_path, "Image")
        if self.data_mode == "RGBD":
            self.depth_dir = os.path.join(self.te_data_path, "Depth")
        else:
            self.depth_dir = ""

        self.has_masks = args.has_masks
        if self.has_masks:
            self.mask_dir = os.path.join(self.te_data_path, "Mask")
        else:
            self.mask_dir = ""
        check_dir_path_valid([self.te_data_path, self.image_dir, self.mask_dir])

        self.save_pre = args.save_pre
        if self.save_pre:
            self.save_path = args.save_path
            if not os.path.exists(self.save_path):
                print(f" ==>> {self.save_path} 不存在, 这里创建一个 <<==")
                os.makedirs(self.save_path)

        self.net = getattr(network, self.model_name)(pretrained=False).to(self.dev)
        self.resume_checkpoint(load_path=args.param_path)
        self.net.eval()

        self.rgb_transform = transforms.Compose(
            [
                transforms.Resize((320, 320), interpolation=Image.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        if self.data_mode == "RGBD":
            self.depth_transform = transforms.Compose(
                [transforms.Resize((320, 320), interpolation=Image.BILINEAR), transforms.ToTensor()]
            )

    def test(self):
        rgb_name_list = os.listdir(self.image_dir)

        cal_total_metrics = CalTotalMetric(num=len(rgb_name_list), beta_for_wfm=1)

        tqdm_iter = tqdm(enumerate(rgb_name_list), total=len(rgb_name_list), leave=False)
        for idx, rgb_name in tqdm_iter:
            tqdm_iter.set_description(f"{self.model_name}:" f"te=>{idx + 1}")

            depth_mask_name = rgb_name[:-4] + ".png"

            rgb_path = os.path.join(self.image_dir, rgb_name)
            rgb_pil = Image.open(rgb_path).convert("RGB")

            original_size = rgb_pil.size

            rgb_tensor = self.rgb_transform(rgb_pil).unsqueeze(0)
            rgb_tensor = rgb_tensor.to(self.dev, non_blocking=True)

            if self.data_mode == "RGBD":
                depth_path = os.path.join(self.depth_dir, depth_mask_name)
                depth_pil = Image.open(depth_path).convert("L")

                depth_tensor = self.depth_transform(depth_pil).unsqueeze(0)
                depth_tensor = depth_tensor.to(self.dev, non_blocking=True)

                with torch.no_grad():
                    pred_tensor = self.net(rgb_tensor, depth_tensor)
            else:
                with torch.no_grad():
                    pred_tensor = self.net(rgb_tensor)

            pred_tensor = pred_tensor.squeeze(0).cpu().detach()

            pred_pil = self.to_pil(pred_tensor).resize(original_size, resample=Image.NEAREST)
            if self.save_pre:
                pred_pil.save(osp.join(self.save_path, depth_mask_name))

            if self.has_masks:
                pred_array = np.asarray(pred_pil)
                max_pred_array = pred_array.max()
                min_pred_array = pred_array.min()
                if max_pred_array == min_pred_array:
                    pred_array = pred_array / 255
                else:
                    pred_array = (pred_array - min_pred_array) / (max_pred_array - min_pred_array)

                mask_path = os.path.join(self.mask_dir, depth_mask_name)
                mask_pil = Image.open(mask_path).convert("L")
                mask_array = np.asarray(mask_pil)
                mask_array = mask_array / (mask_array.max() + 1e-8)
                mask_array = np.where(mask_array > 0.5, 1, 0)

                cal_total_metrics.update(pred_array, mask_array)

        if self.has_masks:
            results = cal_total_metrics.show()
            fixed_pre_results = {k: f"{v:.3f}" for k, v in results.items()}
            print(f" ==>> 在{self.te_data_path}上的测试结果\n >> {fixed_pre_results}")

    def resume_checkpoint(self, load_path):
        """
        从保存节点恢复模型

        Args:
            load_path (str): 模型存放路径
        """
        if os.path.exists(load_path) and os.path.isfile(load_path):
            print(f" =>> loading checkpoint '{load_path}' <<== ")
            checkpoint = torch.load(load_path, map_location=self.dev)
            self.net.load_state_dict(checkpoint)
            print(f" ==> loaded checkpoint '{load_path}' " f"(only has the net's weight params) <<== ")
        else:
            raise Exception(f"{load_path}路径不正常，请检查")


if __name__ == "__main__":
    # 保存备份数据 ###########################################################
    print(f" ===========>> {datetime.now()}: 初始化开始 <<=========== ")
    init_start = datetime.now()
    tester = Tester(args=my_args)
    print(f" ==>> 初始化完毕，用时：{datetime.now() - init_start} <<== ")

    # 训练模型 ###############################################################
    print(f" ===========>> {datetime.now()}: 开始测试 <<=========== ")
    tester.test()
    print(f" ===========>> {datetime.now()}: 结束测试 <<=========== ")
