# -*- coding: utf-8 -*-
# @Time    : 2020/7/5
# @Author  : Lart Pang
# @Email   : lartpang@163.com
# @File    : main.py
# @Project : HDFNet
# @GitHub  : https://github.com/lartpang
import os
import os.path as osp
import shutil
from datetime import datetime
from pprint import pprint

import numpy as np
import torch
import torch.backends.cudnn as torchcudnn
from PIL import Image
from torch.nn import BCELoss
from torch.optim import SGD
from torchvision import transforms
from tqdm import tqdm
import network
from loss.HEL import HEL
from config import arg_config, path_config, proj_root
from utils.data.create_loader_imgs import create_loader
from utils.misc import AvgMeter, construct_path_dict, make_log, pre_mkdir
from utils.metric import CalTotalMetric

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torchcudnn.benchmark = True
torchcudnn.enabled = True
torchcudnn.deterministic = True


class Trainer:
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args
        self.dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to_pil = transforms.ToPILImage()
        pprint(self.args)

        self.data_mode = self.args["data_mode"]
        if self.args["suffix"]:
            self.model_name = self.args["model"] + "_" + self.args["suffix"]
        else:
            self.model_name = self.args["model"]
        self.path = construct_path_dict(proj_root=proj_root, exp_name=self.model_name)

        if self.data_mode == "RGBD":
            self.tr_data_path = self.args["rgbd_data"]["tr_data_path"]
            self.te_data_list = self.args["rgbd_data"]["te_data_list"]
        elif self.data_mode == "RGB":
            self.tr_data_path = self.args["rgb_data"]["tr_data_path"]
            self.te_data_list = self.args["rgb_data"]["te_data_list"]
        else:
            raise NotImplementedError

        self.save_path = self.path["save"]
        self.save_pre = self.args["save_pre"]

        # 依赖与前面属性的属性
        self.pth_path = self.path["final_state_net"]
        self.tr_loader = create_loader(
            data_path=self.tr_data_path, mode="train", get_length=False, data_mode=self.data_mode,
        )

        self.net = getattr(network, self.args["model"])(pretrained=True).to(self.dev)

        # 损失函数
        self.loss_funcs = [BCELoss(reduction=self.args["reduction"]).to(self.dev)]
        if self.args["use_aux_loss"]:
            self.loss_funcs.append(HEL().to(self.dev))

        # 设置优化器
        self.opti = self.make_optim()

        # 训练相关
        if self.args["resume"]:
            self.resume_checkpoint(load_path=self.path["final_full_net"], mode="all")
        else:
            self.start_epoch = 0
        self.end_epoch = self.args["epoch_num"]
        self.iter_num = self.end_epoch * len(self.tr_loader)

    def total_loss(self, train_preds, train_alphas):
        loss_list = []
        loss_item_list = []

        assert len(self.loss_funcs) != 0, "请指定损失函数`self.loss_funcs`"
        for loss in self.loss_funcs:
            loss_out = loss(train_preds, train_alphas)
            loss_list.append(loss_out)
            loss_item_list.append(f"{loss_out.item():.5f}")

        train_loss = sum(loss_list)
        return train_loss, loss_item_list

    def train(self):
        for curr_epoch in range(self.start_epoch, self.end_epoch):
            train_loss_record = AvgMeter()

            if self.args["lr_type"] == "poly":
                self.change_lr(curr_epoch)
            else:
                raise NotImplementedError

            for train_batch_id, train_data in enumerate(self.tr_loader):
                curr_iter = curr_epoch * len(self.tr_loader) + train_batch_id

                self.opti.zero_grad()
                train_inputs, train_masks, *train_other_data = train_data
                train_inputs = train_inputs.to(self.dev, non_blocking=True)
                train_masks = train_masks.to(self.dev, non_blocking=True)
                if self.data_mode == "RGBD":
                    # train_other_data是一个list
                    train_depths = train_other_data[-1]
                    train_depths = train_depths.to(self.dev, non_blocking=True)
                    train_preds = self.net(train_inputs, train_depths)
                elif self.data_mode == "RGB":
                    train_preds = self.net(train_inputs)
                else:
                    raise NotImplementedError

                train_loss, loss_item_list = self.total_loss(train_preds, train_masks)
                train_loss.backward()
                self.opti.step()

                # 仅在累计的时候使用item()获取数据
                train_iter_loss = train_loss.item()
                train_batch_size = train_inputs.size(0)
                train_loss_record.update(train_iter_loss, train_batch_size)

                # 记录每一次迭代的数据
                if self.args["print_freq"] > 0 and (curr_iter + 1) % self.args["print_freq"] == 0:
                    log = (
                        f"[I:{curr_iter}/{self.iter_num}][E:{curr_epoch}:{self.end_epoch}]>"
                        f"[{self.model_name}]"
                        f"[Lr:{self.opti.param_groups[0]['lr']:.7f}]"
                        f"[Avg:{train_loss_record.avg:.5f}|Cur:{train_iter_loss:.5f}|"
                        f"{loss_item_list}]"
                    )
                    print(log)
                    make_log(self.path["tr_log"], log)

            # 每个周期都进行保存测试，保存的是针对第curr_epoch+1周期的参数
            self.save_checkpoint(
                curr_epoch + 1, full_net_path=self.path["final_full_net"], state_net_path=self.path["final_state_net"],
            )

        # 进行最终的测试，首先输出验证结果
        print(f" ==>> 训练结束 <<== ")

        for data_name, data_path in self.te_data_list.items():
            print(f" ==>> 使用测试集{data_name}测试 <<== ")
            self.te_loader, self.te_length = create_loader(
                data_path=data_path, mode="test", get_length=True, data_mode=self.data_mode,
            )
            self.save_path = os.path.join(self.path["save"], data_name)
            if not os.path.exists(self.save_path):
                print(f" ==>> {self.save_path} 不存在, 这里创建一个 <<==")
                os.makedirs(self.save_path)
            results = self.test(save_pre=self.save_pre)
            fixed_pre_results = {k: f"{v:.3f}" for k, v in results.items()}
            msg = f" ==>> 在{data_name}:'{data_path}'测试集上结果\n >> {fixed_pre_results}"
            print(msg)
            make_log(self.path["te_log"], msg)

    def test(self, save_pre):
        self.net.eval()

        cal_total_metrics = CalTotalMetric(num=self.te_length, beta_for_wfm=1)

        tqdm_iter = tqdm(enumerate(self.te_loader), total=len(self.te_loader), leave=False)
        for test_batch_id, test_data in tqdm_iter:
            tqdm_iter.set_description(f"{self.model_name}:" f"te=>{test_batch_id + 1}")
            with torch.no_grad():
                in_imgs, in_names, in_mask_paths, *in_depths = test_data
                in_imgs = in_imgs.to(self.dev, non_blocking=True)
                if self.data_mode == "RGBD":
                    in_depths = in_depths[0]
                    in_depths = in_depths.to(self.dev, non_blocking=True)
                    outputs = self.net(in_imgs, in_depths)
                elif self.data_mode == "RGB":
                    outputs = self.net(in_imgs)
                else:
                    raise NotImplementedError

            outputs_np = outputs.cpu().detach()

            for item_id, out_item in enumerate(outputs_np):
                gimg_path = osp.join(in_mask_paths[item_id])
                gt_img = Image.open(gimg_path).convert("L")
                out_img = self.to_pil(out_item).resize(gt_img.size, resample=Image.NEAREST)

                if save_pre:
                    oimg_path = osp.join(self.save_path, in_names[item_id] + ".png")
                    out_img.save(oimg_path)

                gt_img = np.asarray(gt_img)
                out_img = np.asarray(out_img)

                gt_img = gt_img / (gt_img.max() + 1e-8)
                gt_img = np.where(gt_img > 0.5, 1, 0)

                max_out_img = out_img.max()
                min_out_img = out_img.min()
                if max_out_img == min_out_img:
                    out_img = out_img / 255
                else:
                    out_img = (out_img - min_out_img) / (max_out_img - min_out_img)

                cal_total_metrics.update(out_img, gt_img)
        results = cal_total_metrics.show()
        return results

    def change_lr(self, curr):
        total_num = self.end_epoch
        if self.args["lr_type"] == "poly":
            ratio = pow((1 - float(curr) / total_num), self.args["lr_decay"])
            self.opti.param_groups[0]["lr"] = self.opti.param_groups[0]["lr"] * ratio
            self.opti.param_groups[1]["lr"] = self.opti.param_groups[0]["lr"]
        else:
            raise NotImplementedError

    def make_optim(self):
        if self.args["optim"] == "sgd_trick":
            # https://github.com/implus/PytorchInsight/blob/master/classification/imagenet_tricks.py
            params = [
                {
                    "params": [p for name, p in self.net.named_parameters() if ("bias" in name or "bn" in name)],
                    "weight_decay": 0,
                },
                {
                    "params": [
                        p for name, p in self.net.named_parameters() if ("bias" not in name and "bn" not in name)
                    ]
                },
            ]
            optimizer = SGD(
                params,
                lr=self.args["lr"],
                momentum=self.args["momentum"],
                weight_decay=self.args["weight_decay"],
                nesterov=self.args["nesterov"],
            )
        else:
            raise NotImplementedError
        print("optimizer = ", optimizer)
        return optimizer

    def save_checkpoint(self, current_epoch, full_net_path, state_net_path):
        """
        保存完整参数模型（大）和状态参数模型（小）

        Args:
            current_epoch (int): 当前周期
            full_net_path (str): 保存完整参数模型的路径
            state_net_path (str): 保存模型权重参数的路径
        """
        state_dict = {
            "epoch": current_epoch,
            "net_state": self.net.state_dict(),
            "opti_state": self.opti.state_dict(),
        }
        torch.save(state_dict, full_net_path)
        torch.save(self.net.state_dict(), state_net_path)

    def resume_checkpoint(self, load_path, mode="all"):
        """
        从保存节点恢复模型

        Args:
            load_path (str): 模型存放路径
            mode (str): 选择哪种模型恢复模式：'all'：回复完整模型，包括训练中的的参数；'onlynet'：仅恢复模型权重参数
        """
        if os.path.exists(load_path) and os.path.isfile(load_path):
            print(f" =>> loading checkpoint '{load_path}' <<== ")
            checkpoint = torch.load(load_path, map_location=self.dev)
            if mode == "all":
                self.start_epoch = checkpoint["epoch"]
                self.net.load_state_dict(checkpoint["net_state"])
                self.opti.load_state_dict(checkpoint["opti_state"])
                print(f" ==> loaded checkpoint '{load_path}' (epoch {checkpoint['epoch']})")
            elif mode == "onlynet":
                self.net.load_state_dict(checkpoint)
                print(f" ==> loaded checkpoint '{load_path}' " f"(only has the net's weight params) <<== ")
            else:
                raise NotImplementedError
        else:
            raise Exception(f"{load_path}路径不正常，请检查")


if __name__ == "__main__":
    # 保存备份数据 ###########################################################
    print(f" ===========>> {datetime.now()}: 初始化开始 <<=========== ")
    init_start = datetime.now()
    pre_mkdir()
    trainer = Trainer(arg_config, path_config)
    print(f" ==>> 初始化完毕，用时：{datetime.now() - init_start} <<== ")

    shutil.copy(f"{proj_root}/config.py", path_config["cfg_log"])
    shutil.copy(f"{proj_root}/train.py", path_config["trainer_log"])

    # 训练模型 ###############################################################
    print(f" ===========>> {datetime.now()}: 开始训练 <<=========== ")
    trainer.train()
    print(f" ===========>> {datetime.now()}: 结束训练 <<=========== ")
