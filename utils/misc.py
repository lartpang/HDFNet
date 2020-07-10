# @Time    : 2020/7/8
# @Author  : Lart Pang
# @Email   : lartpang@163.com
# @File    : misc.py
# @Project : HDFNet
# @GitHub  : https://github.com/lartpang
import os
from datetime import datetime


class AvgMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def pre_mkdir(path_config):
    # 提前创建好记录文件，避免自动创建的时候触发文件创建事件
    check_mkdir(path_config["pth_log"])
    make_log(path_config["te_log"], f"=== te_log {datetime.now()} ===")
    make_log(path_config["tr_log"], f"=== tr_log {datetime.now()} ===")

    # 提前创建好存储预测结果和存放模型以及tensorboard的文件夹
    check_mkdir(path_config["save"])
    check_mkdir(path_config["pth"])
    check_mkdir(path_config["tb"])


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def make_log(path, context):
    with open(path, "a") as log:
        log.write(f"{context}\n")


def check_dir_path_valid(path: list):
    for p in path:
        if p:
            assert os.path.exists(p)
            assert os.path.isdir(p)


def construct_path_dict(proj_root, exp_name):
    ckpt_path = os.path.join(proj_root, "output")

    pth_log_path = os.path.join(ckpt_path, exp_name)

    tb_path = os.path.join(pth_log_path, "tb")
    save_path = os.path.join(pth_log_path, "pre")
    pth_path = os.path.join(pth_log_path, "pth")

    final_full_model_path = os.path.join(pth_path, "checkpoint_final.pth.tar")
    final_state_path = os.path.join(pth_path, "state_final.pth")

    tr_log_path = os.path.join(pth_log_path, f"tr_{str(datetime.now())[:10]}.txt")
    te_log_path = os.path.join(pth_log_path, f"te_{str(datetime.now())[:10]}.txt")
    cfg_log_path = os.path.join(pth_log_path, f"cfg_{str(datetime.now())[:10]}.txt")
    trainer_log_path = os.path.join(pth_log_path, f"trainer_{str(datetime.now())[:10]}.txt")

    path_config = {
        "ckpt_path": ckpt_path,
        "pth_log": pth_log_path,
        "tb": tb_path,
        "save": save_path,
        "pth": pth_path,
        "final_full_net": final_full_model_path,
        "final_state_net": final_state_path,
        "tr_log": tr_log_path,
        "te_log": te_log_path,
        "cfg_log": cfg_log_path,
        "trainer_log": trainer_log_path,
    }

    return path_config
