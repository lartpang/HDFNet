# @Time    : 2020/7/8
# @Author  : Lart Pang
# @Email   : lartpang@163.com
# @File    : config.py
# @Project : HDFNet
# @GitHub  : https://github.com/lartpang
import os

__all__ = ["proj_root", "arg_config"]

proj_root = os.path.dirname(__file__)
datasets_root = "/home/lart/Datasets/"

msra10k_path = os.path.join(datasets_root, "Saliency/RGBSOD", "MSRA10K")
ecssd_path = os.path.join(datasets_root, "Saliency/RGBSOD", "ECSSD")
dutomron_path = os.path.join(datasets_root, "Saliency/RGBSOD", "DUT-OMRON")
hkuis_path = os.path.join(datasets_root, "Saliency/RGBSOD", "HKU-IS")
pascals_path = os.path.join(datasets_root, "Saliency/RGBSOD", "PASCAL-S")
dutstr_path = os.path.join(datasets_root, "Saliency/RGBSOD", "DUTS/Train")
dutste_path = os.path.join(datasets_root, "Saliency/RGBSOD", "DUTS/Test")

lfsd_path = os.path.join(datasets_root, "Saliency/RGBDSOD", "LFSD")
rgbd135_path = os.path.join(datasets_root, "Saliency/RGBDSOD", "RGBD135")
dutrgbdtr_path = os.path.join(datasets_root, "Saliency/RGBDSOD", "DUT-RGBD/Train")
dutrgbdte_path = os.path.join(datasets_root, "Saliency/RGBDSOD", "DUT-RGBD/Test")
sip_path = os.path.join(datasets_root, "Saliency/RGBDSOD", "SIP")
ssd_path = os.path.join(datasets_root, "Saliency/RGBDSOD", "SSD")
stereo797_path = os.path.join(datasets_root, "Saliency/RGBDSOD", "STEREO-797")
stereo1000_path = os.path.join(datasets_root, "Saliency/RGBDSOD", "STEREO-1000")
rgbdtr_path = os.path.join(proj_root, "utils/data/data_list", "rgbd_train_jw.lst")
njudte_path = os.path.join(proj_root, "utils/data/data_list", "njud_test_jw.lst")
nlprte_path = os.path.join(proj_root, "utils/data/data_list", "nlpr_test_jw.lst")

# 配置区域 #####################################################################
arg_config = {
    # 常用配置
    "model": "HDFNet_Res50",
    "suffix": "DUTRGBD",
    "resume": False,  # 是否需要恢复模型
    "use_aux_loss": True,  # 是否使用辅助损失
    "save_pre": True,  # 是否保留最终的预测结果
    "epoch_num": 30,  # 训练周期
    "lr": 0.005,
    # 任务模式，支持RGB和RGBD两种类型任务的训练与测试
    "data_mode": "RGBD",  # 'RGB'/'RGBD'
    # RGBD
    "rgbd_data": {
        "tr_data_path": rgbdtr_path,
        # "tr_data_path": dutrgbdtr_path,
        "te_data_list": {
            "dutrgbd": dutrgbdte_path,
            # "lfsd": lfsd_path,
            # "njud": njudte_path,
            # "nlpr": nlprte_path,
            # "rgbd135": rgbd135_path,
            # "sip": sip_path,
            # "ssd": ssd_path,
            # "stereo797": stereo797_path,
            # "stereo1000": stereo1000_path,
        },
    },
    # RGB
    "rgb_data": {
        "tr_data_path": dutstr_path,
        "te_data_list": {
            "dutomron": dutomron_path,
            "hkuis": hkuis_path,
            "ecssd": ecssd_path,
            "pascals": pascals_path,
            "duts": dutste_path,
        },
    },
    "print_freq": 10,  # >0, 保存迭代过程中的信息
    "prefix": (".jpg", ".png"),
    # img_prefix, gt_prefix，用在使用索引文件的时候的对应的扩展名，这里使用的索引文件不包含后缀
    "reduction": "mean",  # 损失处理的方式，可选“mean”和“sum”
    "optim": "sgd_trick",  # 自定义部分的学习率
    "weight_decay": 5e-4,  # 微调时设置为0.0001
    "momentum": 0.9,
    "nesterov": False,
    "lr_type": "poly",
    "lr_decay": 0.9,  # poly
    "batch_size": 4,  # 要是继续训练, 最好使用相同的batchsize
    "num_workers": 8,  # 不要太大, 不然运行多个程序同时训练的时候, 会造成数据读入速度受影响
    "input_size": 320,
}
