# HDFNet

(ECCV 2020) Hierarchical Dynamic Filtering Network for RGB-D Salient Object Detection

```text
@inproceedings{HDFNet-ECCV2020,
    author = {Youwei Pang and Lihe Zhang and Xiaoqi Zhao and Huchuan Lu},
    title = {Hierarchical Dynamic Filtering Network for RGB-D Salient Object Detection},
    booktitle = ECCV,
    year = {2020}
}
```

~~Our code will be released soon.~~

> Author: Lart Pang(`lartpang@163.com`)
>
> This is a complete, modular and easily modified code base based on PyTorch, which is suitable for the training and testing of significant target detection task model.

## Repository Details

* `base`: Store some code for backbone networks.
* `loss`: The code of the loss function.
* `module`: The code of important modules.
* `network`: The code of the network.
* `output`: It saves all results.
* `utils`: Some instrumental code.
    * `data/*py`: Some files about creating the dataloader.
    * `transforms/*py`: Some operations on data augmentation.
    * `metric.py`: max/mean/weighted F-measure, S-measure, E-measure and MAE. (**NOTE: If you find a problem in this part of the code, please notify me in time, thank you.**)
    * `misc.py`: Some useful utility functions.
    * `tensor_ops.py`: Some operations about tensors.
* `config.py`: Configuration file for model training and testing.
* `train.py`: I think you can understand.
* `test.py` and `test.sh`: These files can evaluate the performance of the model on the specified dataset. And the file `test.sh` is a simple example about how to configure and run `test.py`.

## Usage

### Environment

I provided conda environment configuration file (hdfnet.yaml), you can refer to the package version information.

And you can try `conda env create -f hdfnet.yaml` to create an environment to run our code.

### Train your own model

* Add your own module into the `module`.
* Add your own network into the `network` and import your model in the `network/__init__.py`.
* Modify `config.py`:
    * change the dataset path: `datasets_root`
    * change items in `arg_config`
        * `model` corresponds to the name of the model in `network`
        * `suffix`: finally, the form of `<model>_<suffix>` is used to form the alias of the model of this experiment and all files related to this experiment will be saved to the folder `<model>_<suffix>` in `output` folder
        * `resume`: set it to `False` to train normally
        * `data_mode`: set it to `RGBD` or `RGB` for using RGBD SOD datasets or RGB SOD datasets to train mdoel.
        * other items, like `lr`, `batch_size` and so on...
* Run the script: `python train.py`

If the training process is interrupted, you can use the following strategy to resume the training process.

* Set `resume` to `True`.
* Run the script `train.py` again.

### Evaluate model performance

There are two ways:
1. For models that have been trained, you can set `resume` to `True` and run the script `train.py` again.
2. Use the scripts `test.sh` and `test.py`. The specific method of use can be obtained by executing this command: `python test.py --help`.

### Only evaluate generated predictions

You can use the toolkit released by us: <https://github.com/lartpang/Py-SOD-VOS-EvalToolkit>.

## Related Works

* (ECCV 2020 Oral) Suppress and Balance: A Simple Gated Network for Salient Object Detection: https://github.com/Xiaoqi-Zhao-DLUT/GateNet-RGB-Saliency
* (ECCV 2020) A Single Stream Network for Robust and Real-time RGB-D Salient Object Detection: https://github.com/Xiaoqi-Zhao-DLUT/DANet-RGBD-Saliency
* (CVPR 2020) Multi-scale Interactive Network for Salient Object Detection: https://github.com/lartpang/MINet
