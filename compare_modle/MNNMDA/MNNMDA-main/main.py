import os
import argparse
import scipy
import numpy as np
import pandas as pd
from model_zoo import *


def debug_model(model_cls=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--comment", default=None, type=str)
    if model_cls is None:
        parser.add_argument("--model", default="MNNMDA", choices=["MNNMDA", "LRLSHMDA", "NTSHMDA", "GATMDA", "KATZHMDA"])
    parser = Experiment.add_argparse_args(parser)
    options, other_args = parser.parse_known_args()
    model_cls = globals()[options.model]                    # 根据参数选择的模型名称，从全局命名空间中获取模型类
    parser = model_cls.add_argparse_args(parser)            # 向解析器添加模型特定的参数
    config = parser.parse_args()
    experiment = Experiment(**vars(config))                  # 使用解析得到的配置参数创建实验对象
    experiment.run(model_cls, comment=config.comment)        # 运行实验，传入模型类和备注信息


# 定义search_params函数，用于参数搜索
def search_params(model_cls=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--comment", default=None, type=str)
    if model_cls is None:
        parser.add_argument("--model", default="MNNMDA", choices=["MNNMDA", "LRLSHMDA", "NTSHMDA", "GATMDA", "KATZHMDA"])   # 如果没有明确指定模型类，则允许通过命令行参数选择模型
    parser = Experiment.add_argparse_args(parser)                       # 向解析器添加实验参数
    options, other_args = parser.parse_known_args()                     # 解析已知的参数
    model_cls = globals()[options.model]                                # 从全局命名空间中获取模型类
    parser = model_cls.add_argparse_args(parser)                        # 向解析器添加模型特定的参数
    config = parser.parse_args()                                         # 解析所有命令行参数
    experiment = Experiment(**vars(config))                              # 创建实验对象
    # experiment.run(model_cls, comment=config.comment, debug=True)

    for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:                        # 循环遍历alpha和beta参数的不同值，进行参数搜索
        for beta in [0.1, 1.0, 10.0, 100.0]:
            print(f"alpha:{alpha}, beta:{beta}")                        # 打印当前参数组合
            config.alpha = alpha
            config.beta = beta
            try:
                experiment = Experiment(**vars(config))                 # 使用当前参数组合创建新的实验对象
                # 运行实验，传入模型类和备注信息，并开启调试模式
                experiment.run(model_cls, comment=f"search_params/{alpha}-{beta}", debug=True)
            except:
                pass                 # 如果运行过程中出现异常，则忽略
    experiment.collect_result(os.path.join(experiment.DEFAULT_DIR, "search_params"))            # 收集实验结果


if __name__=="__main__":
    # search_params()
    debug_model()
    # save_dir = "/home/jm/PycharmProjects/yss/lcc/MNNMDA"
    # Experiment.collect_result(save_dir)
