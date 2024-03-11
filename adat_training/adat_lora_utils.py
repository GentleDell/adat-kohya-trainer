import os

import tensorboard


def read_file(filename):
    with open(filename, "r") as f:
        contents = f.read()
    return contents


def write_file(filename, contents):
    with open(filename, "w") as f:
        f.write(contents)


def visualize_loss(training_logs_path):
    # 看一下训练收否收敛，保存的模型中离**较小**处最近的那个效果一般最好
    # 参考: ChilloutMix 的finetune loss在 0.043896 左右
    os.chdir(training_logs_path)
    os.system(f"tensorboard --logdir {training_logs_path} --port=6006")


def format_config(configs: dict):
    args = ""
    for k, v in configs.items():
        if k.startswith("_"):
            args += f'"{v}" '
        elif isinstance(v, str):
            args += f'--{k}="{v}" '
        elif isinstance(v, bool) and v:
            args += f"--{k} "
        elif isinstance(v, float) and not isinstance(v, bool):
            args += f"--{k}={v} "
        elif isinstance(v, int) and not isinstance(v, bool):
            args += f"--{k}={v} "

    return args
