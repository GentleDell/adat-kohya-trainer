import os
import concurrent

from tqdm import tqdm

from adat_cmd_args import parser
from adat_lora_utils import visualize_loss
from adat_lora_dreambooth import (
    EnvChecker,
    ModelChecker,
    DataReader,
    DataProcessor,
    DataAnnotator,
    TrainModelConfig,
    LoraConfig,
    start_training,
)


"""
Environments:
    !apt -y update {'-qq' if not verbose else ''}
    !apt install libunwind8-dev {'-qq' if not verbose else ''}
    !apt-get install libnettle8 {'-qq' if not verbose else ''}
    !apt-get install aria2 {'-qq' if not verbose else ''}
    !sudo apt-get install unzip
    
    conda env create --name adat_trainer --file=environment.yml
"""

"""
Arguments:
    root_dir        固定的第一个参数, 训练根目录,里面放的是Common的东西,比如模型
    phone_number    固定的第二个参数, 电话号码, 用于在创建训练数据保存文件夹
    dataset_dir     固定的第三个参数, 数据集所在的文件夹
                        如果上传的直接是图片, 这个就是上传的图片所在文件夹
                        如果上传的是压缩包, 则这个路径也要设置但是不会被用到, 请使用 --zip_file_path
    lora_model_name 固定的第四的参数, 训练出来LoRA模型的名字
    concept_prompt  固定的第五个参数, LoRA模型学习的概念
    test_prompts    固定的第六个参数, 训练中测试时用来生成图片的prompt

    --base_v1_model       可选参数, 指定使用的 SDv1 基础模型
    --base_v2_model       可选参数, 指定使用的 SDv2 基础模型, 不可和v1同时使用
    --vae_model           可选参数, 指定使用的 VAE 模型
    --annotation_model    可选参数, 指定用来进行标注的模型

    --is_zip_file         可选参数, 表明使用的数据集是压缩包的形式
    --zip_file_path       可选参数, 压缩包的路径（需要和上方参数配合使用）

    --network_dim            可选参数, LoRA网络维数
    --network_alpha          可选参数, 网络扩增权重(防止下溢并稳定训练), 0~network_dim
                               注意 如果alpha为16,dim为32,则使用强度为16/32 = 0.5, 
                               这意味着学习率只有“Learning Rate”设置值的一半的效力.
                               如果网络性能不好, 可以试试相比dim较小的alpha值
    --resume_model_path      可选参数, 从已有LoRA继续/恢复训练时, 使用的LoRA路径
    --sampler_for_training   训练时使用的采样器

    --dataset_repeats        可选参数, 数据集重复次数
    --dataset_resolution     可选参数, 数据分辨率
    --number_epochs          可选参数, 训练的批数
    --batch_size             可选参数, 一次迭代使用的数据单元数量
    --save_every_n_epochs    可选参数, 每隔n步保存一个LoRA模型
    --training_seed          可选参数, 训练所使用的随机种子
"""

if __name__ == "__main__":
    arguments = parser.parse_args()

    # =====================================
    dir_settings = EnvChecker(arguments)
    dir_settings.config_env()
    print("Directories and base env configuring succeed.\n")

    # =====================================
    model_checker = ModelChecker(
        dir_settings.root_dir,
        dir_settings.pretrained_model_dir,
        dir_settings.vae_dir,
        arguments,
    )
    model_checker.install_checkpoint()
    model_checker.install_vae()
    print("Base model downloading succeeds.\n")

    # =====================================
    reader = DataReader(dir_settings.phone_number_dir)
    if arguments.is_zip_file:
        if os.path.exists(arguments.zip_file_path):
            zipfile_path = arguments.zip_file_path
            unzip_to = reader.train_data_dir

            os.makedirs(unzip_to, exist_ok=True)
            reader.extract_dataset(zipfile_path, unzip_to)
        else:
            raise FileNotFoundError
    else:
        source_data = os.path.join(arguments.dataset_dir + "/*")
        os.system(f"cp -rT {arguments.dataset_dir} {reader.train_data_dir}")
    print("Training dataset loading succeeds.\n")

    # =====================================
    processor = DataProcessor()
    processor.clean_directory(reader.train_data_dir)
    if processor.convert:
        images = processor.find_images(reader.train_data_dir)
        num_batches = len(images) // processor.batch_size + 1
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for i in tqdm(range(num_batches)):
                start = i * processor.batch_size
                end = start + processor.batch_size
                batch = images[start:end]
                executor.map(processor.process_image, batch)
        print("All images have been converted")
    print("Training dataset processing succeeds.\n")

    # =====================================
    annotator = DataAnnotator(
        reader.train_data_dir,
        os.path.join(dir_settings.pretrained_model_dir, "annotation_model"),
        arguments,
    )
    annotator.annotate_images(dir_settings.finetune_dir)
    annotator.process_annotation()
    print("Training dataset annotation succeeds.\n")

    # =====================================
    model_config = TrainModelConfig(dir_settings, model_checker, arguments)
    model_config.config_dataset(
        reader.train_data_dir, reader.reg_data_dir, dir_settings.config_dir
    )
    print("Training configuration succeeds.\n")

    lora_config = LoraConfig(model_config, dir_settings, arguments)
    print("LoRA configuration succeeds.\n")

    start_training(dir_settings, model_config, lora_config)
    print("Training succeeds.\n")

    visualize_loss(os.path.join(dir_settings.phone_number_dir, "logs"))
