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
    
    !pip install --upgrade bitsandbytes
    !pip install {'-q' if not verbose else ''} --upgrade -r requirements.txt

"""

if __name__ == "__main__":
    arguments = parser.parse_args()

    # =====================================
    dir_settings = EnvChecker(arguments)
    dir_settings.config_env()

    # =====================================
    model_checker = ModelChecker(
        dir_settings.root_dir,
        dir_settings.pretrained_model_dir,
        dir_settings.vae_dir,
        arguments,
    )
    model_checker.install_checkpoint()
    model_checker.install_vae()

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

    # =====================================
    annotator = DataAnnotator(
        reader.train_data_dir,
        os.path.join(dir_settings.pretrained_model_dir, "annotation_model"),
        arguments,
    )
    annotator.annotate_images(dir_settings.finetune_dir)
    annotator.process_annotation()

    # =====================================
    model_config = TrainModelConfig(dir_settings, model_checker, arguments)
    model_config.config_dataset(
        reader.train_data_dir, reader.reg_data_dir, dir_settings.config_dir
    )

    lora_config = LoraConfig(model_config, dir_settings, arguments)

    start_training(dir_settings, model_config, lora_config)

    visualize_loss(os.path.join(dir_settings.phone_number_dir, "logs"))
