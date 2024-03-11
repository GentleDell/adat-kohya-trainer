# -*- coding: utf-8 -*-
import os
import warnings
import shutil
import toml
import glob
import random
import getpass

from PIL import Image

from adat_lora_utils import read_file, write_file, format_config
from adat_lora_constant import (
    models,
    vaes,
    v2_models,
    supported_types,
    background_colors,
    hf_token,
)


class EnvChecker:
    def __init__(self, train_arguments) -> None:
        self.root_dir = train_arguments.root_dir
        self.common_dir = os.path.join(self.root_dir, "Common")  # 可重用数据
        self.repo_dir = os.path.join(self.common_dir, "adat-kohya-trainer")
        self.pretrained_model_dir = os.path.join(self.common_dir, "pretrained_model")
        self.vae_dir = os.path.join(self.common_dir, "vae")

        # 存储电话号码相关训练的所有中间结果和最终结果
        # 在训练中，由于有数据吞吐，所以放到系统云盘，保证速度，避免多训练占用NAS IO
        self.phone_number_dir = os.path.join(
            "/home/" + getpass.getuser() + "/LoRa", train_arguments.phone_number
        )
        self.config_dir = os.path.join(self.phone_number_dir, "config")

        # repo_dir
        self.accelerate_config = os.path.join(
            self.repo_dir, "accelerate_config/config.yaml"
        )
        self.finetune_dir = os.path.join(self.repo_dir, "finetune")

        self.bitsandbytes_main_py = (
            "/home/zhantao/Tools/miniconda3/envs/adat_lora_trainer/lib/python3.10/site-packages/bitsandbytes/cuda_setup/main.py"
            # "/usr/local/lib/python3.10/dist-packages/bitsandbytes/cuda_setup/main.py" # 只有用系统python直接装才是这个路径
        )
        assert os.path.exists(self.bitsandbytes_main_py), "bitsandbytes path is wrong"

        # Train branch
        self.branch = "feature/adatTrainer"

    def config_hg_accelerat(self):

        from accelerate.utils import write_basic_config

        if not os.path.exists(self.accelerate_config):
            write_basic_config(save_location=self.accelerate_config)

    def config_env(self):
        os.chdir(self.root_dir)

        for dir in [
            self.phone_number_dir,
            self.config_dir,
            self.pretrained_model_dir,
            self.vae_dir,
        ]:
            os.makedirs(dir, exist_ok=True)

        if not os.path.exists(self.repo_dir):
            os.chdir(self.common_dir)
            status = os.system(
                f"git clone git@github.com:GentleDell/adat-kohya-trainer.git"
            )
            if status != 0:
                raise Exception("Failed to clone branch or commit")

        if self.branch:
            os.chdir(self.repo_dir)
            status = os.system(f"git fetch")
            status = os.system(f"git checkout {self.branch}")
            if status != 0:
                raise Exception("Failed to checkout branch or commit")

        self.config_hg_accelerat()

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["BITSANDBYTES_NOWELCOME"] = "1"
        os.environ["SAFETENSORS_FAST_GPU"] = "1"

        cuda_path = "/usr/local/cuda-12.3/targets/x86_64-linux/lib/"
        assert os.path.exists(cuda_path), "cuda lib path is wrong"

        ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = f"{ld_library_path}{cuda_path}"
        os.chdir(self.root_dir)


class ModelChecker:
    def __init__(
        self, root_dir: str, pretrained_model_dir: str, vae_dir: str, arguments
    ) -> None:
        self.root_dir = root_dir
        self.pretrained_model_dir = pretrained_model_dir
        self.vae_dir = vae_dir

        self.SD_Models = []
        self.SD_v2Models = []
        self.SD_vaes = []

        model_name = arguments.base_v1_model
        v2_model_name = arguments.base_v2_model
        assert not (
            (len(model_name) > 0) and (len(v2_model_name) > 0)
        ), "Only train LoRA for one base model at a time."

        vae_name = arguments.vae_model

        if model_name:
            model_url = models.get(model_name)
            if model_url:
                self.SD_Models.append((model_name, model_url))

        if v2_model_name:
            v2_model_url = v2_models.get(v2_model_name)
            if v2_model_url:
                self.SD_v2Models.append((v2_model_name, v2_model_url))

        if vae_name in vaes:
            vae_url = vaes[vae_name]
            if vae_url:
                self.SD_vaes.append((vae_name, vae_url))

    def download_model(self, checkpoint_name, url):
        os.chdir(self.root_dir)
        ext = "ckpt" if url.endswith(".ckpt") else "safetensors"

        if os.path.isfile(
            os.path.join(self.pretrained_model_dir, f"{checkpoint_name}.{ext}")
        ):
            print(
                f"{checkpoint_name}.{ext}"
                + " already exist in "
                + self.pretrained_model_dir
            )
        else:
            user_header = f'"Authorization: Bearer {hf_token}"'
            os.system(
                f"aria2c --console-log-level=error --summary-interval=10 --header={user_header} -c -x 16 -k 1M -s 16 -d {self.pretrained_model_dir} -o {checkpoint_name}.{ext} {url}"
            )

    def install_checkpoint(self):
        if len(self.SD_Models) + len(self.SD_v2Models) > 1:
            raise ValueError("This script can train only 1 LoRA per running.")
        for model in self.SD_Models:
            self.download_model(model[0], model[1])
        for v2model in self.SD_v2Models:
            self.download_model(v2model[0], v2model[1])

    def download_vae(self, vae_name, url):
        if os.path.isfile(os.path.join(self.vae_dir, vae_name)):
            print(vae_name + " already exists in " + vae_name)
        else:
            user_header = f'"Authorization: Bearer {hf_token}"'
            os.system(
                f"aria2c --console-log-level=error --summary-interval=10 --header={user_header} -c -x 16 -k 1M -s 16 -d {self.vae_dir} -o {vae_name} {url}"
            )

    def install_vae(self):
        for vae in self.SD_vaes:
            self.download_vae(vae[0], vae[1])


class DataReader:
    def __init__(self, phone_number_dir) -> None:
        self.phone_number_dir = phone_number_dir

        # Create a folder in the phone number folder.
        self.train_data_dir = os.path.join(self.phone_number_dir, "train_data/")

        # Regularization Images is optional and can be skipped.
        self.reg_data_dir = os.path.join(self.phone_number_dir, "reg_data/")

        for dir in [self.train_data_dir, self.reg_data_dir]:
            os.makedirs(dir, exist_ok=True)

        print(f"Your train_data_dir : {self.train_data_dir}")
        print(f"Your reg_data_dir : {self.reg_data_dir}")

    def extract_dataset(self, zip_file, output_path):
        os.system(f"unzip -j -o {zip_file} -d {output_path}")

    def remove_files(self, train_dir, files_to_move):
        for filename in os.listdir(train_dir):
            file_path = os.path.join(train_dir, filename)
            if filename in files_to_move:
                if not os.path.exists(file_path):
                    shutil.move(file_path, self.phone_number_dir)
                else:
                    os.remove(file_path)


class DataProcessor:
    # This section will delete unnecessary files and unsupported media,
    # such as `.mp4`, `.webm`, and `.gif`, etc.

    def __init__(self) -> None:
        # Set the `convert` parameter to convert your transparent dataset with
        # an alpha channel (RGBA) to RGB and give it a white background.
        self.convert = False
        print("If images have transparency channel, set DataProcessor to convert!")

        # Give it a `random_color` background instead of white by checking the
        # corresponding option.
        self.random_color = False

        # Use the `recursive` option to preprocess subfolders as well.
        self.recursive = True

        self.batch_size = 32

    def clean_directory(self, directory):
        for item in os.listdir(directory):
            file_path = os.path.join(directory, item)
            if os.path.isfile(file_path):

                rn_item = os.path.splitext(item)[0] + os.path.splitext(item)[1].lower()
                rn_file_path = os.path.join(directory, rn_item)
                os.rename(file_path, rn_file_path)

                file_ext = os.path.splitext(rn_item)[1]
                if file_ext not in supported_types:
                    print(f"Deleting file {rn_item} from {directory}")
                    os.remove(rn_file_path)
            elif os.path.isdir(file_path) and self.recursive:
                self.clean_directory(file_path)

    def process_image(self, image_path):
        img = Image.open(image_path)
        img_dir, image_name = os.path.split(image_path)

        if img.mode in ("RGBA", "LA"):
            if self.random_color:
                background_color = random.choice(background_colors)
            else:
                background_color = (255, 255, 255)
            bg = Image.new("RGB", img.size, background_color)
            bg.paste(img, mask=img.split()[-1])

            if image_name.endswith(".webp"):
                bg = bg.convert("RGB")
                new_image_path = os.path.join(
                    img_dir, image_name.replace(".webp", ".jpg")
                )
                bg.save(new_image_path, "JPEG")
                os.remove(image_path)
                print(
                    f" Converted image: {image_name} to {os.path.basename(new_image_path)}"
                )
            else:
                bg.save(image_path, "PNG")
                print(f" Converted image: {image_name}")
        else:
            if image_name.endswith(".webp"):
                new_image_path = os.path.join(
                    img_dir, image_name.replace(".webp", ".jpg")
                )
                img.save(new_image_path, "JPEG")
                os.remove(image_path)
                print(
                    f" Converted image: {image_name} to {os.path.basename(new_image_path)}"
                )
            else:
                img.save(image_path, "PNG")

    def find_images(self, directory):
        images = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".png") or file.endswith(".webp"):
                    images.append(os.path.join(root, file))
        return images


class DataAnnotator:
    # Waifu Diffusion 1.4 Tagger V2 is a Danbooru-styled image classification
    # model developed by SmilingWolf. It can also be useful for general image
    # tagging, for example, `1girl, solo, looking_at_viewer, short_hair, bangs,
    # simple_background`. Choices of models:
    #     - "SmilingWolf/wd-v1-4-convnextv2-tagger-v2",
    #     - "SmilingWolf/wd-v1-4-swinv2-tagger-v2",
    #     - "SmilingWolf/wd-v1-4-convnext-tagger-v2",
    #     - "SmilingWolf/wd-v1-4-vit-tagger-v2"
    def __init__(self, train_data_dir, annotation_model, arguments) -> None:

        # Batchsize 8 ==> 7.8GB VRAM, adjust according to your GPU
        batch_size = 8

        max_data_loader_n_workers = min(os.cpu_count(), 16)
        model = "SmilingWolf/wd-v1-4-convnextv2-tagger-v2"
        # `recursive` option to process subfolders, for multi-concept training.
        self.recursive = True
        # Debug while tagging, it will print your image file with general tags
        # and character tags.
        verbose_logging = True
        # Separate `undesired_tags` with comma `(,)` if you want to remove
        # multiple tags, e.g. `1girl,solo,smile`.
        undesired_tags = ""
        # `general_threshold` for pruning tags (less tags, less flexible).
        # `character_threshold` is to train character tags, e.g. `hakurei reimu`.
        general_threshold = 0.35  # min:0, max:1
        character_threshold = 0.6  # min:0, max:1, step:0.05}

        config = {
            "_train_data_dir": train_data_dir,  # prepend _ means the para is mandatory
            "batch_size": batch_size,
            "repo_id": model,
            "model_dir": annotation_model,
            "recursive": self.recursive,
            "remove_underscore": True,
            "general_threshold": general_threshold,
            "character_threshold": character_threshold,
            "caption_extension": ".txt",
            "max_data_loader_n_workers": max_data_loader_n_workers,
            "debug": verbose_logging,
            "undesired_tags": undesired_tags,
        }

        self.args = format_config(config)

        self.final_args = f"python tag_images_by_wd14_tagger.py {self.args}"

        # .txt for SD tagger, .cap for BLIP
        self.extension = ".txt"

        # For example, "_red_skirt slim_body_" means the training is for these tags.
        self.custom_tag = arguments.concept_prompt

        # Enable this to append custom tags at the end of lines.
        # Otherwise, the prompts are prepended.
        self.append = False

        # Enable this if you want to remove the given captions/tags instead.
        self.remove_tag = False

        # Use `sub_folder` option to specify a subfolder for multi-concept training.
        # > Specify `--all` to process all subfolders/`recursive`
        sub_folder = "--all"

        if sub_folder == "":
            self.image_dir = train_data_dir
        elif sub_folder == "--all":
            self.image_dir = train_data_dir
            self.recursive = True
        else:
            self.image_dir = os.path.join(train_data_dir, sub_folder)
            os.makedirs(self.image_dir, exist_ok=True)

    def annotate_images(self, finetune_dir):
        os.chdir(finetune_dir)

        # The installation cache is in ~/cache/pip/huggingface
        os.system(f"python tag_images_by_wd14_tagger.py {self.args}")

    def process_tags(self, filename, custom_tag, append, remove_tag):
        contents = read_file(filename)
        tags = [tag.strip() for tag in contents.split(",")]
        custom_tags = [tag.strip() for tag in custom_tag.split(",")]

        for custom_tag in custom_tags:
            custom_tag = custom_tag.replace("_", " ")
            if remove_tag:
                while custom_tag in tags:
                    tags.remove(custom_tag)
            else:
                if custom_tag not in tags:
                    if append:
                        tags.append(custom_tag)
                    else:
                        tags.insert(0, custom_tag)

        contents = ", ".join(tags)
        write_file(filename, contents)

    def process_directory(self, image_dir, tag, append, remove_tag, recursive):
        for filename in os.listdir(image_dir):
            file_path = os.path.join(image_dir, filename)

            if os.path.isdir(file_path) and recursive:
                self.process_directory(file_path, tag, append, remove_tag, recursive)
            elif filename.endswith(self.extension):
                self.process_tags(file_path, tag, append, remove_tag)

    def process_annotation(self):
        if not any(
            [
                filename.endswith(self.extension)
                for filename in os.listdir(self.image_dir)
            ]
        ):
            for filename in os.listdir(self.image_dir):
                if filename.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
                    open(
                        os.path.join(
                            self.image_dir, filename.split(".")[0] + self.extension
                        ),
                        "w",
                    ).close()

        if self.custom_tag:
            self.process_directory(
                self.image_dir,
                self.custom_tag,
                self.append,
                self.remove_tag,
                self.recursive,
            )


class TrainModelConfig:
    def __init__(
        self,
        env_checker: EnvChecker,
        model_checker: ModelChecker,
        arguments,
    ) -> None:
        self.v2 = len(arguments.base_v2_model) > 0
        self.v_parameterization = False

        self.project_name = arguments.lora_model_name
        if not self.project_name:
            self.project_name = "last"

        chk_pt = model_checker.SD_Models + model_checker.SD_v2Models
        chk_pt_name = chk_pt[0][0]
        chk_pt_ext = "ckpt" if chk_pt[0][1].endswith(".ckpt") else "safetensors"
        self.pretrained_model_dir_name_or_path = os.path.join(
            env_checker.pretrained_model_dir, f"{chk_pt_name}.{chk_pt_ext}"
        )

        vae_name = model_checker.SD_vaes[0][0]
        self.vae = os.path.join(env_checker.vae_dir, vae_name)

        self.output_dir = os.path.join(env_checker.phone_number_dir, "output")

        sample_dir = os.path.join(self.output_dir, "sample")
        for dir in [self.output_dir, sample_dir]:
            if os.path.exists(dir):
                raise FileExistsError(
                    "Raising ERROR in case important results get erased."
                )
            os.makedirs(dir)

        self.dataset_repeats = arguments.dataset_repeats

        # `activation_word` is not used but is printed to metadata. Keep it to follow old format
        self.activation_word = "placeholder"

        self.caption_extension = ".txt"
        self.resolution = arguments.dataset_resolution
        self.keep_tokens = 0

        print("Project Name: ", self.project_name)
        print("Model Version: Stable Diffusion V1.x") if not self.v2 else ""
        (
            print("Model Version: Stable Diffusion V2.x")
            if self.v2 and not self.v_parameterization
            else ""
        )
        (
            print("Model Version: Stable Diffusion V2.x 768v")
            if self.v2 and self.v_parameterization
            else ""
        )
        (
            print("Pretrained Model Path: ", self.pretrained_model_dir_name_or_path)
            if self.pretrained_model_dir_name_or_path
            else print("No Pretrained Model path specified.")
        )
        print("VAE Path: ", self.vae) if self.vae else print("No VAE path specified.")
        print("Output Path: ", self.output_dir)

    def parse_folder_name(self, folder_name, default_num_repeats, default_class_token):
        folder_name_parts = folder_name.split("_")

        if len(folder_name_parts) == 2:
            if folder_name_parts[0].isdigit():
                num_repeats = int(folder_name_parts[0])
                class_token = folder_name_parts[1].replace("_", " ")
            else:
                num_repeats = default_num_repeats
                class_token = default_class_token
        else:
            num_repeats = default_num_repeats
            class_token = default_class_token

        return num_repeats, class_token

    def find_image_files(self, path):
        supported_extensions = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
        return [
            file
            for file in glob.glob(path + "/**/*", recursive=True)
            if file.lower().endswith(supported_extensions)
        ]

    def process_data_dir(
        self, data_dir, default_num_repeats, default_class_token, is_reg=False
    ):
        subsets = []

        images = self.find_image_files(data_dir)
        if images:
            subsets.append(
                {
                    "image_dir": data_dir,
                    "class_tokens": default_class_token,
                    "num_repeats": default_num_repeats,
                    **({"is_reg": is_reg} if is_reg else {}),
                }
            )

        for root, dirs, files in os.walk(data_dir):
            for folder in dirs:
                folder_path = os.path.join(root, folder)
                images = self.find_image_files(folder_path)

                if images:
                    num_repeats, class_token = self.parse_folder_name(
                        folder, default_num_repeats, default_class_token
                    )

                    subset = {
                        "image_dir": folder_path,
                        "class_tokens": class_token,
                        "num_repeats": num_repeats,
                    }

                    if is_reg:
                        subset["is_reg"] = True

                    subsets.append(subset)

        return subsets

    def config_dataset(self, train_data_dir, reg_data_dir, config_dir):
        train_subsets = self.process_data_dir(
            train_data_dir, self.dataset_repeats, self.activation_word
        )
        reg_subsets = self.process_data_dir(
            reg_data_dir, self.dataset_repeats, self.activation_word, is_reg=True
        )

        subsets = train_subsets + reg_subsets

        config = {
            "general": {
                "enable_bucket": True,
                "caption_extension": self.caption_extension,
                "shuffle_caption": True,
                "keep_tokens": self.keep_tokens,
                "bucket_reso_steps": 64,
                "bucket_no_upscale": False,
            },
            "datasets": [
                {
                    "resolution": self.resolution,
                    "min_bucket_reso": 320 if self.resolution > 640 else 256,
                    "max_bucket_reso": 1280 if self.resolution > 640 else 1024,
                    "caption_dropout_rate": 0,
                    "caption_tag_dropout_rate": 0,
                    "caption_dropout_every_n_epochs": 0,
                    "flip_aug": False,
                    "color_aug": False,
                    "face_crop_aug_range": [2.0, 3.0],  # TODO test
                    "subsets": subsets,
                }
            ],
        }

        self.dataset_config = os.path.join(config_dir, "dataset_config.toml")

        for key in config:
            if isinstance(config[key], dict):
                for sub_key in config[key]:
                    if config[key][sub_key] == "":
                        config[key][sub_key] = None
            elif config[key] == "":
                config[key] = None

        config_str = toml.dumps(config)

        with open(self.dataset_config, "w") as f:
            f.write(config_str)

        print(config_str)


class LoraConfig:
    def __init__(
        self, model_config: TrainModelConfig, env_checker: EnvChecker, arguments
    ) -> None:
        network_category = "LoRA"

        # higher value for `dim` or `alpha`, consider using a higher learning
        # rate as models with higher dimensions tend to learn faster.
        network_dim = arguments.network_dim  # recommended: 32, recommend not > 64
        network_alpha = arguments.network_alpha  # recommended: 1

        network_weight = arguments.resume_model_path
        network_module = "networks.lora"
        network_args = ""  # LoRA 不需要其他参数

        # Gamma for reducing the weight of high-loss timesteps. Lower numbers
        # have a stronger effect. The paper recommends 5.
        # Read the paper [here](https://arxiv.org/abs/2303.09556).
        min_snr_gamma = -1

        # Options: "AdamW", "AdamW8bit", "Lion", "SGDNesterov",
        # "SGDNesterov8bit", "DAdaptation", "AdaFactor"
        optimizer_type = "AdamW8bit"  # use_8bit_adam

        # Additional arguments for optimizer
        # e.g: ["decouple=True","weight_decay=0.6"]
        optimizer_args = ""

        # @markdown Set `unet_lr` to `1.0` if you use `DAdaptation` optimizer.
        # @markdown It is recommended to set `text_encoder_lr = 0.5 * unet_lr`.
        train_unet = True
        unet_lr = 1e-4
        train_text_encoder = True
        text_encoder_lr = 5e-5

        # Options: "linear", "cosine", "cosine_with_restarts", "polynomial",
        # "constant", "constant_with_warmup", "adafactor"] {allow-input: false}
        lr_scheduler = "constant"
        lr_warmup_steps = 0

        # You can define `num_cycles` value for `cosine_with_restarts` or
        # `power` value for `polynomial` in the field below.
        lr_scheduler_num_cycles = 0
        lr_scheduler_power = 0

        print("- LoRA Config:")
        (
            print(f"  - Min-SNR Weighting: {min_snr_gamma}")
            if not min_snr_gamma == -1
            else ""
        )
        print(f"  - Loading network module: {network_module}")
        print(f"  - {network_module} linear_dim set to: {network_dim}")
        print(f"  - {network_module} linear_alpha set to: {network_alpha}")
        if not network_weight:
            print("  - No LoRA weight loaded.")
        else:
            if os.path.exists(network_weight):
                print(f"  - Loading LoRA weight: {network_weight}")
                warnings.warn(f"\033[93m Will train from LoRA weight: {network_weight}")
            else:
                print(f"  - {network_weight} does not exist.")
                network_weight = ""

        print("- Optimizer Config:")
        print(f"  - Additional network category: {network_category}")
        print(f"  - Using {optimizer_type} as Optimizer")
        if optimizer_args:
            print(f"  - Optimizer Args: {optimizer_args}")
        if train_unet and train_text_encoder:
            print("  - Train UNet and Text Encoder")
            print(f"    - UNet learning rate: {unet_lr}")
            print(f"    - Text encoder learning rate: {text_encoder_lr}")
        if train_unet and not train_text_encoder:
            print("  - Train UNet only")
            print(f"    - UNet learning rate: {unet_lr}")
        if train_text_encoder and not train_unet:
            print("  - Train Text Encoder only")
            print(f"    - Text encoder learning rate: {text_encoder_lr}")
        print(f"  - Learning rate warmup steps: {lr_warmup_steps}")
        print(f"  - Learning rate Scheduler: {lr_scheduler}")
        if lr_scheduler == "cosine_with_restarts":
            print(f"  - lr_scheduler_num_cycles: {lr_scheduler_num_cycles}")
        elif lr_scheduler == "polynomial":
            print(f"  - lr_scheduler_power: {lr_scheduler_power}")

        lowram = True
        enable_sample_prompt = True

        sampler = arguments.sampler_for_training
        noise_offset = 0.0
        num_epochs = arguments.number_epochs

        vae_batch_size = arguments.batch_size
        train_batch_size = arguments.batch_size

        # Options: "no","fp16","bf16"
        mixed_precision = "fp16"

        # Options: "float", "fp16", "bf16"
        save_precision = "fp16"

        # Options: "save_every_n_epochs", "save_n_epoch_ratio"
        save_n_epochs_type = "save_every_n_epochs"

        save_n_epochs_type_value = arguments.save_every_n_epochs
        save_model_as = "safetensors"  # Options: "ckpt", "pt", "safetensors"
        clip_skip = 2
        gradient_checkpointing = False
        gradient_accumulation_steps = 1
        seed = arguments.training_seed
        logging_dir = os.path.join(env_checker.phone_number_dir, "logs")
        prior_loss_weight = 1.0

        # 训练过程中会使用下面这一组prompt进行生成，方便监控生成效果的变化，可以根据电话号码更改。
        # 中间生成的图会存在 5.1输出的log 的 "Output Path" 处。不同的几组prompt使用中括号分割，
        # 例如: "[blue dress, girl], [red skirt, boy], [long dress, beach]" 代表三组prompt。
        sample_prompt = arguments.test_prompts

        sample_str = (
            sample_prompt
            + f"""
        --n lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry \
        --w 512 \
        --h 768 \
        --l 7 \
        --s 28
        """
        )

        config = {
            "model_arguments": {
                "v2": model_config.v2,
                "v_parameterization": (
                    model_config.v_parameterization
                    if model_config.v2 and model_config.v_parameterization
                    else False
                ),
                "pretrained_model_name_or_path": model_config.pretrained_model_dir_name_or_path,
                "vae": model_config.vae,
            },
            "additional_network_arguments": {
                "no_metadata": False,
                "unet_lr": float(unet_lr) if train_unet else None,
                "text_encoder_lr": (
                    float(text_encoder_lr) if train_text_encoder else None
                ),
                "network_weights": network_weight,
                "network_module": network_module,
                "network_dim": network_dim,
                "network_alpha": network_alpha,
                "network_args": network_args,
                "network_train_unet_only": (
                    True if train_unet and not train_text_encoder else False
                ),
                "network_train_text_encoder_only": (
                    True if train_text_encoder and not train_unet else False
                ),
                "training_comment": None,
            },
            "optimizer_arguments": {
                "min_snr_gamma": min_snr_gamma if not min_snr_gamma == -1 else None,
                "optimizer_type": optimizer_type,
                "learning_rate": unet_lr,
                "max_grad_norm": 1.0,
                "optimizer_args": eval(optimizer_args) if optimizer_args else None,
                "lr_scheduler": lr_scheduler,
                "lr_warmup_steps": lr_warmup_steps,
                "lr_scheduler_num_cycles": (
                    lr_scheduler_num_cycles
                    if lr_scheduler == "cosine_with_restarts"
                    else None
                ),
                "lr_scheduler_power": (
                    lr_scheduler_power if lr_scheduler == "polynomial" else None
                ),
            },
            "dataset_arguments": {
                "cache_latents": True,
                "debug_dataset": False,
                "vae_batch_size": vae_batch_size,
            },
            "training_arguments": {
                "output_dir": model_config.output_dir,
                "output_name": model_config.project_name,
                "save_precision": save_precision,
                "save_every_n_epochs": (
                    save_n_epochs_type_value
                    if save_n_epochs_type == "save_every_n_epochs"
                    else None
                ),
                "save_n_epoch_ratio": (
                    save_n_epochs_type_value
                    if save_n_epochs_type == "save_n_epoch_ratio"
                    else None
                ),
                "save_last_n_epochs": None,
                "save_state": None,
                "save_last_n_epochs_state": None,
                "resume": None,
                "train_batch_size": train_batch_size,
                "max_token_length": 225,
                "mem_eff_attn": False,
                "xformers": True,
                "max_train_epochs": num_epochs,
                "max_data_loader_n_workers": 8,
                "persistent_data_loader_workers": True,
                "seed": seed if seed > 0 else None,
                "gradient_checkpointing": gradient_checkpointing,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "mixed_precision": mixed_precision,
                "clip_skip": clip_skip if not model_config.v2 else None,
                "logging_dir": logging_dir,
                "log_prefix": model_config.project_name,
                "noise_offset": noise_offset if noise_offset > 0 else None,
                "lowram": lowram,
            },
            "sample_prompt_arguments": {
                "sample_every_n_steps": None,
                "sample_every_n_epochs": 1 if enable_sample_prompt else 999999,
                "sample_sampler": sampler,
            },
            "dreambooth_arguments": {
                "prior_loss_weight": prior_loss_weight,
            },
            "saving_arguments": {"save_model_as": save_model_as},
        }

        self.config_path = os.path.join(env_checker.config_dir, "config_file.toml")
        self.prompt_path = os.path.join(env_checker.config_dir, "sample_prompt.txt")

        for key in config:
            if isinstance(config[key], dict):
                for sub_key in config[key]:
                    if config[key][sub_key] == "":
                        config[key][sub_key] = None
            elif config[key] == "":
                config[key] = None

        config_str = toml.dumps(config)

        write_file(self.config_path, config_str)
        write_file(self.prompt_path, sample_str)

        print(config_str)


def start_training(
    env_checker: EnvChecker, model_config: ModelChecker, lora_config: LoraConfig
):
    accelerate_conf = {
        "config_file": env_checker.accelerate_config,
        "num_cpu_threads_per_process": 1,
    }

    train_conf = {
        "sample_prompts": lora_config.prompt_path,
        "dataset_config": model_config.dataset_config,
        "config_file": lora_config.config_path,
    }

    accelerate_args = format_config(accelerate_conf)
    train_args = format_config(train_conf)
    os.chdir(env_checker.repo_dir)

    # Will download the CLIPvit model from hg, ~1.71GB
    # The model is saved in huggingface cache folder, e.g. ~/.cache/huggingface/hub
    os.system(f"accelerate launch {accelerate_args} train_network.py {train_args}")
