import argparse

from adat_lora_constant import models, v2_models, vaes

#######################################################
# Refer to https://note.com/kohya_ss/n/nad3bce9a3622 for more detailed explanation
#######################################################

parser = argparse.ArgumentParser()

parser.add_argument(
    "root_dir",  # root_dir, 设置为NAS比较好，这样很多东西不用重装
    type=str,
    help="root directory of the training folder",
)
parser.add_argument(
    "phone_number",
    type=str,
    help="The phone number used to create folder",
    default="",
)
parser.add_argument(
    "dataset_dir",
    type=str,
    help="The directory where the dataset is located",
    default="",
)
parser.add_argument(
    "lora_model_name",
    type=str,
    help="The name of the trained lora model",
    default="",
)
parser.add_argument(
    "concept_prompt",
    type=str,
    help="The concept to be trained",
    default="",
)
parser.add_argument(
    "test_prompts",
    type=str,
    help="Prompts used in evaluation to visualize the improvement during "
    + "training, multi prompt groups can be splited in lists, as given "
    + "in the default value",
    default="[blue dress, girl], [red skirt, boy], [long dress, beach]",
)
parser.add_argument(
    "--base_v1_model",
    type=str,
    help="The name of the base model, derived from SD v1.5",
    choices=list(models.keys()),
    default="v1-5-pruned-emaonly",
)
parser.add_argument(
    "--base_v2_model",
    type=str,
    help="The name of the base model, derived from SD v2.0. By default, not use v2 model.",
    choices=list(v2_models.keys()),
    default="",
)
parser.add_argument(
    "--vae_model",
    type=str,
    help="The name of the vae model",
    choices=list(vaes.keys()),
    default="stablediffusion.vae.pt",
)
parser.add_argument(
    "--annotation_model",
    type=str,
    help="The name of the model for image annotation",
    choices=[
        "SmilingWolf/wd-v1-4-convnextv2-tagger-v2",
        "SmilingWolf/wd-v1-4-swinv2-tagger-v2",
        "SmilingWolf/wd-v1-4-convnext-tagger-v2",
        "SmilingWolf/wd-v1-4-vit-tagger-v2",
    ],
    default="SmilingWolf/wd-v1-4-convnextv2-tagger-v",
)
parser.add_argument(
    "--is_zip_file",
    action="store_true",
    help="set if the dataset for training is in a zipfile",
)
parser.add_argument(
    "--zip_file_path",
    type=str,
    help="The path to the zipped dataset",
    default="",
)
parser.add_argument(
    "--dataset_repeats",
    type=int,
    default=20,
    help="times to repeat the dataset",
)
parser.add_argument(
    "--dataset_resolution",
    type=int,
    default=512,
    help="image shape of the images",
)
parser.add_argument(
    "--network_dim",
    type=int,
    default=64,
    help="Network dimension of the LoRa",
)
parser.add_argument(
    "--network_alpha",
    type=int,
    default=16,
    help="Network alpha of the LoRa",
)
parser.add_argument(
    "--resume_model_path",
    type=str,
    help="Specify the path to the pretrained LoRA to resume training",
    default="",
)
parser.add_argument(
    "--sampler_for_training",
    type=str,
    help="The sampler used during training",
    default="ddim",
    choices=[
        "ddim",
        "pndm",
        "lms",
        "euler",
        "euler_a",
        "heun",
        "dpm_2",
        "dpm_2_a",
        "dpmsolver",
        "dpmsolver++",
        "dpmsingle",
        "k_lms",
        "k_euler",
        "k_euler_a",
        "k_dpm_2",
        "k_dpm_2_a",
    ],
)
parser.add_argument(
    "--number_epochs",
    type=int,
    default=20,
    help="number of training epochs",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=6,
    help="size of data batch for training",
)
parser.add_argument(
    "--save_every_n_epochs",
    type=int,
    default=5,
    help="save and test model every n epochs",
)
parser.add_argument(
    "--training_seed",
    type=int,
    default=42,
    help="Seed to fix the randomness for reproduction and comparitability",
)
