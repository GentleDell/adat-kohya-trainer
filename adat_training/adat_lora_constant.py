models = {
    "Animefull-final-pruned": "https://huggingface.co/Linaqruf/personal-backup/resolve/main/models/animefull-final-pruned.ckpt",
    "Anything-v3-1": "https://huggingface.co/cag/anything-v3-1/resolve/main/anything-v3-1.safetensors",
    "AnyLoRA": "https://huggingface.co/Linaqruf/stolen/resolve/main/pruned-models/AnyLoRA_noVae_fp16-pruned.safetensors",
    "AnyLoRA-anime-mix": "https://huggingface.co/Lykon/AnyLoRA/resolve/main/AAM_Anylora_AnimeMix.safetensors",
    "AnimePastelDream": "https://huggingface.co/Lykon/AnimePastelDream/resolve/main/AnimePastelDream_Soft_noVae_fp16.safetensors",
    "Chillout-mix": "https://huggingface.co/Linaqruf/stolen/resolve/main/pruned-models/chillout_mix-pruned.safetensors",
    "OpenJourney-v4": "https://huggingface.co/prompthero/openjourney-v4/resolve/main/openjourney-v4.ckpt",
    "v1-5-pruned-emaonly": "https://huggingface.co/Linaqruf/stolen/resolve/main/pruned-models/stable_diffusion_1_5-pruned.safetensors",
}

v2_models = {
    "stable-diffusion-2-1-base": "https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.safetensors",
    "stable-diffusion-2-1-768v": "https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/v2-1_768-ema-pruned.safetensors",
    "plat-diffusion-v1-3-1": "https://huggingface.co/p1atdev/pd-archive/resolve/main/plat-v1-3-1.safetensors",
    "replicant-v1": "https://huggingface.co/gsdf/Replicant-V1.0/resolve/main/Replicant-V1.0.safetensors",
    "illuminati-diffusion-v1-0": "https://huggingface.co/IlluminatiAI/Illuminati_Diffusion_v1.0/resolve/main/illuminati_diffusion_v1.0.safetensors",
    "illuminati-diffusion-v1-1": "https://huggingface.co/4eJIoBek/Illuminati-Diffusion-v1-1/resolve/main/illuminatiDiffusionV1_v11.safetensors",
    "waifu-diffusion-1-4-anime-e2": "https://huggingface.co/hakurei/waifu-diffusion-v1-4/resolve/main/wd-1-4-anime_e2.ckpt",
    "waifu-diffusion-1-5-e2": "https://huggingface.co/waifu-diffusion/wd-1-5-beta2/resolve/main/checkpoints/wd-1-5-beta2-fp32.safetensors",
    "waifu-diffusion-1-5-e2-aesthetic": "https://huggingface.co/waifu-diffusion/wd-1-5-beta2/resolve/main/checkpoints/wd-1-5-beta2-aesthetic-fp32.safetensors",
}

vaes = {
    "none": "",
    "anime.vae.pt": "https://huggingface.co/Linaqruf/personal-backup/resolve/main/vae/animevae.pt",
    "waifudiffusion.vae.pt": "https://huggingface.co/hakurei/waifu-diffusion-v1-4/resolve/main/vae/kl-f8-anime.ckpt",
    "stablediffusion.vae.pt": "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.ckpt",
}


supported_types = [
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".caption",
    ".npz",
    ".txt",
    ".json",
]

background_colors = [
    (255, 255, 255),
    (0, 0, 0),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
]

hf_token = "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"
