# uncompyle6 version 3.9.2
# Python bytecode version base 3.8.0 (3413)
# Decompiled from: Python 3.8.19 (default, Mar 20 2024, 19:58:24) 
# [GCC 11.2.0]
# Embedded file name: /userHome/userhome4/demo/baby_video/Baby_Video/AnimateDiff/scripts/animate.py
# Compiled at: 2024-09-18 22:47:12
# Size of source mod 2**32: 10004 bytes
import argparse, datetime, inspect, os
from omegaconf import OmegaConf
import torch
import torchvision.transforms as transforms
import diffusers
from diffusers import AutoencoderKL, DDIMScheduler
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from animatediff.models.unet import UNet3DConditionModel
from animatediff.models.sparse_controlnet import SparseControlNetModel
from animatediff.pipelines.pipeline_animation import AnimationPipeline
from animatediff.utils.util import save_videos_grid
from animatediff.utils.util import load_weights, auto_download
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange, repeat
import csv, pdb, glob, math
from pathlib import Path
from PIL import Image
import numpy as np

@torch.no_grad()
def main(args):
    *_, func_args = inspect.getargvalues(inspect.currentframe())
    func_args = dict(func_args)
    
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
    time_str = datetime.datetime.now().strftime("%Y-%m-%d")
    savedir = os.path.join(f'{args.save_dir}',f'{Path(args.config).stem}/{time_str}')
    os.makedirs(savedir, exist_ok=True)
    
    config = OmegaConf.load(args.config)
    samples = []
    tokenizer = CLIPTokenizer.from_pretrained((args.pretrained_model_path), subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained((args.pretrained_model_path), subfolder="text_encoder").cuda()
    vae = AutoencoderKL.from_pretrained((args.pretrained_model_path), subfolder="vae").cuda()
    sample_idx = 0
    for model_idx, model_config in enumerate(config):
        model_config.W = model_config.get("W", args.W)
        model_config.H = model_config.get("H", args.H)
        model_config.L = model_config.get("L", args.L)
        inference_config = OmegaConf.load(model_config.get("inference_config", args.inference_config))
        unet = UNet3DConditionModel.from_pretrained_2d((args.pretrained_model_path), subfolder="unet", unet_additional_kwargs=(OmegaConf.to_container(inference_config.unet_additional_kwargs))).cuda()
        controlnet = controlnet_images = None
        if model_config.get("controlnet_path", "") != "":
            assert model_config.get("controlnet_images", "") != ""
            assert model_config.get("controlnet_config", "") != ""
            unet.config.num_attention_heads = 8
            unet.config.projection_class_embeddings_input_dim = None
            controlnet_config = OmegaConf.load(model_config.controlnet_config)
            controlnet = SparseControlNetModel.from_unet(unet, controlnet_additional_kwargs=(controlnet_config.get("controlnet_additional_kwargs", {})))
            auto_download((model_config.controlnet_path), is_dreambooth_lora=False)
            print(f"loading controlnet checkpoint from {model_config.controlnet_path} ...")
            controlnet_state_dict = torch.load((model_config.controlnet_path), map_location="cpu")
            controlnet_state_dict = controlnet_state_dict["controlnet"] if "controlnet" in controlnet_state_dict else controlnet_state_dict
            controlnet_state_dict = {name: param for name, param in controlnet_state_dict.items() if "pos_encoder.pe" not in name}
            controlnet_state_dict.pop("animatediff_config", "")
            controlnet.load_state_dict(controlnet_state_dict)
            controlnet.cuda()
            image_paths = model_config.controlnet_images
            if isinstance(image_paths, str):
                image_paths = [
                 image_paths]
            print("controlnet image paths:")
            for path in image_paths:
                print(path)
            assert len(image_paths) <= model_config.L

            image_transforms = transforms.Compose([
             transforms.RandomResizedCrop((
              model_config.H, model_config.W),
               (1.0, 1.0), ratio=(
              model_config.W / model_config.H, model_config.W / model_config.H)),
             transforms.ToTensor()])
            
            if model_config.get("normalize_condition_images", False):

                def image_norm(image):
                    image = image.mean(dim=0, keepdim=True).repeat(3, 1, 1)
                    image -= image.min()
                    image /= image.max()
                    return image

            else:
                image_norm = lambda x: x
            controlnet_images = [image_norm(image_transforms(Image.open(path).convert("RGB"))) for path in image_paths]
            os.makedirs((os.path.join(savedir, "control_images")), exist_ok=True)

            for i, image in enumerate(controlnet_images):
                Image.fromarray((255.0 * image.numpy().transpose(1, 2, 0)).astype(np.uint8)).save(f"{savedir}/control_images/{i}.png")

            controlnet_images = torch.stack(controlnet_images).unsqueeze(0).cuda()
            controlnet_images = rearrange(controlnet_images, "b f c h w -> b c f h w")
            if controlnet.use_simplified_condition_embedding:
                num_controlnet_images = controlnet_images.shape[2]
                controlnet_images = rearrange(controlnet_images, "b c f h w -> (b f) c h w")
                controlnet_images = vae.encode(controlnet_images * 2.0 - 1.0).latent_dist.sample() * 0.18215
                controlnet_images = rearrange(controlnet_images, "(b f) c h w -> b c f h w", f=num_controlnet_images)
        if is_xformers_available() and not args.without_xformers:
            unet.enable_xformers_memory_efficient_attention()
            if controlnet is not None:
                controlnet.enable_xformers_memory_efficient_attention()
        pipeline = AnimationPipeline(vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=controlnet,
            scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs))).to("cuda")
        pipeline = load_weights(pipeline,
            motion_module_path=(model_config.get("motion_module", "")),
            motion_module_lora_configs=(model_config.get("motion_module_lora_configs", [])),
            adapter_lora_path=(model_config.get("adapter_lora_path", "")),
            adapter_lora_scale=(model_config.get("adapter_lora_scale", 1.0)),
            dreambooth_model_path=(model_config.get("dreambooth_path", "")),
            lora_model_path=(model_config.get("lora_model_path", "")),
            lora_alpha=(model_config.get("lora_alpha", 0.8))).to("cuda")
        if args.my_prompt:
            with open(args.my_prompt, "r") as f:
                prompts = [line.strip() for line in f.readlines()]
            n_prompts = [
                "beard, EasyNegative, lowres, chromatic aberration, depth of field, motion blur, blurry, bokeh, bad quality, worst quality, multiple arms, badhand"] * len(prompts)
            random_seeds = model_config.get("seed", [-1])[0]
            random_seeds = [random_seeds] if isinstance(random_seeds, int) else list(random_seeds)
            random_seeds = random_seeds * len(prompts) if len(random_seeds) == 1 else random_seeds
        else:
            prompts = model_config.prompt
            n_prompts = list(model_config.n_prompt) * len(prompts) if len(model_config.n_prompt) == 1 else model_config.n_prompt
            random_seeds = model_config.get("seed", [-1])
            random_seeds = [random_seeds] if isinstance(random_seeds, int) else list(random_seeds)
            random_seeds = random_seeds * len(prompts) if len(random_seeds) == 1 else random_seeds
        config[model_idx].random_seed = []
        for prompt_idx, (prompt, n_prompt, random_seed) in enumerate(zip(prompts, n_prompts, random_seeds)):
            if random_seed != -1:
                torch.manual_seed(random_seed)
            else:
                torch.seed()
            config[model_idx].random_seed.append(torch.initial_seed())
            print(f"current seed: {torch.initial_seed()}")
            print(f"sampling {prompt} ...")
            sample = pipeline(prompt,
              negative_prompt=n_prompt,
              num_inference_steps=(model_config.steps),
              guidance_scale=(model_config.guidance_scale),
              width=(model_config.W),
              height=(model_config.H),
              video_length=(model_config.L),
              controlnet_images=controlnet_images,
              controlnet_image_index=(model_config.get("controlnet_image_indexs", [0]))).videos
            samples.append(sample)
            save_videos_grid(sample, f"{savedir}/{sample_idx}_prompt.gif")
            sample_idx += 1
        if sample_idx%3==0 and sample_idx>=3:
            samples = torch.concat(samples)
            #titles=sample_idx//3
            save_videos_grid(samples, f"{savedir}/total_sample.gif", n_rows=3)
            samples=[]
        break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-model-path", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--inference-config", type=str, default="configs/inference/inference-v1.yaml")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--L", type=int, default=16)
    parser.add_argument("--W", type=int, default=512)
    parser.add_argument("--H", type=int, default=512)
    parser.add_argument("--without-xformers", action="store_true")
    parser.add_argument("--my_prompt", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()
    main(args)
