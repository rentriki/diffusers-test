import re

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

MODEL_ID = "CompVis/stable-diffusion-v1-4"
DEVICE   = "cuda"

CHAR_PAT = re.compile(r'[^0-9A-Za-z ]')

def make_outfilename(prompt):
    return '_'.join(CHAR_PAT.sub('', prompt).lower().split()) + '.png'

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Generate image with ' + \
                                                 'Stable Diffusion.')
    parser.add_argument('--prompt', '-p', type=str, help='text prompt', 
                        required=True)
    parser.add_argument('--seed', '-s', type=int, default=1024)
    parser.add_argument('--guidance_scale', '-g', type=float, default=7.5)
    parser.add_argument('--num_inference_steps', '-n', type=int, default=50)
    args = parser.parse_args()

    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, 
                                                   use_auth_token=True)
    pipe = pipe.to(DEVICE)

    with autocast(DEVICE):
        image = pipe(args.prompt, 
                     guidance_scale=args.guidance_scale,
                     num_inference_steps=args.num_inference_steps,
                     generator=torch.Generator(DEVICE).manual_seed(args.seed)
                    )["sample"][0]  
    
    image.save(make_outfilename(args.prompt))