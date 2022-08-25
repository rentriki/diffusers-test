import re

from datetime import datetime

import torch

from torch import autocast
from diffusers import StableDiffusionPipeline

MODEL_ID = "CompVis/stable-diffusion-v1-4"
DEVICE   = "cuda"

CHAR_PAT      = re.compile(r'[^0-9A-Za-z ]')
TIMESTAMP_PAT = re.compile(r'[:\.]')

def make_outfilename(prompt):
    title = '-'.join(CHAR_PAT.sub('', prompt).lower().split())
    now = datetime.now().isoformat(sep='-', timespec='milliseconds')
    stamp = TIMESTAMP_PAT.sub('-', now)
    return '_'.join((title, stamp)) + '.png'

if __name__ == '__main__':

    import argparse
    import os

    parser = argparse.ArgumentParser(description='Generate image with ' + \
                                                 'Stable Diffusion.')
    parser.add_argument('--prompt', '-p', type=str, help='text prompt', 
                        required=True)
    parser.add_argument('--seed', '-s', type=int, default=1024)
    parser.add_argument('--guidance_scale', '-g', type=float, default=7.5)
    parser.add_argument('--num_inference_steps', '-n', type=int, default=50)
    parser.add_argument('--outdir', '-o', type=str, default='output')
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
    
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    image.save(os.sep.join((args.outdir, make_outfilename(args.prompt))))