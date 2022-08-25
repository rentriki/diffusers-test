import json
import re

from datetime import datetime

import torch

from torch import autocast
from diffusers import StableDiffusionPipeline

MODEL_ID = "CompVis/stable-diffusion-v1-4"
DEVICE   = "cuda"

CHAR_PAT      = re.compile(r'[^0-9A-Za-z ]')
TIMESTAMP_PAT = re.compile(r'[:\.]')

def make_outfilename(prompt, maxlen=40):
    title = '-'.join(CHAR_PAT.sub('', prompt).lower().split())
    now = datetime.now().isoformat(sep='-', timespec='milliseconds')
    stamp = TIMESTAMP_PAT.sub('-', now)
    return '_'.join((title, stamp))[:maxlen]

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
    parser.add_argument('--mspaint', '-m', action='store_true')
    args = parser.parse_args()

    metadata = { 'creation_date' : datetime.now().isoformat(timespec='milliseconds'), 
                 'prompt' : args.prompt,
                 'seed' : args.seed,
                 'guidance_scale' : args.guidance_scale,
                 'num_inference_steps' : args.num_inference_steps
               }

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
    pngdir = os.sep.join((args.outdir, 'png'))
    jsondir = os.sep.join((args.outdir, 'json'))
    if not os.path.exists(pngdir):
        os.makedirs(pngdir)
    if not os.path.exists(jsondir):
        os.makedirs(jsondir)
    outfilename_base = make_outfilename(args.prompt)
    image_filename = os.sep.join((pngdir, outfilename_base + '.png'))
    json_filename = os.sep.join((jsondir, outfilename_base + '.json'))
    image.save(image_filename)
    with open(json_filename, mode='w') as json_outfile:
        pretty_json = json.dumps(metadata, sort_keys=True, indent=4)
        print(pretty_json)
        json_outfile.write(pretty_json)
    if args.mspaint:
        os.system(f'mspaint {image_filename}')