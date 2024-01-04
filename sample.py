import argparse
import os
import einops

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import glob
import torch
import imageio
import matplotlib
from einops import rearrange
from torch.nn.functional import interpolate
from tqdm import tqdm
from torchvision.transforms import GaussianBlur
from video_diffusion_pytorch import Unet3DSPADE, Trainer, KarrasDiffusion
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_mode', type=str, default='dummy', choices=['echo', 'dummy'])
    parser.add_argument('--data_dir', type=str, default='samples')
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--diffusion_steps', type=int, default=64)
    parser.add_argument('--log_interval', type=int, default=2000)
    parser.add_argument('--train_num_steps', type=int, default=100000)
    parser.add_argument('--checkpoint', type=str, default="checkpoints/checkpoint.pt")
    parser.add_argument('--results_folder', type=str, default='results')
    return parser.parse_args()


def blur_sampleloc(sample_loc):
    """
    Blurs the sample location to make it more realistic
    sample_loc: (b, t, h, w, 2)
    """
    b, t = sample_loc.shape[0], sample_loc.shape[1]
    sample_loc = rearrange(sample_loc, "b t h w c -> (b t) c h w")
    sample_loc = GaussianBlur(3)(sample_loc)
    sample_loc = rearrange(sample_loc, "(b t) c h w -> b t h w c", b=b, t=t)
    return sample_loc


if __name__ == '__main__':
    # wandb = wandb.init(project="video-diffusion")
    args = get_args()
    results_folder = os.path.join(args.results_folder, f"{args.data_dir}_{args.num_frames}_{args.image_size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Unet3DSPADE(
        dim=32,
        dim_mults=(1, 2, 4),
    )

    diffusion = KarrasDiffusion(
        model,
        image_size=args.image_size,
        num_frames=args.num_frames,
        timesteps=args.diffusion_steps,  # number of steps
        loss_type='l1'  # L1 or L2
    ).to(device)

    trainer = Trainer(
        diffusion,
        folder=os.path.join('data', args.data_dir),
        data_mode=args.data_mode,  # 'video', 'echo' or 'dummy'
        train_batch_size=args.batch_size,
        train_lr=1e-4,
        save_and_sample_every=args.log_interval,
        train_num_steps=args.train_num_steps,  # total training steps
        gradient_accumulate_every=2,  # gradient accumulation steps
        ema_decay=0.995,  # exponential moving average decay
        amp=True,  # turn on mixed precision
        checkpoint=None,  # this is for resuming training, for now we don't need it
        results_folder=results_folder,
    )

    state_dict = torch.load(args.checkpoint, map_location=device)
    trainer.ema_model.load_state_dict(state_dict)

    if not os.path.exists(args.results_folder):
        os.makedirs(args.results_folder)

    for idx, seg_path in tqdm(enumerate(glob.glob(os.path.join(args.data_dir, "*")))):
        cond = imageio.imread(seg_path)
        cond = torch.from_numpy(cond)
        # resizes the condition to the same size as the image
        cond_small = interpolate(cond[None, None, ...].float(), size=(args.image_size, args.image_size),
                                 mode="nearest").squeeze()
        cond_small = torch.repeat_interleave(cond_small[None, ...], args.batch_size, dim=0).long()

        output = trainer.ema_model.sample(batch_size=args.batch_size, cond=cond_small.to(device))

        output = rearrange(output, 'b c t h w -> b t h w c')
        images = output.detach().cpu().numpy()
        images = (images * 255).astype('uint8')
        cond_small = cond_small.detach().cpu().numpy()
        cond_small_color = matplotlib.cm.get_cmap('viridis')(cond_small / cond.max())[:, :, :, :3]
        cond_small_color = einops.repeat(cond_small_color, 'b h w c -> b t h w c', t=args.num_frames)
        cond_small_color = (cond_small_color * 255).astype('uint8')
        for batch_idx, image in enumerate(images):
            image = np.concatenate([cond_small_color[batch_idx], image], axis=2)
            imageio.mimwrite(os.path.join(args.results_folder, f"{idx}_{batch_idx}.gif"), image, loop=0)