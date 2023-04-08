import argparse
import importlib
from v_diffusion import make_beta_schedule
from copy import deepcopy
import torch
from train_utils import make_visualization
import cv2

def make_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", type=str, default='celeba_u')
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/celeba/base_6/checkpoint.pt")
    parser.add_argument("--out_file", help="Path to image.", type=str, default="./images/celeba_u_6.png")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--diffusion", help="Diffusion model.", type=str, default="GaussianDiffusion")
    parser.add_argument("--time_scale", help="Diffusion time scale.", type=int, default=1)
    parser.add_argument("--clipped_sampling", help="Use clipped sampling mode.", type=bool, default=True)
    parser.add_argument("--clipping_value", help="Noise clipping value.", type=float, default=1.2)
    parser.add_argument("--eta", help="Amount of random noise in clipping sampling mode(recommended "
                                      "non-zero values only for not distilled model).", type=float, default=0)
    return parser

def make_diffusion(args, model, n_timestep, time_scale, device):
    betas = make_beta_schedule("cosine", cosine_s=8e-3, n_timestep=n_timestep).to(device)
    M = importlib.import_module("v_diffusion")
    D = getattr(M, args.diffusion)
    sampler = "ddpm"
    if args.clipped_sampling:
        sampler = "clipped"
    return D(model, betas, time_scale=time_scale, sampler=sampler)

def sample_images(args, make_model):
    print(f"sample_images()...")
    device = torch.device("cuda")
    model = make_model().to(device)
    print(f"model = make_model().to({device})")

    print(f"Load ckpt: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint)
    model.load_state_dict(ckpt["G"])
    n_timesteps = ckpt["n_timesteps"]//args.time_scale
    time_scale = ckpt["time_scale"]*args.time_scale
    print(f"  ckpt['n_timesteps']: {ckpt['n_timesteps']}")
    print(f"  ckpt['time_scale'] : {ckpt['time_scale']}")
    print(f"  args.time_scale    : {args.time_scale}")
    print(f"  final n_timesteps  : {n_timesteps}")
    print(f"  final time_scale   : {time_scale}")
    del ckpt
    print("Model loaded.")

    teacher_diffusion = make_diffusion(args, model, n_timesteps, time_scale, device)
    image_size = deepcopy(model.image_size)
    image_size[0] = args.batch_size

    img = make_visualization(teacher_diffusion, device, image_size, need_tqdm=True,
                             eta=args.eta, clip_value=args.clipping_value)
    if img.shape[2] == 1:
        img = img[:, :, 0]
    cv2.imwrite(args.out_file, img)

    print("sample_images()...Finished.")

def main():
    parser = make_argument_parser()
    args = parser.parse_args()
    print(args)

    M = importlib.import_module(args.module)
    print(f"make_model() function is defined in args.module: {args.module}")
    make_model = getattr(M, "make_model")

    sample_images(args, make_model)
if __name__ == '__main__':
    main()
