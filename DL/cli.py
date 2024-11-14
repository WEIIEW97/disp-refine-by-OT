import torch
import torch.nn.functional as F
import numpy as np

import cv2
from tqdm import tqdm

from .inference import preprocess, DPTV2_model_configs
from .depth_anything.dpt import DPT_DINOv2
from .depth_anything_v2.dpt import DepthAnythingV2
import os
import glob
import argparse
import matplotlib


def inference(args):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model, transform = preprocess(args, DEVICE)

    if os.path.isfile(args.img_path):
        if args.img_path.endswith("txt"):
            with open(args.img_path, "r") as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = os.listdir(args.img_path)
        filenames = [
            os.path.join(args.img_path, filename)
            for filename in filenames
            if not filename.startswith(".")
        ]
        filenames.sort()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.outdir + "/raw", exist_ok=True)
    os.makedirs(args.outdir + "/cm", exist_ok=True)

    for filename in tqdm(filenames):
        raw_image = cv2.imread(filename)
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0

        h, w = image.shape[:2]

        image = transform({"image": image})["image"]
        image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            depth = model(image)

        depth = F.interpolate(
            depth[None], (h, w), mode="bilinear", align_corners=False
        )[0, 0]
        depth_raw = depth.cpu().numpy().astype(np.uint16)
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0

        depth = depth.cpu().numpy().astype(np.uint8)

        if args.grayscale:
            depth_cm = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth_cm = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)

        filename = os.path.basename(filename)

        cv2.imwrite(
            os.path.join(
                args.outdir + "/raw", filename[: filename.rfind(".")] + "_depth_raw.png"
            ),
            depth_raw,
        )
        cv2.imwrite(
            os.path.join(
                args.outdir + "/cm", filename[: filename.rfind(".")] + "_depth_cm.png"
            ),
            depth_cm,
        )

    print("-> Done!")

def DPTV2_inference(args):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = DepthAnythingV2(**DPTV2_model_configs[args.encoder])
    ckpt = torch.load(args.model_path)
    model.load_state_dict(ckpt)
    model = model.to(DEVICE).eval()

    total_params = sum(param.numel() for param in model.parameters())
    print("Total parameters: {:.2f}M".format(total_params / 1e6))

    if os.path.isfile(args.img_path):
        if args.img_path.endswith("txt"):
            with open(args.img_path, "r") as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = glob.glob(os.path.join(args.img_path, "**/*"), recursive=True)

    os.makedirs(args.outdir, exist_ok=True)

    cmap = matplotlib.colormaps.get_cmap("Spectral_r")

    for k, filename in enumerate(filenames):
        print(f"Progress {k+1}/{len(filenames)}: {filename}")

        raw_image = cv2.imread(filename)
        depth = model.infer_image(raw_image, args.input_size)
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)

        if args.grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

        if args.pred_only:
            cv2.imwrite(
                os.path.join(
                    args.outdir,
                    os.path.splitext(os.path.basename(filename))[0] + ".png",
                ),
                depth,
            )
        else:
            split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
            combined_result = cv2.hconcat([raw_image, split_region, depth])

            cv2.imwrite(
                os.path.join(
                    args.outdir,
                    os.path.splitext(os.path.basename(filename))[0] + ".png",
                ),
                combined_result,
            )

    print("--> done!")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str)
    parser.add_argument(
        "--encoder", type=str, default="vitl", choices=["vits", "vitb", "vitl", "vitg"]
    )
    parser.add_argument("--outdir", type=str, default="vis_depth")
    parser.add_argument("--model_path", type=str, default="depth_anything_vitl14.pth")
    parser.add_argument("--input_size", type=int, default=518)
    parser.add_argument(
        "--pred_only",
        dest="pred_only",
        action="store_true",
        help="only display the prediction",
    )
    parser.add_argument(
        "--grayscale",
        dest="grayscale",
        action="store_true",
        help="do not apply colorful palette",
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # for DPTV1
    # inference(args)

    # for DPTV2
    DPTV2_inference(args)
