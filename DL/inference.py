import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm

# from .depth_anything.dpt import DPT_DINOv2
# from .depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from .depth_anything.dpt import DPT_DINOv2
from .depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str)
    parser.add_argument("--outdir", type=str, default="vis_depth")
    parser.add_argument("--model_path", type=str, default="depth_anything_vitl14.pth")

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


def preprocess(args, device):
    ## cannot do this in local files
    # model = DepthAnything.from_pretrained(args.model_path,  local_files_only=True)
    model = DPT_DINOv2("vitl", features=256, out_channels=[256, 512, 1024, 1024])
    ckpt = torch.load(args.model_path)
    model.load_state_dict(ckpt)
    model = model.to(device).eval()

    total_params = sum(param.numel() for param in model.parameters())
    print("Total parameters: {:.2f}M".format(total_params / 1e6))

    transform = Compose(
        [
            Resize(
                width=518,
                height=518,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method="lower_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ]
    )
    return model, transform


def preprocess(model_path, device):
    model = DPT_DINOv2("vitl", features=256, out_channels=[256, 512, 1024, 1024])
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt)
    model = model.to(device).eval()

    total_params = sum(param.numel() for param in model.parameters())
    print("Total parameters: {:.2f}M".format(total_params / 1e6))

    transform = Compose(
        [
            Resize(
                width=518,
                height=518,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method="lower_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ]
    )
    return model, transform


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


def inference_single(image, model_path, is_scale=False) -> np.ndarray:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model, transform = preprocess(model_path=model_path, device=DEVICE)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

    h, w = image.shape[:2]

    image = transform({"image": image})["image"]
    image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        depth = model(image)

    depth = F.interpolate(depth[None], (h, w), mode="bilinear", align_corners=False)[
        0, 0
    ]
    depth_raw = depth.cpu().numpy().astype(np.float32)
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0

    depth = depth.cpu().numpy().astype(np.uint8)
    if is_scale:
        return depth
    else:
        return depth_raw


if __name__ == "__main__":
    args = parse_args()
    inference(args)
