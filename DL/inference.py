import argparse
import cv2
import numpy as np
import os
import glob
import matplotlib
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm

from .depth_anything.dpt import DPT_DINOv2
from .depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

from .depth_anything_v2.dpt import DepthAnythingV2
from .depth_pro import (
    create_model_and_transforms,
    load_rgb,
    DEFAULT_MONODEPTH_CONFIG_DICT,
)


DPTV2_model_configs = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },
    "vitg": {
        "encoder": "vitg",
        "features": 384,
        "out_channels": [1536, 1536, 1536, 1536],
    },
}


def _safe_torch_load(model_path, device):
    load_kwargs = {"map_location": device}
    try:
        return torch.load(model_path, weights_only=True, **load_kwargs)
    except TypeError:
        # Older Torch versions don't support `weights_only`.
        return torch.load(model_path, **load_kwargs)
    except Exception:
        # Some checkpoints contain non-tensor objects and need legacy loading.
        return torch.load(model_path, weights_only=False, **load_kwargs)


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            value = ckpt.get(key)
            if isinstance(value, dict):
                ckpt = value
                break
    if isinstance(ckpt, dict) and any(k.startswith("module.") for k in ckpt.keys()):
        ckpt = {k.replace("module.", "", 1): v for k, v in ckpt.items()}
    return ckpt


def preprocess(args, device):
    ## cannot do this in local files
    # model = DepthAnything.from_pretrained(args.model_path,  local_files_only=True)
    model = DPT_DINOv2("vitl", features=256, out_channels=[256, 512, 1024, 1024])
    ckpt = _extract_state_dict(_safe_torch_load(args.model_path, device))
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
    ckpt = _extract_state_dict(_safe_torch_load(model_path, device))
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


def inference_single_v2(
    image, model_path, encoder="vitl", input_size=518, is_scale=False
):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = DepthAnythingV2(**DPTV2_model_configs[encoder])
    ckpt = _extract_state_dict(_safe_torch_load(model_path, DEVICE))
    model.load_state_dict(ckpt)
    model = model.to(DEVICE).eval()

    total_params = sum(param.numel() for param in model.parameters())
    print("Total parameters: {:.2f}M".format(total_params / 1e6))

    depth = model.infer_image(image, input_size)
    depth_raw = depth.astype(np.float32)
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0

    depth = depth.astype(np.uint8)
    if is_scale:
        return depth
    else:
        return depth_raw


def _resolve_torch_device(device: str = "auto") -> torch.device:
    """Resolve the Torch device from an explicit user choice."""
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is not available. "
                "Check `nvidia-smi`, CUDA driver/runtime, and your PyTorch CUDA build."
            )
        return torch.device(device)

    if device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available on this machine.")
        return torch.device("mps")

    if device == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unsupported device '{device}'. Use auto/cpu/cuda/cuda:N/mps.")


class InferDAM:
    def __init__(self, device: str = "auto"):
        self.device = _resolve_torch_device(device)

    def initialize(
        self, model_path, is_scale=False, mode="v2", encoder="vitl", input_size=518
    ):

        self.is_scale = is_scale
        self.mode = mode

        assert mode in (
            "v1",
            "v2",
        ), f"{mode} is not supported! only support v1 and v2 now!"

        if mode == "v2":
            assert (
                encoder is not None and input_size is not None
            ), "you should specify encoder and input_size arguments."

        if mode == "v1":
            self.model, self.transform = preprocess(
                model_path=model_path, device=self.device
            )
        else:
            self.model = DepthAnythingV2(**DPTV2_model_configs[encoder])
            ckpt = _extract_state_dict(_safe_torch_load(model_path, self.device))
            self.model.load_state_dict(ckpt)
            self.input_size = input_size


    def _infer_v1(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        h, w = image.shape[:2]

        image = self.transform({"image": image})["image"]
        image = torch.from_numpy(image).unsqueeze(0).to(self.device)
 
        with torch.no_grad():
            depth = self.model(image)

        depth = F.interpolate(
            depth[None], (h, w), mode="bilinear", align_corners=False
        )[0, 0]
        depth_raw = depth.cpu().numpy().astype(np.float32)
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0

        depth = depth.cpu().numpy().astype(np.uint8)
        if self.is_scale:
            return depth
        else:
            return depth_raw

    def _infer_v2(self, image_path):
        image = cv2.imread(image_path)
        self.model = self.model.to(self.device).eval()

        depth = self.model.infer_image(image, self.input_size)
        depth_raw = depth.astype(np.float32)
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0

        depth = depth.astype(np.uint8)
        if self.is_scale:
            return depth
        else:
            return depth_raw

    def get_params_count(self):
        total_params = sum(param.numel() for param in self.model.parameters())
        print("Total parameters: {:.2f}M".format(total_params / 1e6))

    def infer(self, image_path):
        if self.mode == "v1":
            return self._infer_v1(image_path)
        else:
            return self._infer_v2(image_path)


class InferDepthPro:
    def __init__(self):
        self.device = _resolve_torch_device()

    def initialize(self, model_path, is_half=True):
        preci = torch.half if is_half else torch.float32
        config = DEFAULT_MONODEPTH_CONFIG_DICT
        config.checkpoint_uri = model_path
        self.model, self.transform = create_model_and_transforms(
            config=config,
            device=self.device,
            precision=preci,
        )
        self.model.eval()

    def infer(self, image_path):
        image, _, f_px = load_rgb(image_path)
        print(f"image max value is {image.max()}, min value is {image.min()}")
        pred = self.model.infer(self.transform(image), f_px=f_px)
        depth = pred["depth"]  # Depth in [m].
        focallenth_px = pred["focallength_px"]  # Focal length in pixels.

        return depth, focallenth_px

    def get_device_info(self):
        print(f"You are running at {self.device}!")
