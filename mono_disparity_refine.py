import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from rectified_flow import (
    Camera,
    RealFusionDataset,
    TinyUNet,
    align_mono_disp_to_stereo,
    confidence_from_cost_volume,
    depth_from_disp,
    disp_from_depth,
    load_disp_any,
    load_left_rgb,
    make_costvol_from_disp,
    mono_refine_loss,
    mono_refine_sample,
    mono_to_disparity,
    set_seed,
)


def build_real_dataset(args):
    left_path = os.path.join(args.data_dir, args.left_name)
    right_path = os.path.join(args.data_dir, args.right_name)
    mono_path = os.path.join(args.data_dir, args.mono_name)
    disp_path = os.path.join(args.data_dir, args.disp_name)
    gt_disp_path = os.path.join(args.data_dir, args.gt_disp_name)

    img = load_left_rgb(left_path)
    img_r = load_left_rgb(right_path)
    dm = np.load(mono_path).astype(np.float32)
    disp_s = load_disp_any(disp_path, args.disp_scale).astype(np.float32)

    print(f"mono mode: {'inverse-depth' if args.mono_is_inverse else 'depth'}")
    print(
        f"disp_s stats: dtype={disp_s.dtype}, min={float(np.min(disp_s)):.4f}, "
        f"max={float(np.max(disp_s)):.4f}, scale={args.disp_scale}"
    )

    if dm.shape != disp_s.shape:
        dm = cv2.resize(
            dm, (disp_s.shape[1], disp_s.shape[0]), interpolation=cv2.INTER_LINEAR
        )

    if args.real_w > 0 and args.real_h > 0:
        img_hwc = np.transpose(img, (1, 2, 0))
        img_hwc = cv2.resize(
            img_hwc, (args.real_w, args.real_h), interpolation=cv2.INTER_LINEAR
        )
        img = np.transpose(img_hwc, (2, 0, 1)).astype(np.float32)

        img_r_hwc = np.transpose(img_r, (1, 2, 0))
        img_r_hwc = cv2.resize(
            img_r_hwc, (args.real_w, args.real_h), interpolation=cv2.INTER_LINEAR
        )
        img_r = np.transpose(img_r_hwc, (2, 0, 1)).astype(np.float32)

        dm = cv2.resize(dm, (args.real_w, args.real_h), interpolation=cv2.INTER_LINEAR)
        disp_s = cv2.resize(
            disp_s, (args.real_w, args.real_h), interpolation=cv2.INTER_LINEAR
        )

    h, w = disp_s.shape
    disp_max = (
        args.disp_max
        if args.disp_max is not None
        else max(32.0, float(np.nanpercentile(disp_s, 99.5) * 1.5))
    )
    valid = (disp_s > 1e-6).astype(np.float32)
    costvol = make_costvol_from_disp(img, disp_s, args.D, disp_max)
    disp_r = disp_s.copy()

    gt = np.zeros_like(disp_s, dtype=np.float32)
    if os.path.isfile(gt_disp_path):
        gt_disp = load_disp_any(gt_disp_path, args.gt_disp_scale).astype(np.float32)
        if gt_disp.shape != disp_s.shape:
            gt_disp = cv2.resize(
                gt_disp,
                (disp_s.shape[1], disp_s.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        gt_valid = gt_disp > 1e-6
        gt[gt_valid] = (
            args.fx * args.baseline / np.clip(gt_disp[gt_valid], 1e-6, None)
        ).astype(np.float32)

    sample = (
        torch.from_numpy(img),
        torch.from_numpy(img_r),
        torch.from_numpy(dm).unsqueeze(0),
        torch.from_numpy(disp_s).unsqueeze(0),
        torch.from_numpy(costvol),
        torch.from_numpy(disp_r).unsqueeze(0),
        torch.from_numpy(gt).unsqueeze(0),
        torch.from_numpy(valid).unsqueeze(0),
    )

    train_ds = RealFusionDataset(sample=sample, n=max(args.train_n, args.batch))
    test_ds = RealFusionDataset(sample=sample, n=max(args.test_n, 1))

    return train_ds, test_ds, h, w


def save_viz(args, ep, img, disp_gt, disp_s, disp_m, disp_f, c, valid):
    try:
        import matplotlib.pyplot as plt

        b0 = 0
        I = img[b0].detach().cpu().permute(1, 2, 0).numpy()
        GT = disp_gt[b0, 0].detach().cpu().numpy()
        DS = disp_s[b0, 0].detach().cpu().numpy()
        DM = disp_m[b0, 0].detach().cpu().numpy()
        DH = disp_f[b0, 0].detach().cpu().numpy()
        C = c[b0, 0].detach().cpu().numpy()

        def norm_vis(x):
            x = x.copy()
            x[x <= 0] = np.nan
            vmin = np.nanpercentile(x, 2)
            vmax = np.nanpercentile(x, 98)
            x = np.clip((x - vmin) / (vmax - vmin + 1e-6), 0, 1)
            return np.nan_to_num(x, nan=0.0)

        fig = plt.figure(figsize=(14, 7))
        ax = plt.subplot(2, 4, 1)
        ax.set_title("RGB")
        ax.imshow(I)
        ax.axis("off")

        has_gt = np.any(GT > 0.0)
        ax = plt.subplot(2, 4, 2)
        ax.set_title("GT disparity" if has_gt else "GT disparity (N/A)")
        ax.imshow(norm_vis(GT) if has_gt else np.zeros_like(DH))
        ax.axis("off")

        ax = plt.subplot(2, 4, 3)
        ax.set_title("Stereo disparity")
        ax.imshow(norm_vis(DS))
        ax.axis("off")

        ax = plt.subplot(2, 4, 4)
        ax.set_title("Mono disparity")
        ax.imshow(norm_vis(DM))
        ax.axis("off")

        ax = plt.subplot(2, 4, 5)
        ax.set_title("Confidence")
        ax.imshow(np.clip(C, 0, 1))
        ax.axis("off")

        ax = plt.subplot(2, 4, 6)
        ax.set_title("Valid mask")
        ax.imshow(valid[b0, 0].detach().cpu().numpy(), cmap="gray")
        ax.axis("off")

        ax = plt.subplot(2, 4, 7)
        ax.set_title("Refined disparity")
        ax.imshow(norm_vis(DH))
        ax.axis("off")

        ax = plt.subplot(2, 4, 8)
        ax.set_title("|Refined - GT|" if has_gt else "N/A")
        ax.imshow(np.abs(DH - GT) if has_gt else np.zeros_like(DH), cmap="magma")
        ax.axis("off")

        plt.tight_layout()
        out_path = os.path.join(args.save_dir, f"viz_ep{ep + 1:03d}.png")
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {out_path}")
    except Exception as e:
        print(f"Visualization skipped: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Mono disparity refinement guided by stereo confidence"
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--train_n", type=int, default=256)
    parser.add_argument("--test_n", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--D", type=int, default=96)
    parser.add_argument("--disp_max", type=float, default=None)
    parser.add_argument("--save_dir", type=str, default="outputs_mono_refine")

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--left_name", type=str, default="l_00004.png")
    parser.add_argument("--right_name", type=str, default="r_00004.png")
    parser.add_argument("--mono_name", type=str, default="l_00004_vitl.npy")
    parser.add_argument("--disp_name", type=str, default="nvpd_00004.png")
    parser.add_argument("--gt_disp_name", type=str, default="d_00004.exr")
    parser.add_argument("--disp_scale", type=float, default=64.0)
    parser.add_argument("--gt_disp_scale", type=float, default=1.0)

    parser.add_argument("--fx", type=float, default=420.0)
    parser.add_argument("--fy", type=float, default=420.0)
    parser.add_argument("--baseline", type=float, default=0.10)
    parser.add_argument("--real_w", type=int, default=480)
    parser.add_argument("--real_h", type=int, default=272)
    parser.add_argument("--mono_is_inverse", action="store_true")

    parser.add_argument("--residual_clip", type=float, default=1.0)
    parser.add_argument("--refine_conf_power", type=float, default=1.2)
    parser.add_argument("--refine_stereo_blend_power", type=float, default=0.8)
    parser.add_argument("--photo_weight", type=float, default=0.0)
    parser.add_argument("--photo_ssim_weight", type=float, default=0.85)
    parser.add_argument("--photo_lowconf_power", type=float, default=1.5)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)
    torch.backends.cudnn.enabled = False

    device = torch.device(
        args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    )

    global cv2
    import cv2

    train_ds, test_ds, h, w = build_real_dataset(args)
    cam = Camera(
        fx=args.fx,
        fy=args.fy,
        cx=w * 0.5,
        cy=h * 0.5,
        baseline=args.baseline,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, drop_last=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=min(args.batch, len(test_ds)), shuffle=False
    )

    model = TinyUNet(in_ch=7, base=48).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    print(f"Device: {device}")
    print(
        f"Train: {len(train_ds)} | Test: {len(test_ds)} | D={args.D} | mono-refine=True"
    )

    for ep in range(args.epochs):
        model.train()
        running = {}
        for it, batch in enumerate(train_loader):
            img, img_r, dm, disp_s, costvol, disp_r, gt, valid = [
                b.to(device) for b in batch
            ]

            c = confidence_from_cost_volume(costvol, disp_s, valid, disp_r)
            disp_m_raw = mono_to_disparity(
                dm, cam.fx, cam.baseline, args.mono_is_inverse
            )
            disp_m = align_mono_disp_to_stereo(disp_m_raw, disp_s, c, valid)

            loss, stats = mono_refine_loss(
                model=model,
                img=img,
                img_r=img_r,
                disp_m=disp_m,
                disp_s=disp_s,
                c=c,
                valid=valid,
                conf_power=args.refine_conf_power,
                residual_clip=args.residual_clip,
                photo_weight=args.photo_weight,
                photo_ssim_weight=args.photo_ssim_weight,
                photo_lowconf_power=args.photo_lowconf_power,
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            for k, v in stats.items():
                running[k] = running.get(k, 0.0) + v

            if (it + 1) % 50 == 0:
                denom = 50
                keys = [
                    "loss",
                    "loss_sup",
                    "loss_anchor",
                    "loss_smooth",
                    "loss_photo",
                    "c_mean",
                ]
                msg = " | ".join([f"{k}:{running[k]/denom:.4f}" for k in keys])
                print(
                    f"[ep {ep+1}/{args.epochs}] iter {it+1}/{len(train_loader)} :: {msg}"
                )
                running = {}

        model.eval()
        with torch.no_grad():
            batch = next(iter(test_loader))
            img, img_r, dm, disp_s, costvol, disp_r, gt, valid = [
                b.to(device) for b in batch
            ]

            c = confidence_from_cost_volume(costvol, disp_s, valid, disp_r)
            disp_m_raw = mono_to_disparity(
                dm, cam.fx, cam.baseline, args.mono_is_inverse
            )
            disp_m = align_mono_disp_to_stereo(disp_m_raw, disp_s, c, valid)
            disp_f = mono_refine_sample(
                model=model,
                img=img,
                disp_m=disp_m,
                disp_s=disp_s,
                c=c,
                valid=valid,
                conf_power=args.refine_conf_power,
                residual_clip=args.residual_clip,
                stereo_blend_power=args.refine_stereo_blend_power,
            )

            disp_gt = disp_from_depth(torch.clamp(gt, min=1e-6), cam.fx, cam.baseline)
            valid_gt = (disp_gt > 0.0).float()
            abs_rel_f = torch.mean(
                valid_gt * torch.abs(disp_f - disp_gt) / torch.clamp(disp_gt, min=1e-6)
            ) / (valid_gt.mean() + 1e-6)
            abs_rel_s = torch.mean(
                valid_gt * torch.abs(disp_s - disp_gt) / torch.clamp(disp_gt, min=1e-6)
            ) / (valid_gt.mean() + 1e-6)
            abs_rel_m = torch.mean(
                valid_gt * torch.abs(disp_m - disp_gt) / torch.clamp(disp_gt, min=1e-6)
            ) / (valid_gt.mean() + 1e-6)

            mask_stereo = (valid_gt > 0.5) & (valid > 0.5)
            mask_hole = (valid_gt > 0.5) & (valid <= 0.5)
            abs_rel_f_stereo = (
                torch.abs(disp_f - disp_gt) / torch.clamp(disp_gt, min=1e-6)
            )[mask_stereo].mean()
            abs_rel_s_stereo = (
                torch.abs(disp_s - disp_gt) / torch.clamp(disp_gt, min=1e-6)
            )[mask_stereo].mean()
            abs_rel_f_hole = (
                torch.abs(disp_f - disp_gt) / torch.clamp(disp_gt, min=1e-6)
            )[mask_hole].mean()
            abs_rel_s_hole = (
                torch.abs(disp_s - disp_gt) / torch.clamp(disp_gt, min=1e-6)
            )[mask_hole].mean()

            print(
                f"[ep {ep+1}] real disp abs-rel: fused={abs_rel_f.item():.4f} | stereo={abs_rel_s.item():.4f} | mono_aligned={abs_rel_m.item():.4f}"
            )
            print(
                f"[ep {ep+1}] split metrics: stereo_valid_ratio={mask_stereo.float().mean().item():.4f} | "
                f"hole_ratio={mask_hole.float().mean().item():.4f} | fused@stereo={abs_rel_f_stereo.item():.4f} | "
                f"stereo@stereo={abs_rel_s_stereo.item():.4f} | fused@hole={abs_rel_f_hole.item():.4f} | "
                f"stereo@hole={abs_rel_s_hole.item():.4f} | conf_mean={c.mean().item():.4f}"
            )

            dhat = depth_from_disp(torch.clamp(disp_f, min=1e-6), cam.fx, cam.baseline)
            save_viz(args, ep, img, disp_gt, disp_s, disp_m, disp_f, c, valid)
            np.save(
                os.path.join(args.save_dir, "fused_depth.npy"),
                dhat[0, 0].detach().cpu().numpy().astype(np.float32),
            )
            np.save(
                os.path.join(args.save_dir, "fused_disparity.npy"),
                disp_f[0, 0].detach().cpu().numpy().astype(np.float32),
            )
            np.save(
                os.path.join(args.save_dir, "confidence.npy"),
                c[0, 0].detach().cpu().numpy().astype(np.float32),
            )

    ckpt_path = os.path.join(args.save_dir, "tiny_unet_mono_disparity_refine.pt")
    torch.save({"model": model.state_dict(), "args": vars(args)}, ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
