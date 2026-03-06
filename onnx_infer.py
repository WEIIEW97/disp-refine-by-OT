import onnxruntime as ort
import onnx
import numpy as np
import cv2
import os

from DL.depth_anything.util.transform import load_image
from tqdm import tqdm

def save_numpy_array(arr, save_path):
    np.save(save_path, arr)

def make_colormap(m:np.ndarray) -> np.ndarray:
    m_norm = cv2.normalize(m, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    m_cm = cv2.applyColorMap(m_norm, cv2.COLORMAP_MAGMA)
    return m_cm

def main():
    onnx_model = onnx.load('/home/william/extdisk/checkpoints/depth-anything/model_q4f16.onnx')
    onnx.checker.check_model(onnx_model)
    session = ort.InferenceSession(
        '/home/william/extdisk/checkpoints/depth-anything/model_q4f16.onnx',
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    save_path = "/home/william/extdisk/data/realsense-D455_depth_image/damv2_q4f16"
    os.makedirs(save_path, exist_ok=True)

    image_path = "/home/william/extdisk/data/realsense-D455_depth_image/color"

    file_names = [f for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]

    for file_name in tqdm(file_names):
        full_path = os.path.join(image_path, file_name)
        image, (orig_h, orig_w) = load_image(full_path)
        depth_raw = session.run(None, {"pixel_values": image})[0]
        depth = cv2.resize(depth_raw[0, :, :], (orig_w, orig_h))
        predicted_save_path = f"{save_path}/{file_name}_q4f16.npy"
        save_numpy_array(depth, predicted_save_path)
        dam_cm = make_colormap(depth)

        # cv2.imwrite(f"{dp_save_path}/dp_12_f_{est_focal}.jpg", dp_cm)
        cv2.imwrite(f"{save_path}/{file_name}_q4f16_cm.jpg", dam_cm)
    print("done!")


if __name__ == "__main__":
    main()



