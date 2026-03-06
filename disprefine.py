import numpy as np
import h5py
import cv2
from DL import inference_single, inference_single_v2, InferDepthPro, InferDAM
from tqdm import tqdm


def image_resize(img, sz):
    return cv2.resize(img, sz, interpolation=cv2.INTER_LANCZOS4)


def load_disp_from_mat(path, dataset_name="data"):
    """load left disparity from .mat file."""
    with h5py.File(path, "r") as file:
        print("Keys: %s" % file.keys())
        dataset = file[dataset_name]
        disp = dataset[:]

    return disp


def save_numpy_array_to_matlab(arr, save_path):
    with h5py.File(save_path, 'w') as file:
        file.create_dataset('data', data=arr)
    # print("===> done !")


def save_numpy_array(arr, save_path):
    np.save(save_path, arr)
    # print("===> done !")


def make_colormap(m:np.ndarray) -> np.ndarray:
    m_norm = cv2.normalize(m, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    m_cm = cv2.applyColorMap(m_norm, cv2.COLORMAP_MAGMA)
    return m_cm

if __name__ == "__main__":
    import os
    m_types = ['vits', 'vitb', 'vitl']
    # model_path="D:/william/checkpoints/depth-anything/v2/depth_anything_v2_vitb.pth"
    # subfolders = [11, 12, 19]
    # top_dir = "/home/william/extdisk/data/test_mono/raw/"
    # predicted_save_dir = f"{top_dir}/output"
    # os.makedirs(predicted_save_dir, exist_ok=True)
    # subfolders = [f for f in os.listdir(top_dir) if os.path.isdir(os.path.join(top_dir, f))]
    # subfiles = [f for f in os.listdir(top_dir) if os.path.isfile(os.path.join(top_dir, f)) and f.endswith(".png")]
    # for type in m_types:
    #     print(f"processing {type} right now ...")
    #     model_path = f"/home/william/extdisk/checkpoints/depth-anything/depth_anything_v2_{type}.pth"
    #     for sub in subfiles:
    #         print(f"processing {sub} ...")
    #         # path = f"{top_dir}/{sub}/output_0222_agg_mask.mat"
    #         # disp = load_disp_from_mat(path, "out/s1out/cleanDispL")
    #         # print(disp.T.shape)
    #         # disp_save_path = f'{top_dir}/{sub}/output_0222_agg.npy'
    #         # save_numpy_array(disp.T, disp_save_path)

    #         # image_path = f"{top_dir}/{sub}/left.bmp"
    #         image_path = f"{top_dir}/{sub}"
    #         image = cv2.imread(image_path)
    #         # predicted = inference_single(image, model_path)
    #         predicted = inference_single_v2(image, model_path, encoder=f'{type}')
    #     # print(predicted)
    #     # print(predicted.shape)
    #     # save_path = "data/11/output_0222_DL.mat"  
    #     # save_numpy_array_to_matlab(predicted, save_path)
            
    #         predicted_save_path = f"{predicted_save_dir}/dam_v2_{type}_out_{sub}.npy"
    #         save_numpy_array(predicted, predicted_save_path)

    # dp_model_path="/home/william/extdisk/checkpoints/depth-pro/depth_pro.pt"
    dam_model_path="/home/william/extdisk/checkpoints/depth-anything/depth_anything_v2_vitl.pth"
    # dp_save_path = "/home/william/extdisk/data/mono/dp_out"
    dam_save_path = "/home/william/Downloads/ot_data"
    # os.makedirs(dp_save_path, exist_ok=True)
    os.makedirs(dam_save_path, exist_ok=True)

    # dp_infer = InferDepthPro()
    dam_infer = InferDAM(device="cuda:0")
    # dp_infer.initialize(model_path=dp_model_path, is_half=False)
    dam_infer.initialize(model_path=dam_model_path, encoder='vitl')
    # dp_infer.get_device_info()
    dam_infer.get_params_count()
    # image_path = "/home/william/extdisk/data/mono/left.png"
    # depth, est_focal = dp_infer.infer(image_path)
    # depth = depth.detach().cpu().numpy()
    # print(f"out depth shape is {depth.shape}")
    # predicted_save_path = f"{dp_save_path}/dp_12_f_{est_focal}.npy"
    # save_numpy_array(depth, predicted_save_path)
    # print(f"estimated focal length is: {est_focal}")

    # image_path = "/home/william/Codes/pc-optimize/data/color"

    # file_names = [f for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]

    # for file_name in tqdm(file_names):
    #     full_path = os.path.join(image_path, file_name)
    #     depth_dam = dam_infer.infer(full_path)
    #     predicted_save_path = f"{dam_save_path}/{file_name}_vitl.npy"
    #     save_numpy_array(depth_dam, predicted_save_path)

    #     # dp_cm = make_colormap(depth)
    #     dam_cm = make_colormap(depth_dam)

    #     # cv2.imwrite(f"{dp_save_path}/dp_12_f_{est_focal}.jpg", dp_cm)
    #     cv2.imwrite(f"{dam_save_path}/{file_name}_vitl_cm.jpg", dam_cm)

    
    img_path="/home/william/Downloads/ot_data/l_00004.png"
    depth_dam = dam_infer.infer(img_path)
    save_numpy_array(depth_dam, f"{dam_save_path}/l_00004_vitl.npy")
    dam_cm = make_colormap(depth_dam)
    cv2.imwrite(f"{dam_save_path}/l_00004_vitl_cm.jpg", dam_cm)
    
    print("=====================> done! <=====================")  
