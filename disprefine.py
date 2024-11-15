import numpy as np
import h5py
import cv2
from DL import inference_single, inference_single_v2, InferDepthPro


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
    print("===> done !")


def save_numpy_array(arr, save_path):
    np.save(save_path, arr)
    print("===> done !")
 

if __name__ == "__main__":
    import os
    m_types = ['vits', 'vitb', 'vitl']
    # model_path="D:/william/checkpoints/depth-anything/v2/depth_anything_v2_vitb.pth"
    # subfolders = [11, 12, 19]
    top_dir = "/home/william/extdisk/data/test_mono/"
    predicted_save_dir = f"{top_dir}/output"
    os.makedirs(predicted_save_dir, exist_ok=True)
    subfolders = [f for f in os.listdir(top_dir) if os.path.isdir(os.path.join(top_dir, f))]
    subfiles = [f for f in os.listdir(top_dir) if os.path.isfile(os.path.join(top_dir, f)) and f.endswith(".png")]
    for type in m_types:
        print(f"processing {type} right now ...")
        model_path = f"/home/william/extdisk/checkpoints/depth-anything/depth_anything_v2_{type}.pth"
        for sub in subfiles:
            print(f"processing {sub} ...")
            # path = f"{top_dir}/{sub}/output_0222_agg_mask.mat"
            # disp = load_disp_from_mat(path, "out/s1out/cleanDispL")
            # print(disp.T.shape)
            # disp_save_path = f'{top_dir}/{sub}/output_0222_agg.npy'
            # save_numpy_array(disp.T, disp_save_path)

            # image_path = f"{top_dir}/{sub}/left.bmp"
            image_path = f"{top_dir}/{sub}"
            image = cv2.imread(image_path)
            # predicted = inference_single(image, model_path)
            predicted = inference_single_v2(image, model_path, encoder=f'{type}')
        # print(predicted)
        # print(predicted.shape)
        # save_path = "data/11/output_0222_DL.mat"  
        # save_numpy_array_to_matlab(predicted, save_path)
            
            predicted_save_path = f"{predicted_save_dir}/dam_v2_{type}_out_{sub}.npy"
            save_numpy_array(predicted, predicted_save_path)
    print("=====================> done! <=====================")  