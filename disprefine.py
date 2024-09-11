import numpy as np
import h5py
import cv2
from DL import inference_single, inference_single_v2


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
    model_path="/home/william/Downloads/depth_anything_v2_vitb.pth"
    # subfolders = [11, 12, 19]
    top_dir = "/home/william/extdisk/data/disp-refine/feynman/"
    subfolders = [f for f in os.listdir(top_dir) if os.path.isdir(os.path.join(top_dir, f))]
    for sub in subfolders:
        # path = f"data/{sub}/output_0222_agg_mask.mat"
        # disp = load_disp_from_mat(path)
        # print(disp.T.shape)
        # disp_save_path = f'data/{sub}/output_0222_agg.npy'
        # save_numpy_array(disp.T, disp_save_path)

        image_path = f"{top_dir}/{sub}/left.bmp"
        image = cv2.imread(image_path)
        # predicted = inference_single(image, model_path)
        predicted = inference_single_v2(image, model_path, encoder='vitb')
    # print(predicted)
    # print(predicted.shape)
    # save_path = "data/11/output_0222_DL.mat"  
    # save_numpy_array_to_matlab(predicted, save_path)
        predicted_save_path = f"{top_dir}/{sub}/dam_v2_vitb_out.npy"
        save_numpy_array(predicted, predicted_save_path)
    print("=====================> done! <=====================")  