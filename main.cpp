#include <iostream>
#include <opencv2/opencv.hpp>
// #include <opencv2/core/eigen.hpp>
#include "ot/ot.h"
#include "ot/pipe.h"

int main() {
  std::string agg_path =
      "/algdata01/wei.wei/codes/disp-refine-by-DL/data/11/output_0222_agg.npy";
  std::string dl_path =
      "/algdata01/wei.wei/codes/disp-refine-by-DL/data/11/output_0222_DL.npy";
  std::string dataset_name = "data";

  // auto agg = load_hdf5_to_eigen_row_major(agg_path, dataset_name);
  auto agg = load_npy_to_eigen_row_major(agg_path);
  std::cout << agg.rows() << ", " << agg.cols() << std::endl;
  return 0;
}