#include "pipe.h"
#include "interp2d.h"
#include <memory>

using namespace ot;

struct NormalDistCrate {
  ot::RowMajorMatrixXd norm;
  double mu;
  double sigma;
};

struct MinMaxDistCrate {
  ot::RowMajorMatrixXd norm;
  double minv;
  double maxv;
};

double* load_hdf5(std::string data_path, std::string dataset_name) {
  H5::H5File file(data_path, H5F_ACC_RDONLY);
  H5::DataSet dataset = file.openDataSet(dataset_name);
  H5::DataSpace dataspace = dataset.getSpace();
  hsize_t dims[1];
  int ndims = dataspace.getSimpleExtentDims(dims, nullptr);
  double* data = new double[dims[0]];
  H5::DataSpace memspace(1, dims);
  dataset.read(data, H5::PredType::NATIVE_DOUBLE, memspace, dataspace);

  return data;
}

void free_hdf5(double* data) { delete[] data; }

ot::RowMajorMatrixXd distribution_normalize(const ot::RowMajorMatrixXd& M) {
  auto mu = M.mean();
  auto sigma = std::sqrt((M.array() - mu).square().mean());

  return (M.array() - mu) / sigma;
}

ot::RowMajorMatrixXd distribution_normalize(const ot::RowMajorMatrixXd& M,
                                            double mu, double sigma) {
  return (M.array() - mu) / sigma;
}

ot::RowMajorMatrixXd distribution_minmax(const ot::RowMajorMatrixXd& M) {
  auto minv = M.minCoeff();
  auto maxv = M.maxCoeff();

  if (minv == maxv)
    return M;

  return (M.array() - minv) / (maxv - minv);
}

ot::RowMajorMatrixXd distribution_minmax(const ot::RowMajorMatrixXd& M,
                                         double minv, double maxv) {
  if (minv == maxv)
    return M;

  return (M.array() - minv) / (maxv - minv);
}

ot::RowMajorMatrixXd restore_from_normal(const ot::RowMajorMatrixXd& M,
                                         double mu, double sigma) {
  return (M.array() * sigma) + mu;
}

ot::RowMajorMatrixXd restore_from_minmax(const ot::RowMajorMatrixXd& M,
                                         double minv, double maxv) {
  return (M.array() * (maxv - minv) + minv);
}

NormalDistCrate normal_dist_normalizer(const ot::RowMajorMatrixXd& M) {
  auto mu_ = M.mean();
  auto sigma_ = std::sqrt((M.array() - mu_).square().mean());
  auto norm_ = distribution_normalize(M, mu_, sigma_);

  NormalDistCrate crate;
  crate.mu = mu_;
  crate.sigma = sigma_;
  crate.norm = norm_;

  return crate;
}

MinMaxDistCrate minmax_dist_normalizer(const ot::RowMajorMatrixXd& M) {
  auto minv_ = M.minCoeff();
  auto maxv_ = M.maxCoeff();
  auto norm_ = distribution_minmax(M, minv_, maxv_);

  MinMaxDistCrate crate;
  crate.maxv = maxv_;
  crate.minv = minv_;
  crate.norm = norm_;

  return crate;
}

ot::RowMajorMatrixXd load_hdf5_to_eigen_row_major(std::string data_path,
                                                  std::string dataset_name) {
  H5::H5File file(data_path, H5F_ACC_RDONLY);
  H5::DataSet dataset = file.openDataSet(dataset_name);

  // Get dataspace of the dataset
  H5::DataSpace dataspace = dataset.getSpace();

  // Get the dimension size of each dimension in the dataspace and rank
  int rank = dataspace.getSimpleExtentNdims();
  hsize_t dims_out[2];
  dataspace.getSimpleExtentDims(dims_out, nullptr);

  // Allocate the Eigen matrix of the correct size
  // Eigen::MatrixXd matrix(dims_out[0], dims_out[1]);
  ot::RowMajorMatrixXd matrix(dims_out[0], dims_out[1]);

  // Read the data
  dataset.read(matrix.data(), H5::PredType::NATIVE_DOUBLE);

  return matrix;
}

Eigen::MatrixXd load_hdf5_to_eigen_col_major(std::string data_path,
                                             std::string dataset_name) {
  H5::H5File file(data_path, H5F_ACC_RDONLY);
  H5::DataSet dataset = file.openDataSet(dataset_name);

  // Get dataspace of the dataset
  H5::DataSpace dataspace = dataset.getSpace();

  // Get the dimension size of each dimension in the dataspace and rank
  int rank = dataspace.getSimpleExtentNdims();
  hsize_t dims_out[2];
  dataspace.getSimpleExtentDims(dims_out, nullptr);

  // Allocate the Eigen matrix of the correct size
  Eigen::MatrixXd matrix(dims_out[0], dims_out[1]);
  // ot::RowMajorMatrixXd matrix(dims_out[0], dims_out[1]);

  // Read the data
  dataset.read(matrix.data(), H5::PredType::NATIVE_DOUBLE);

  return matrix;
}

ot::RowMajorMatrixXd recheck(const ot::RowMajorMatrixXd& Xs,
                             const ot::RowMajorMatrixXd& Xt, int kernel_size,
                             double alpha, double Ws, double Wt) {
  assert(Xs.rows() == Xt.rows() && Xs.cols() == Xt.cols() &&
         "Images must have the same dimensions.");
  assert(Xs.rows() % kernel_size == 0 &&
         "Image height must be divisible by kernel_size.");
  assert(Xs.cols() % kernel_size == 0 &&
         "Image width must be divisible by kernel_size.");

  ot::RowMajorMatrixXd Xsc = Xs;

  for (int i = 0; i <= Xs.rows() - kernel_size; i += kernel_size) {
    for (int j = 0; j <= Xs.cols() - kernel_size; j += kernel_size) {
      // Calculate the block mean and min max
      auto block_s = Xsc.block(i, j, kernel_size, kernel_size);
      auto block_t = Xt.block(i, j, kernel_size, kernel_size);
      double mu_s = block_s.mean();
      double mu_t = block_t.mean();
      double minv_s = block_s.minCoeff();
      double minv_t = block_t.minCoeff();
      double maxv_s = block_s.maxCoeff();
      double maxv_t = block_t.maxCoeff();

      bool cond = std::abs(mu_s - mu_t) <= alpha * mu_t &&
                  std::abs(minv_s - minv_t) <= alpha * minv_t &&
                  std::abs(maxv_s - maxv_t) <= alpha * maxv_t;

      if (!cond) {
        Xsc.block(i, j, kernel_size, kernel_size) = block_t * Wt + block_s * Ws;
      }
    }
  }

  std::cout << "===> rechecking is completed!" << std::endl;
  return Xsc;
}

void build_pipeline(ot::RowMajorMatrixXd Xs, ot::RowMajorMatrixXd Xt,
                    double hollow_thr, std::string method) {
  auto insufficient_indices = where(Xs, hollow_thr);
  indexing_op(Xs, insufficient_indices, 0.f);
  indexing_op(Xt, insufficient_indices, 0.f);

  if (method == "normal") {
    auto norm_Xs_crate = normal_dist_normalizer(Xs);
    auto norm_Xt_crate = normal_dist_normalizer(Xt);

    // add EMD transportation later
  } else if (method == "minmax") {
    auto norm_Xs_crate = minmax_dist_normalizer(Xs);
    auto norm_Xt_crate = minmax_dist_normalizer(Xt);
  }
}
