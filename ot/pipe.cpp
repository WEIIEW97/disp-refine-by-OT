#include "pipe.h"
#include "interp2d.h"
#include <memory>

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

vector<pair<int, int>> where(const ot::RowMajorMatrixXd& M, double thr) {
  vector<pair<int, int>> indices;

  for (int i = 0; i < M.rows(); ++i) {
    for (int j = 0; j < M.cols(); ++j) {
      if (M.coeffRef(i, j) <= thr) {
        indices.emplace_back(i, j);
      }
    }
  }

  return indices;
}

void indexing_op(ot::RowMajorMatrixXd& M, const vector<pair<int, int>>& indices,
                 double v) {
  for (const auto& index : indices) {
    M(index.first, index.second) = v;
  }
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

void build_pipeline(ot::RowMajorMatrixXd Xs, ot::RowMajorMatrixXd Xt,
                    double hollow_thr, std::string method) {
  auto insufficient_indices = where(Xs, hollow_thr);
  indexing_op(Xs, insufficient_indices, 0.f);
  indexing_op(Xt, insufficient_indices, 0.f);

  if (method == "normal") {
    auto norm_Xs_crate = normal_dist_normalizer(Xs);
    auto norm_Xt_crate = normal_dist_normalizer(Xt);
  }
}
