/*
 * Copyright (c) 2022-2023, William Wei. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cmath>
#include <Eigen/Core>
#include <vector>

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    RowMajorMatrixXd;

inline Eigen::ArrayXd unif(int n) {
  // n has to be greater than 0
  return Eigen::ArrayXd::Ones(n) / n;
}

inline RowMajorMatrixXd euclidean_distances(const RowMajorMatrixXd& X,
                                     const RowMajorMatrixXd& Y,
                                     bool squared = false) {
  Eigen::VectorXd X_sq = X.rowwise().squaredNorm();
  Eigen::VectorXd Y_sq = Y.rowwise().squaredNorm();

  RowMajorMatrixXd dot = -2.0 * X * Y.transpose();
  dot.colwise() += X_sq;
  dot.rowwise() += Y_sq.transpose();

  dot = dot.cwiseMax(0.0);

  if (!squared) {
    dot = dot.cwiseSqrt();
  }

  if (&X == &Y) {
    dot.diagonal().setZero();
  }

  return dot;
}

inline RowMajorMatrixXd minkowski_distance(const RowMajorMatrixXd& X,
                                    const RowMajorMatrixXd& Y, double p) {
  assert(X.cols() == Y.cols() &&
         "X and Y must have the same number of columns.");
  RowMajorMatrixXd dist(X.rows(), Y.rows());

  for (int i = 0; i < X.rows(); ++i) {
    for (int j = 0; j < Y.rows(); ++j) {
      auto diff = (X.row(i) - Y.row(j)).array().abs().pow(p);
      dist(i, j) = std::pow(diff.sum(), 1.0 / p);
    }
  }

  return dist;
}

inline RowMajorMatrixXd dist(const RowMajorMatrixXd& x1, const RowMajorMatrixXd& x2,
                      const std::string& metric = "sqeuclidean",
                      double p = 2.0f) {
  if (metric == "sqeuclidean")
    return euclidean_distances(x1, x2, true);
  if (metric == "euclidean")
    return euclidean_distances(x1, x2, false);
  else
    return minkowski_distance(x1, x2, p);
}