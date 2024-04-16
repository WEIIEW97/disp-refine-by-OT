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

#pragma once

#include "EMD.h"
#include <string>

#include <Eigen/Core>
#include <Eigen/Dense>

namespace ot {
  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      RowMajorMatrixXd;

  typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      RowMajorMatrixXi;

  struct EMDCluster {
    RowMajorMatrixXd G;
    double cost;
    Eigen::ArrayXd alpha;
    Eigen::ArrayXd beta;
    int result_code;
  };

  struct EMDCluster1d {
    Eigen::ArrayXd G;
    Eigen::ArrayXd indices;
    double cost;
  };

  struct AlphaBetaCrater {
    Eigen::ArrayXd alpha;
    Eigen::ArrayXd beta;
  };

  std::string check_result(int result_code);
  std::vector<std::pair<int, int>> where(const ot::RowMajorMatrixXd& M,
                                         double thr);
  void indexing_op(ot::RowMajorMatrixXd& M,
                   const std::vector<std::pair<int, int>>& indices, double v);
  EMDCluster emd_c(Eigen::ArrayXd a, Eigen::ArrayXd b, RowMajorMatrixXd M,
                   uint64_t max_iter, int numThreads);
  EMDCluster1d emd_1d_sorted(Eigen::ArrayXd u_weights, Eigen::ArrayXd v_weights,
                             Eigen::ArrayXd u, Eigen::ArrayXd v,
                             std::string metric = "sqeuclidean",
                             double p = 1.f);

} // namespace ot