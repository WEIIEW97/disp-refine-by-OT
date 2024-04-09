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

#include "ot.h"
#include <H5Cpp.h>
#include <vector>
#include <utility>

using std::vector, std::pair;


Eigen::MatrixXd load_hdf5_to_eigen_col_major(std::string data_path,
                                             std::string dataset_name);
ot::RowMajorMatrixXd load_hdf5_to_eigen_row_major(std::string data_path,
                                                  std::string dataset_name);
vector<pair<int, int>> where(const ot::RowMajorMatrixXd& M, double thr);