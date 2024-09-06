/*
 * Copyright (c) 2022-2024, William Wei. All rights reserved.
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

#include <iostream>
#include <filesystem>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <libsgm.h>
#include <boost/program_options.hpp>

#include "sample_common.h"

namespace fs = std::filesystem;
namespace po = boost::program_options;
using namespace std;

int main(int argc, char** argv) {

  string left_path, right_path, out_path;
  int max_d, p1, p2, n_path, cencus_type;
  float uniqueness, scale_ratio;

  po::options_description desc("Allowed options");
  desc.add_options()
  ("left_path,l", po::value<string>(&left_path)->default_value(""), "left image path.")
  ("right_path,r", po::value<string>(&right_path)->default_value(""), "right image path.")
  ("out_path,o", po::value<string>(&out_path)->default_value(""), "output u16 path.")
  ("d_level,d", po::value<int>(&max_d)->default_value(256), "maximum disparity level.")
  ("p_1", po::value<int>(&p1)->default_value(10), "penalty 1.")
  ("p_2", po::value<int>(&p2)->default_value(120), "penalty 2.")
  ("uniq", po::value<float>(&uniqueness)->default_value(0.f), "penalty 2.")
  ("resize", po::value<float>(&scale_ratio)->default_value(1.0f), "resize ratio")
  ("path", po::value<int>(&n_path)->default_value(8), "direction choices. [4 or 8].")
  ("cencus", po::value<int>(&cencus_type)->default_value(1), "cencus transformation type. [1 or 2].");

  cv::Mat I1 = cv::imread(left_path, cv::IMREAD_GRAYSCALE);
  cv::Mat I2 = cv::imread(right_path, cv::IMREAD_GRAYSCALE);

  if (scale_ratio != 1.0f) {
    cv::resize(I1, I1, cv::Size(), scale_ratio, scale_ratio);
    cv::resize(I2, I2, cv::Size(), scale_ratio, scale_ratio);
  }
  
  const int min_disp = 0;
  const int LR_max_diff = 1;
  const auto census_type = static_cast<sgm::CensusType>(1);

  ASSERT_MSG(!I1.empty() && !I2.empty(), "imread failed.");
  ASSERT_MSG(I1.size() == I2.size() && I1.type() == I2.type(),
             "input images must be same size and type.");
  ASSERT_MSG(I1.type() == CV_8U || I1.type() == CV_16U,
             "input image format must be CV_8U or CV_16U.");
  ASSERT_MSG(max_d == 64 || max_d == 128 || max_d == 256,
             "disparity size must be 64, 128 or 256.");
  ASSERT_MSG(n_path == 4 || n_path == 8,
             "number of scanlines must be 4 or 8.");
  ASSERT_MSG(census_type == sgm::CensusType::CENSUS_9x7 ||
                 census_type == sgm::CensusType::SYMMETRIC_CENSUS_9x7,
             "census type must be 0 or 1.");

  const int src_depth = I1.type() == CV_8U ? 8 : 16;
  const int dst_depth = 16;
  const sgm::PathType path_type =
      n_path == 8 ? sgm::PathType::SCAN_8PATH : sgm::PathType::SCAN_4PATH;

  const sgm::StereoSGM::Parameters param(p1, p2, uniqueness, false, path_type,
                                         min_disp, LR_max_diff, census_type);
  sgm::StereoSGM ssgm(I1.cols, I1.rows, max_d, src_depth, dst_depth,
                      sgm::EXECUTE_INOUT_HOST2HOST, param);

  cv::Mat disparity(I1.size(), CV_16S);

  ssgm.execute(I1.data, I2.data, disparity.data);

  // create mask for invalid disp
  const cv::Mat mask = disparity == ssgm.get_invalid_disparity();
  cv::Mat u16_d;
  disparity.convertTo(u16_d, CV_16U);
  u16_d.setTo(0, mask);

  cv::imwrite(out_path, u16_d);

  return 0;
}
