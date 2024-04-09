#include "interp2d.h"

Eigen::MatrixXd interp_2d_bilinear(const Eigen::MatrixXd& M, float scalar,
                                   bool align_corners) {
  int originalRows = M.rows(), originalCols = M.cols();
  int newRows = align_corners ? 1 + (originalRows - 1) * scalar
                              : std::round(originalRows * scalar);
  int newCols = align_corners ? 1 + (originalCols - 1) * scalar
                              : std::round(originalCols * scalar);

  Eigen::MatrixXd resized(newRows, newCols);

  double rowScale = align_corners
                        ? (originalRows - 1) / static_cast<double>(newRows - 1)
                        : originalRows / static_cast<double>(newRows);
  double colScale = align_corners
                        ? (originalCols - 1) / static_cast<double>(newCols - 1)
                        : originalCols / static_cast<double>(newCols);

  for (int i = 0; i < newRows; ++i) {
    double y = align_corners ? i * rowScale : (i + 0.5) * rowScale - 0.5;
    int y_low = std::floor(y);
    int y_high = std::min(y_low + 1, originalRows - 1);
    double y_lerp = y - y_low;

    for (int j = 0; j < newCols; ++j) {
      double x = align_corners ? j * colScale : (j + 0.5) * colScale - 0.5;
      int x_low = std::floor(x);
      int x_high = std::min(x_low + 1, originalCols - 1);
      double x_lerp = x - x_low;

      double topLeft = M.coeffRef(y_low, x_low);
      double topRight = M.coeffRef(y_low, x_high);
      double bottomLeft = M.coeffRef(y_high, x_low);
      double bottomRight = M.coeffRef(y_high, x_high);

      double topInterp = topLeft + (topRight - topLeft) * x_lerp;
      double bottomInterp = bottomLeft + (bottomRight - bottomLeft) * x_lerp;
      resized(i, j) = topInterp + (bottomInterp - topInterp) * y_lerp;
    }
  }

  return resized;
}

ot::RowMajorMatrixXd interp_2d_bilinear(const ot::RowMajorMatrixXd& M,
                                        float scalar, bool align_corners) {
  int originalRows = M.rows(), originalCols = M.cols();
  int newRows = align_corners ? 1 + (originalRows - 1) * scalar
                              : std::round(originalRows * scalar);
  int newCols = align_corners ? 1 + (originalCols - 1) * scalar
                              : std::round(originalCols * scalar);

  ot::RowMajorMatrixXd resized(newRows, newCols);

  double rowScale = align_corners
                        ? (originalRows - 1) / static_cast<double>(newRows - 1)
                        : originalRows / static_cast<double>(newRows);
  double colScale = align_corners
                        ? (originalCols - 1) / static_cast<double>(newCols - 1)
                        : originalCols / static_cast<double>(newCols);

  for (int i = 0; i < newRows; ++i) {
    double y = align_corners ? i * rowScale : (i + 0.5) * rowScale - 0.5;
    int y_low = std::floor(y);
    int y_high = std::min(y_low + 1, originalRows - 1);
    double y_lerp = y - y_low;

    for (int j = 0; j < newCols; ++j) {
      double x = align_corners ? j * colScale : (j + 0.5) * colScale - 0.5;
      int x_low = std::floor(x);
      int x_high = std::min(x_low + 1, originalCols - 1);
      double x_lerp = x - x_low;

      double topLeft = M.coeffRef(y_low, x_low);
      double topRight = M.coeffRef(y_low, x_high);
      double bottomLeft = M.coeffRef(y_high, x_low);
      double bottomRight = M.coeffRef(y_high, x_high);

      double topInterp = topLeft + (topRight - topLeft) * x_lerp;
      double bottomInterp = bottomLeft + (bottomRight - bottomLeft) * x_lerp;
      resized(i, j) = topInterp + (bottomInterp - topInterp) * y_lerp;
    }
  }

  return resized;
}

Eigen::MatrixXd interp_2d_nearest(const Eigen::MatrixXd& M, float scalar,
                                  bool align_corners) {
  int originalRows = M.rows(), originalCols = M.cols();
  int newRows = align_corners
                    ? 1 + static_cast<int>((originalRows - 1) * scalar)
                    : static_cast<int>(std::round(originalRows * scalar));
  int newCols = align_corners
                    ? 1 + static_cast<int>((originalCols - 1) * scalar)
                    : static_cast<int>(std::round(originalCols * scalar));

  Eigen::MatrixXd resized(newRows, newCols);

  double rowScale = align_corners
                        ? (originalRows - 1) / static_cast<double>(newRows - 1)
                        : originalRows / static_cast<double>(newRows);
  double colScale = align_corners
                        ? (originalCols - 1) / static_cast<double>(newCols - 1)
                        : originalCols / static_cast<double>(newCols);

  for (int i = 0; i < newRows; ++i) {
    double srcY = align_corners ? i * rowScale : (i + 0.5) * rowScale - 0.5;
    srcY = std::min(std::max(srcY, 0.0), static_cast<double>(originalRows - 1));
    int nearestRow = static_cast<int>(std::round(srcY));

    for (int j = 0; j < newCols; ++j) {
      double srcX = align_corners ? j * colScale : (j + 0.5) * colScale - 0.5;
      srcX =
          std::min(std::max(srcX, 0.0), static_cast<double>(originalCols - 1));
      int nearestCol = static_cast<int>(std::round(srcX));

      resized(i, j) = M(nearestRow, nearestCol);
    }
  }

  return resized;
}

ot::RowMajorMatrixXd interp_2d_nearest(const ot::RowMajorMatrixXd& M,
                                       float scalar, bool align_corners) {
  int originalRows = M.rows(), originalCols = M.cols();
  int newRows = align_corners
                    ? 1 + static_cast<int>((originalRows - 1) * scalar)
                    : static_cast<int>(std::round(originalRows * scalar));
  int newCols = align_corners
                    ? 1 + static_cast<int>((originalCols - 1) * scalar)
                    : static_cast<int>(std::round(originalCols * scalar));

  ot::RowMajorMatrixXd resized(newRows, newCols);

  double rowScale = align_corners
                        ? (originalRows - 1) / static_cast<double>(newRows - 1)
                        : originalRows / static_cast<double>(newRows);
  double colScale = align_corners
                        ? (originalCols - 1) / static_cast<double>(newCols - 1)
                        : originalCols / static_cast<double>(newCols);

  for (int i = 0; i < newRows; ++i) {
    double srcY = align_corners ? i * rowScale : (i + 0.5) * rowScale - 0.5;
    srcY = std::min(std::max(srcY, 0.0), static_cast<double>(originalRows - 1));
    int nearestRow = static_cast<int>(std::round(srcY));

    for (int j = 0; j < newCols; ++j) {
      double srcX = align_corners ? j * colScale : (j + 0.5) * colScale - 0.5;
      srcX =
          std::min(std::max(srcX, 0.0), static_cast<double>(originalCols - 1));
      int nearestCol = static_cast<int>(std::round(srcX));

      resized(i, j) = M(nearestRow, nearestCol);
    }
  }

  return resized;
}