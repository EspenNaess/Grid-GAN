#include <algorithm>
#include <random>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

/*
  generate_small_grid_ground_truth:
    Generates a small ground truth grid segmentation, where the output is
    an image of the same dimensions as the grid size.

    Args:
      mask: mask to create grid segmentation mask from
      grid_size: size of the grid

    Ret:
      Matrix of marked cells from the mask. Matrix size equal to the grid size.
*/
Mat generate_small_grid_ground_truth(const Mat &mask, const int grid_size) {
  Mat marked_cells = Mat::zeros(cv::Size(grid_size, grid_size), CV_8UC1);
  int grid_cell_size = int(mask.rows/grid_size);

  // find the marked cells for the grid segmentation ground truth
  for (int i = 0; i<mask.rows; i++)
    for (int k = 0; k<mask.cols; k++)
      if (mask.at<uchar>(i,k) > 127)
        marked_cells.at<uchar>(int(i/grid_cell_size), int(k/grid_cell_size)) = 255;

  return marked_cells;
}

/*
  generate_grid_ground_truth:
    Given a corresponding ground truth mask and a grid size, this function generates the corresponding
    grid segmentation mask of the same size as the original mask. See master thesis for more details.

    Args:
      mask: mask to create grid segmentation mask from
      grid_size: size of the grid

    Ret:
      Grid segmentation mask
*/
Mat generate_grid_ground_truth(const Mat &mask, const int grid_size) {
  Mat marked_cells = generate_small_grid_ground_truth(mask, grid_size);

  Mat ground_truth = Mat::zeros(cv::Size(mask.rows, mask.cols), CV_8UC1);
  int grid_cell_size = int(mask.rows/grid_size);

  // create grid segmentation ground truth
  for (int i = 0; i<mask.rows; i++)
    for (int k = 0; k<mask.cols; k++)
      if (marked_cells.at<uchar>(int(i/grid_cell_size), int(k/grid_cell_size)) == 255)
        ground_truth.at<uchar>(i,k) = 255;

  return ground_truth;
}
/*
  Generates a grid encoding. See thesis for more detailed explanations.

  Args:
    img_size: size of the grid encoding
    grid_size: size of the grid in the grid encoding

  Ret:
    The grid encoding, given as a matrix
*/
Mat generate_grid_encoding(const int img_size, const int grid_size) {
  Mat grid_enc = Mat::zeros(cv::Size(img_size, img_size), CV_8UC1);
  int grid_cell_len = int(img_size/grid_size);

  for (int i = 0; i<img_size; i++)
    for (int k = 0; k<img_size; k++)
      if (int(i/grid_cell_len) % 2 != int(k/grid_cell_len) % 2)
        grid_enc.at<uchar>(i,k) = 255;

  return grid_enc;
}
