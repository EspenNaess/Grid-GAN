#include <opencv2/opencv.hpp>

namespace fs = boost::filesystem;

/*
  rotate_image:
  Rotates the image.

  Args:
    img: the image to rotate
    rotation_angle: angle to rotate image with, given in degrees.
*/
Mat rotate_image(const Mat &img, const int rotation_angle) {
  Mat dst;
  Point2f pt(img.cols/2., img.rows/2.);
  Mat r = getRotationMatrix2D(pt, rotation_angle, 1.0);
  warpAffine(img, dst, r, Size(img.cols, img.rows));

  return dst;
}

/*
  resize_images:
  Resizes the images in the given folder containing the data and converts them to .png.

  Args:
    path: path to the images
    n, m: dimensions to resize images with
    orig_ftype: filetype of the images.
    with_contrast_enhancement: wether to contrast enhance the images or not, using the CLAHE-algorithm.
*/
void resize_images(const string path, const int n, const int m, const string orig_ftype, const bool with_contrast_enhancement=false) {
  string img_resized_path = path+"images_resized/";
  string mask_resized_path = path+"masks_resized/";

  string img_path = path+"images/";
  string mask_path = path+"masks/";

  fs::create_directories(img_resized_path);
  fs::create_directories(mask_resized_path);

  // resize masks
  for (auto &fp : fs::recursive_directory_iterator(mask_path)) {
    auto path = fp.path();
    if (path.extension() != orig_ftype)
        continue;

    Mat img = imread(path.string(), CV_LOAD_IMAGE_GRAYSCALE);
    if (!img.data) {
        cout << "Could not find/open the image to be resized. Terminating";
        return;
    }

    Mat resized_img;
    cv::resize(img, resized_img, Size(n,m));
    try {
      imwrite(mask_resized_path+"/"+path.stem().string()+".png", resized_img);
    } catch (runtime_error& ex) {
      fprintf(stderr, "Exception when saving resized image in png format: %s\n", ex.what());
    }
  }

  // resize images
  for (auto &fp : fs::recursive_directory_iterator(img_path)) {
    auto path = fp.path();
    if (path.extension() != orig_ftype)
        continue;

    Mat img = imread(path.string(), CV_LOAD_IMAGE_COLOR);

    if (!img.data) {
        cout << "Could not find/open the image to be resized. Terminating";
        return;
    }

    Mat resized_img;
    cv::resize(img, resized_img, Size(n,m));

    if (with_contrast_enhancement) {
      cv::Ptr<CLAHE> clahe = cv::createCLAHE(2.0);
      vector<Mat> rgb_chans;
      split(resized_img, rgb_chans);
      vector<Mat> new_rgb_chans = rgb_chans;
      clahe->apply(rgb_chans[0], new_rgb_chans[0]);
      clahe->apply(rgb_chans[1], new_rgb_chans[1]);
      clahe->apply(rgb_chans[2], new_rgb_chans[2]);
      merge(new_rgb_chans, resized_img);
    }

    try {
      imwrite(img_resized_path+"/"+path.stem().string()+".png", resized_img);
    } catch (runtime_error& ex) {
      fprintf(stderr, "Exception when saving resized image in png format: %s\n", ex.what());
    }
  }
}
