#include <iostream>

#include <boost/filesystem.hpp>
#include <boost/range/combine.hpp>

#include "grid_framework_ops.cpp"
#include "basic_img_ops.cpp"

using namespace std;
using namespace cv;
namespace fs = boost::filesystem;

/*
  train_test_split:
  Splits dataset into training and testing datasets.
  Implemented as a C++ alternative to the train_test_split function in Pythons Scipy library.

  Args:
    data: vector containing the data
    test_size: size of the test data partition

  Ret:
    Tuple of train and test datasets
*/
template<typename T>
tuple<vector<T>, vector<T>> train_test_split(vector<T> &data, const int test_size) {
  if (test_size == 0) {
    vector<T> &train_data = data;
    vector<T> test_data = {};
    return make_tuple(train_data, test_data);
  }

  // shuffle with a time-based seed
  const unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  shuffle(data.begin(), data.end(), std::default_random_engine(seed));

  vector<T> test_data(&data[0], &data[test_size]);
  vector<T> train_data(&data[test_size], &data[data.size()-1]);

  if (test_data.size() != test_size)
    cout << "Error; could not retrieve test data";

  return make_tuple(train_data, test_data);
}

/*
  split_dataset:
    splits dataset into either two or three partitions based on the given parameters

  Args:
    dataset: dataset to split
    val_percent_of_train: the amount of validation data, given as a percent of the training data
    test_size_percent: size of the test data partition

  Ret:
    Tuple, where the first element is a vector containing the names of the datasets.
    The second tuple element is a vector containing the resulting datasets, which are splitted.
*/

template<typename T>
tuple<vector<string>, vector<vector<T>>> split_dataset(vector<T> &dataset, const double val_percent_of_train, const double test_size_percent) {
  vector<string> datasets_names;
  vector<vector<T>> datasets_splitted;

  int total_data_size = dataset.size();
  int test_size = int(test_size_percent*total_data_size);

  if (val_percent_of_train == 0){
    vector<string> train_names, test_names;

    auto datasets = train_test_split(dataset, test_size);
    tie(train_names, test_names) = datasets;

    datasets_names = {"train", "test"};
    datasets_splitted = {train_names, test_names};
  } else {
    vector<string> train_names, val_names, test_names;

    int train_and_val_size = total_data_size-test_size;
    int val_size = int(val_percent_of_train*train_and_val_size);

    tie(train_names, test_names) = train_test_split(dataset, test_size);
    tie(train_names, val_names) = train_test_split(train_names, val_size);

    datasets_names = {"train", "val", "test"};
    datasets_splitted = {train_names, val_names, test_names};
  }

  return make_tuple(datasets_names, datasets_splitted);
}

/*
  generate_data:
  Augments the datasets with grid information. For each image and ground truth pair
  in the dataset, a corresponding grid encoding and a grid segmentation ground truth is generated
  for all possible grid sizes for the given image.

  Args:
    dataset_path: path to dataset folders
    img_path: path to images
    mask_path: path to ground truth
    grid_sizes: vector of the different grid sizes the dataset will be augmented with
    val_percent_of_train: the amount of validation data, given as a percent of the training data
    test_size_percent: size of the test data partition
    rot_aug: wether to include rotation augmentation or not
*/
void generate_data(const string dataset_path, const string img_path, const string mask_path, const vector<int> &grid_sizes, const double val_percent_of_train, const double test_size_percent, const bool rot_aug=true) {
  if (!fs::exists(img_path)) {
    cout << "Aborts. Resized masks and image folders images_resized and masks_resized do not exist" << endl;
    return;
  }

  vector<string> img_names;
  int img_count = 0;
  for (auto &fp : fs::recursive_directory_iterator(img_path)) {
    if (fp.path().extension() == ".png") {
      img_names.push_back(fp.path().filename().string());
      img_count++;
    }
  }

  if (img_count == 0) {
    cout << "Aborts. No images found in image path folder" << endl;
    return;
  }

  // read first img to retrieve basic meta information about the data
  Mat img = imread(img_path+img_names[0], CV_LOAD_IMAGE_GRAYSCALE);
  int img_size = img.cols;

  // get train, val and test datasets
  vector<string> datasets_names;
  vector<vector<string>> datasets_splitted;
  auto split_result = split_dataset(img_names, val_percent_of_train, test_size_percent);
  tie(datasets_names, datasets_splitted) = split_result;

  // create grid augmented datasets
  for (auto tup : boost::combine(datasets_names, datasets_splitted)) {
    string dataset_name;
    cout << dataset_name;
    vector<string> dataset_partition;
    boost::tie(dataset_name, dataset_partition) = tup;

    const string imgs_partition_path = dataset_path+dataset_name+"/imgs/";
    const string masks_partition_path = dataset_path+dataset_name+"/masks/";

    // Rotation augments of the dataset
    vector<int> rotations_augments;
    if (dataset_name.compare("train") == 0 && rot_aug) {
        rotations_augments = {90,180,270,360};
    } else {
        rotations_augments = {0};
    }

    // create directories for grid sizes and create corresponding grid size encodings
    for (const int grid_size : grid_sizes) {
      string grids_path = dataset_path+dataset_name+"/"+to_string(grid_size)+"x"+to_string(grid_size)+"grids"+"/";
      fs::create_directories(grids_path);
      Mat grid_enc = generate_grid_encoding(img_size, grid_size);
      imwrite(grids_path+"grid_encoding.png", grid_enc);
    }

    int img_count = 0;
    for (auto &fp : dataset_partition) {
      Mat img = imread(img_path+fp, CV_LOAD_IMAGE_COLOR);
      Mat mask = imread(mask_path+fp, CV_LOAD_IMAGE_GRAYSCALE);

      if (!img.data) {
          cout << "Could not find/open image. Terminating";
          return;
      }
      if (img.cols != img.rows)
        throw "Images are not square";

      for (int rotation : rotations_augments) {
        for (int grid_size : grid_sizes) {
          Mat img_rotated = rotate_image(img, rotation);
          imwrite(imgs_partition_path+to_string(img_count)+".png", img_rotated);

          Mat grid_mask = generate_grid_ground_truth(mask, grid_size);
          Mat mask_rotated = rotate_image(grid_mask, rotation);
          imwrite(masks_partition_path+to_string(img_count)+".png", mask_rotated);

          string grids_path = dataset_path+dataset_name+"/"+to_string(grid_size)+"x"+to_string(grid_size)+"grids"+"/";
          string grids_path_imgs = grids_path+"/imgs/";
          string grids_path_masks = grids_path+"/masks/";

          fs::create_directories(grids_path_imgs);
          fs::create_directories(grids_path_masks);

          imwrite(grids_path_imgs+to_string(img_count)+".png", img_rotated);
          imwrite(grids_path_masks+to_string(img_count)+".png", mask_rotated);

          img_count++;
        }
      }
    }
  }
}

int main(int argc, char **argv) {
  string path = "/Users/espennaess/Documents/Universitetsarbeid/Masterting/backup av stasjonær/GitHub Public/preproc_res/";
  //resize_images(path+"Kvasir-SEG/", 128,128, ".jpg");
  vector<int> grid_sizes = {2,4,8,16,128};
  string dataset_path = path+"Kvasir-SEG/";
  string img_resized_path = dataset_path+"images_resized/";
  string mask_resized_path = dataset_path+"masks_resized/";

  generate_data(dataset_path, img_resized_path, mask_resized_path, grid_sizes, 0, 0.2);
  return 0;
}
