//
// Created by otost on 21/10/2022.
//

#ifndef PV021_NEURAL_NETWORK_IMAGE_HOLDER_H
#define PV021_NEURAL_NETWORK_IMAGE_HOLDER_H
#include <string>
#include <vector>
#include "matrix.h"

using namespace std;


class ImageHolder {
    //string images;

public:
    explicit ImageHolder(string images_path, string labels_path, int image_limit = -1);

    bool load_images();
    bool load_labels();

    int get_num_images() const;
    Matrix get_image_as_matrix(int i);
    vector<double> get_image(int i);
    vector<double> get_label(int i);
    vector<vector<double>> get_images(int start, int end);
    vector<vector<double>> get_labels(int start, int end);
    void compute_mean();
    void compute_std_dev();
    void standardize();

private:
    int num_images;
    int image_limit;
    vector<vector<double>> images;
    vector<int> labels;
    string images_path;
    string labels_path;
    vector<double> mean;
    vector<double> stddev;

};



#endif //PV021_NEURAL_NETWORK_IMAGE_HOLDER_H
