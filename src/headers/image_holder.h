//
// Created by otost on 21/10/2022.
//

#ifndef PV021_NEURAL_NETWORK_IMAGE_HOLDER_H
#define PV021_NEURAL_NETWORK_IMAGE_HOLDER_H
#include <string>
#include <vector>
#include "utils.h"

using namespace std;



class ImageHolder {
    //string images;

public:
    explicit ImageHolder(string images_path, string labels_path);

    bool load_images();
    bool load_labels();

    int get_num_images() const;
    Matrix get_image_as_matrix(int i);
    void compute_mean();
    void compute_std_dev();
    void standardize();

private:
    int num_images;
    vector<vector<double>> images;
    vector<int> labels;
    string images_path;
    string labels_path;
    vector<double> mean;
    vector<double> stddev;

};



#endif //PV021_NEURAL_NETWORK_IMAGE_HOLDER_H
