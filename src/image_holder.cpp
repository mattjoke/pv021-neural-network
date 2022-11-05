//
// Created by otost on 21/10/2022.
//

#include <sstream>
#include <fstream>
#include <iostream>
#include <utility>
#include "image_holder.h"

ImageHolder::ImageHolder(string images_path, string labels_path) {
    this->images_path = std::move(images_path);
    this->labels_path = std::move(labels_path);
    load_images();
    load_labels();
    standardize();
}

bool ImageHolder::load_images() {
    auto outputStringStream = ostringstream{};
    ifstream input_file(images_path);
    // Checks if file exists, otherwise it throws SIGTERM
    if (input_file.fail()){
        cout << "Failed to open file: " << images_path << endl;
        return false;
    }


    if (!input_file.is_open()) {
        return false;
    }
    outputStringStream << input_file.rdbuf();
    string file_content = outputStringStream.str();
    vector<double> image{};
    int num = 0;
    for(char letter : file_content) {
        if(letter == ',') {
            image.emplace_back(num);
            num = 0;
            continue;
        }
        if(letter == '\n') {
            image.emplace_back(num);
            num = 0;
            images.emplace_back(image);
            image = {};
        }
        else {
            int ia = letter - '0';
            num = 10 * num + ia;
        }
    }
    num_images = images.size();
    return true;
}

bool ImageHolder::load_labels() {
    auto outputStringStream = ostringstream{};
    ifstream input_file(labels_path);
    // Checks if file exists, otherwise it throws SIGTERM
    if (input_file.fail()){
        cout << "Failed to open file: " << images_path << endl;
        return false;
    }

    if (!input_file.is_open()) {
        return false;
    }
    outputStringStream << input_file.rdbuf();
    string file_content = outputStringStream.str();
    int label = 0;
    int num = 0;
    for(char letter : file_content) {
        if(letter == '\n') {
            labels.emplace_back(label);
            label = 0;
            num = 0;
        }
        else {
            int ia = letter - '0';
            num = 10 * num + ia;
        }
    }
    return true;
}

int ImageHolder::get_num_images() const {
    return num_images;
}

Matrix ImageHolder::get_image_as_matrix(int i) {
    auto image = this->images[i];
    auto matrixImage = Matrix(1, image.size());
    for(int j=0; j< image.size(); j++) {
        matrixImage.set(0, j, image[j]);
    }
    return matrixImage;
}

void ImageHolder::compute_mean() {
    int image_size = this->images[0].size();
    vector<double> means(image_size);
    for(int i=0; i<num_images; i++) {
        for(int j=0; j<image_size; j++) {
            means[j] += images[i][j];
        }
    }
    for(int j=0; j<image_size; j++) {
        means[j] = means[j] / num_images;
    }
    this->mean = means;
}

void ImageHolder::compute_std_dev() {
    int image_size = this->images[0].size();
    vector<double> standardDeviation(image_size);
    for(int i=0; i<num_images; i++) {
        for(int j=0; j<image_size; j++) {
            double v = images[i][j];
            standardDeviation[j] += ((v - mean[j]) * (v - mean[j]));
        }
    }
    for(int j=0; j<image_size; j++) {
        standardDeviation[j] = standardDeviation[j] / num_images;
    }
    stddev = standardDeviation;
}

void ImageHolder::standardize() {
    compute_mean();
    compute_std_dev();
    int image_size = this->images[0].size();
    for(int i=0; i<num_images; i++) {
        for(int j=0; j<image_size; j++) {
            images[i][j] = ((images[i][j] - mean[j]) / stddev[j]);
        }
    }
}
