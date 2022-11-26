//
// Created by otost on 21/10/2022.
//

#include <sstream>
#include <fstream>
#include <iostream>
#include <utility>
#include <random>
#include "image_holder.h"

ImageHolder::ImageHolder(string images_path, string labels_path, int image_limit) {
    this->images_path = std::move(images_path);
    this->labels_path = std::move(labels_path);
    this->image_limit = image_limit;
    load_images();
    load_labels();

    for (int i = 0; i < this->num_images; i++) {
        for (int j = 0; j < this->images[i].size(); j++) {
            this->images[i][j] = (double) this->images[i][j] / 255.0;
        }
    }
    shuffleIndices();
    //standardize();
}

bool ImageHolder::load_images() {
    auto outputStringStream = ostringstream{};
    ifstream input_file(images_path);
    // Checks if file exists, otherwise it throws SIGTERM
    if (input_file.fail()) {
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
    for (char letter: file_content) {
        if (letter == ',') {
            image.emplace_back(num);
            num = 0;
            continue;
        }
        if (letter == '\n') {
            image.emplace_back(num);
            num = 0;
            images.emplace_back(image);
            if (images.size() >= this->image_limit) {
                break;
            }
            image = {};
        } else {
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
    if (input_file.fail()) {
        cout << "Failed to open file: " << images_path << endl;
        return false;
    }

    if (!input_file.is_open()) {
        return false;
    }
    outputStringStream << input_file.rdbuf();
    string file_content = outputStringStream.str();
    int num = 0;
    for (char letter: file_content) {
        if (letter == '\n') {
            labels.emplace_back(num);
            if (labels.size() >= this->image_limit) {
                break;
            }
            num = 0;
        } else {
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
    for (int j = 0; j < image.size(); j++) {
        matrixImage.set(0, j, image[j]);
    }
    return matrixImage;
}

vector<double> ImageHolder::get_image(int i) {
    return images[i];
}

vector<double> ImageHolder::get_label(int i) {
    vector<double> label(10, 0);
    label[labels[i]] = 1;
    return label;
}

vector<vector<double>> ImageHolder::get_images(int start, int end) {
    vector<vector<double>> out;
    for (int i = start; i < end; i++) {
        out.emplace_back(this->images[indices[i]]);
    }
    return out;
}

vector<vector<double>> ImageHolder::get_labels(int start, int end) {
    vector<vector<double>> out;
    for (int i = start; i < end; i++) {
        auto label = vector<double>(10, 0);
        label[labels[indices[i]]] = 1;
        out.emplace_back(label);
    }
    return out;
}

void ImageHolder::compute_mean() {
    int image_size = this->images[0].size();
    vector<double> means(image_size);
    for (int i = 0; i < num_images; i++) {
        for (int j = 0; j < image_size; j++) {
            means[j] += images[i][j];
        }
    }
    for (int j = 0; j < image_size; j++) {
        means[j] = means[j] / num_images;
    }
    this->mean = means;
}

void ImageHolder::compute_std_dev() {
    int image_size = this->images[0].size();
    vector<double> standardDeviation(image_size);
    for (int i = 0; i < num_images; i++) {
        for (int j = 0; j < image_size; j++) {
            double v = images[i][j];
            standardDeviation[j] += ((v - mean[j]) * (v - mean[j]));
        }
    }
    for (int j = 0; j < image_size; j++) {
        standardDeviation[j] = standardDeviation[j] / num_images;
    }
    stddev = standardDeviation;
}

void ImageHolder::standardize() {
    compute_mean();
    compute_std_dev();
    int image_size = this->images[0].size();
    for (int i = 0; i < num_images; i++) {
        for (int j = 0; j < image_size; j++) {
            images[i][j] = ((images[i][j] - mean[j]) / stddev[j]);
        }
    }
}

void ImageHolder::shuffleIndices() {
    for (int i = 0; i < num_images; i++) {
        indices.emplace_back(i);
    }
    random_device rd;
    mt19937 g(rd());
    shuffle(indices.begin(), indices.end(), g);
}
