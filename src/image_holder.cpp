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
}

bool ImageHolder::load_images() {
    auto ss = ostringstream{};
    ifstream input_file(images_path);
    if (!input_file.is_open()) {
        return false;
    }
    ss << input_file.rdbuf();
    string file_content = ss.str();
    vector<int> image{};
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
    auto ss = ostringstream{};
    ifstream input_file(labels_path);
    if (!input_file.is_open()) {
        return false;
    }
    ss << input_file.rdbuf();
    string file_content = ss.str();
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
