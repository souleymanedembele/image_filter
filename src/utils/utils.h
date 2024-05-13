//
// Created by SoulAndAnyaPC on 5/13/2024.
//

#ifndef IMAGE_FILTERING_UTILS_H
#define IMAGE_FILTERING_UTILS_H

#include "opencv2/opencv.hpp"
#include <iostream>
#include <chrono>
#include <fstream>
#include <filesystem>

namespace fs = std::filesystem;

using namespace cv;
using namespace std;


// Function to calculate SSIM
double getSSIM(const Mat& i1, const Mat& i2) {
    const double C1 = 6.5025, C2 = 58.5225;
    int d = CV_32F;
    Mat I1, I2;
    i1.convertTo(I1, d);
    i2.convertTo(I2, d);
    Mat I2_2 = I2.mul(I2), I1_2 = I1.mul(I1), I1_I2 = I1.mul(I2);
    Mat mu1, mu2;
    GaussianBlur(I1, mu1, Size(11, 11), 1.5);
    GaussianBlur(I2, mu2, Size(11, 11), 1.5);
    Mat mu1_2 = mu1.mul(mu1), mu2_2 = mu2.mul(mu2), mu1_mu2 = mu1.mul(mu2);
    Mat sigma1_2, sigma2_2, sigma12;
    GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;
    GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;
    GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;
    Mat t1 = 2 * mu1_mu2 + C1, t2 = 2 * sigma12 + C2;
    Mat t3 = t1.mul(t2), t4 = mu1_2 + mu2_2 + C1, t5 = sigma1_2 + sigma2_2 + C2;
    t4 = t4.mul(t5);
    Mat ssim_map;
    divide(t3, t4, ssim_map);
    Scalar mssim = mean(ssim_map);
    return mssim.val[0];
}

#endif //IMAGE_FILTERING_UTILS_H
