//
// Created by Soul D on 4/25/24.
//

#ifndef IMAGE_FILTER_IMAGEREADER_H
#define IMAGE_FILTER_IMAGEREADER_H
#include <string>
#include <opencv2/opencv.hpp>

class ImageReader
{
private:
public:
    /* data */
    cv::Mat image;

    ImageReader();
    ImageReader(const std::string &filename);
    ~ImageReader();
    void readImage(const std::string &filename);
    void displayImage(const std::string &windowName);
};

#endif // IMAGE_FILTER_IMAGEREADER_H
