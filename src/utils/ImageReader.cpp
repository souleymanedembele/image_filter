//
// Created by Soul D on 4/25/24.
//
#include "utils/ImageReader.h"

ImageReader::ImageReader(/* args */)
{
}

ImageReader::ImageReader(const std::string &filename)
{
    readImage(filename);
}
ImageReader::~ImageReader()
{
}
void ImageReader::readImage(const std::string &filename)
{
    // load the image from a file path
    image = cv::imread(filename, cv::IMREAD_COLOR);
}
void ImageReader::displayImage(const std::string &windowName)
{
    if (!image.empty())
    {
        // Display the image in a window
        cv::namedWindow(windowName, cv::WINDOW_NORMAL);
        // cv::imshow(windowName, image);
        // cv::waitKey(0);
        // Create a window to display the image
        // cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

        // Show the image in the window
        cv::imshow(windowName, image);

        // Print message to standard output
        std::cout << "Press a key to exit" << std::endl;

        // Wait for a key press indefinitely
        cv::waitKey(0);

        // Destroy the created window
        cv::destroyWindow(windowName);
    }
    else
    {
        // Print error message to standard error
        std::cerr << "Error reading image" << std::endl;
    }
}