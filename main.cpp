#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// Function to apply Gaussian Blur
void applyGaussianBlur(const Mat& inputImage, Mat& outputImage, int kernelSize, double sigmaX) {
    // kernelSize: The size of the kernel to be used (must be odd).
    // sigmaX: The standard deviation in X; a value of 0 lets the algorithm choose.
    GaussianBlur(inputImage, outputImage, Size(kernelSize, kernelSize), sigmaX);
}
// Function to apply Median Filter
void applyMedianFilter(const Mat& inputImage, Mat& outputImage, int kernelSize) {
    // kernelSize: The size of the kernel, must be a positive odd number.
    medianBlur(inputImage, outputImage, kernelSize);
}

// Function to apply Sobel Edge Detection
void applySobelEdgeDetection(const Mat& inputImage, Mat& outputImage, int scale, int delta) {
    Mat gray, grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;

    // Convert to grayscale
    cvtColor(inputImage, gray, COLOR_BGR2GRAY);

    // Gradient X
    Sobel(gray, grad_x, CV_16S, 1, 0, 3, scale, delta, BORDER_DEFAULT);
    convertScaleAbs(grad_x, abs_grad_x);

    // Gradient Y
    Sobel(gray, grad_y, CV_16S, 0, 1, 3, scale, delta, BORDER_DEFAULT);
    convertScaleAbs(grad_y, abs_grad_y);

    // Total Gradient (approximate)
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, outputImage);
}

int main() {
    // Load an image
    Mat image = imread("./images/peakpx.jpg");
    if (image.empty()) {
        cout << "Could not open or find the image" << endl;
        return -1;
    }

    // Display original image
    namedWindow("Original Image", WINDOW_AUTOSIZE);
    imshow("Original Image", image);

    // Apply Gaussian Blur
    Mat imageBlurred;
    applyGaussianBlur(image, imageBlurred, 9, 0);  // You can adjust the kernel size and sigma

    // Display blurred image
    namedWindow("Blurred Image", WINDOW_AUTOSIZE);
    imshow("Blurred Image", imageBlurred);

    // Apply Sobel Edge Detection
    Mat imageSobelEdge;
    applySobelEdgeDetection(image, imageSobelEdge, 1, 0); // Scale and delta can be adjusted

    // Display edge image
    namedWindow("Sobel Edge Detected Image", WINDOW_AUTOSIZE);
    imshow("Sobel Edge Detected Image", imageSobelEdge);

    // Apply Median Filter
    Mat imageMedianFiltered;
    applyMedianFilter(image, imageMedianFiltered, 5); // You can adjust the kernel size

    // Display filtered image
    namedWindow("Median Filtered Image", WINDOW_AUTOSIZE);
    imshow("Median Filtered Image", imageMedianFiltered);


    // Wait for a keystroke in the window
    waitKey(0);
    return 0;
}
