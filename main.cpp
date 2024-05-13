#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>  // Include chrono for high-resolution timing

using namespace cv;
using namespace std;

class ImageFilter {
public:
    virtual void apply(const Mat& inputImage, Mat& outputImage) = 0;
};

class GaussianBlurFilter : public ImageFilter {
    int kernelSize;
    double sigmaX;

public:
    GaussianBlurFilter(int ks, double sigma) : kernelSize(ks), sigmaX(sigma) {}

    void apply(const Mat& inputImage, Mat& outputImage) override {
        GaussianBlur(inputImage, outputImage, Size(kernelSize, kernelSize), sigmaX);
    }
};

class MedianFilter : public ImageFilter {
    int kernelSize;

public:
    MedianFilter(int ks) : kernelSize(ks) {}

    void apply(const Mat& inputImage, Mat& outputImage) override {
        medianBlur(inputImage, outputImage, kernelSize);
    }
};

class SobelEdgeDetection : public ImageFilter {
    int scale;
    int delta;

public:
    SobelEdgeDetection(int sc, int dlt) : scale(sc), delta(dlt) {}

    void apply(const Mat& inputImage, Mat& outputImage) override {
        Mat gray, grad_x, grad_y;
        cvtColor(inputImage, gray, COLOR_BGR2GRAY);
        Sobel(gray, grad_x, CV_16S, 1, 0, 3, scale, delta, BORDER_DEFAULT);
        Sobel(gray, grad_y, CV_16S, 0, 1, 3, scale, delta, BORDER_DEFAULT);
        Mat abs_grad_x, abs_grad_y;
        convertScaleAbs(grad_x, abs_grad_x);
        convertScaleAbs(grad_y, abs_grad_y);
        addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, outputImage);
    }
};

int main() {
    Mat image = imread("./images/peakpx.jpg");
    if (image.empty()) {
        cout << "Could not open or find the image" << endl;
        return -1;
    }

    GaussianBlurFilter gBlur(9, 0);
    MedianFilter mFilter(5);
    SobelEdgeDetection sEdge(1, 0);

    Mat imageBlurred, imageMedianFiltered, imageSobelEdge;

    auto start = chrono::high_resolution_clock::now();
    gBlur.apply(image, imageBlurred);
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
    cout << "Gaussian Blur Time: " << duration.count() << " microseconds" << endl;

    start = chrono::high_resolution_clock::now();
    mFilter.apply(image, imageMedianFiltered);
    stop = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::microseconds>(stop - start);
    cout << "Median Filter Time: " << duration.count() << " microseconds" << endl;

    start = chrono::high_resolution_clock::now();
    sEdge.apply(image, imageSobelEdge);
    stop = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::microseconds>(stop - start);
    cout << "Sobel Edge Detection Time: " << duration.count() << " microseconds" << endl;

    imshow("Original Image", image);
    imshow("Gaussian Blurred Image", imageBlurred);
    imshow("Median Filtered Image", imageMedianFiltered);
    imshow("Sobel Edge Detected Image", imageSobelEdge);

    waitKey(0);
    return 0;
}
