/**
 * Cpp program by Souleymane Dembele on 4/25/24.
 *
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <opencv2/imgcodecs.hpp>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#else
#include <unistd.h>
#endif

namespace fs = std::filesystem;

using namespace cv;
using namespace std;

#ifdef _WIN32
SIZE_T getMemoryUsage() {
    PROCESS_MEMORY_COUNTERS pmc;
    GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
    return pmc.WorkingSetSize;
}
#else
size_t getMemoryUsage() {
    long rss = 0L;
    FILE* fp = nullptr;
    if ((fp = fopen("/proc/self/statm", "r")) != nullptr) {
        if (fscanf(fp, "%*s%ld", &rss) != 1) {
            fclose(fp);
            return 0; // Can't read
        }
        fclose(fp);
        return rss * static_cast<size_t>(sysconf(_SC_PAGESIZE));
    }
    return 0; // Can't open
}
#endif

double getSSIM(const Mat& i1, const Mat& i2) {
    const double C1 = 6.5025, C2 = 58.5225;
    int d = CV_32F;

    Mat I1, I2;
    i1.convertTo(I1, d);
    i2.convertTo(I2, d);

    if (I1.size() != I2.size() || I1.type() != I2.type()) {
        cerr << "Error: Images must be of the same size and type." << endl;
        return -1;
    }

    Mat I2_2 = I2.mul(I2), I1_2 = I1.mul(I1), I1_I2 = I1.mul(I2);
    Mat mu1, mu2, mu1_2, mu2_2, mu1_mu2, sigma1_2, sigma2_2, sigma12;
    GaussianBlur(I1, mu1, Size(11, 11), 1.5);
    GaussianBlur(I2, mu2, Size(11, 11), 1.5);
    mu1_2 = mu1.mul(mu1);
    mu2_2 = mu2.mul(mu2);
    mu1_mu2 = mu1.mul(mu2);
    GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;
    GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;
    GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;
    Mat t1 = 2 * mu1_mu2 + C1;
    Mat t2 = 2 * sigma12 + C2;
    Mat t3, t4, ssim_map;
    multiply(t1, t2, t3);  // Element-wise multiplication
    t4 = (mu1_2 + mu2_2 + C1).mul(sigma1_2 + sigma2_2 + C2); // Element-wise multiplication
    divide(t3, t4, ssim_map);  // Element-wise division
    Scalar mssim = mean(ssim_map);  // Average SSIM over the image
    return mssim.val[0];
}

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
//        Mat gray, grad_x, grad_y;
//        cvtColor(inputImage, gray, COLOR_BGR2GRAY);
//        Sobel(gray, grad_x, CV_16S, 1, 0, 3, scale, delta, BORDER_DEFAULT);
//        Sobel(gray, grad_y, CV_16S, 0, 1, 3, scale, delta, BORDER_DEFAULT);
//        Mat abs_grad_x, abs_grad_y;
//        convertScaleAbs(grad_x, abs_grad_x);
//        convertScaleAbs(grad_y, abs_grad_y);
//        addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, outputImage);

        Mat gray;
        // Convert to grayscale if input is color
        if (inputImage.channels() == 3) {
            cvtColor(inputImage, gray, COLOR_BGR2GRAY);
        } else {
            gray = inputImage;
        }

        Mat grad_x, grad_y;
        Sobel(gray, grad_x, CV_16S, 1, 0, 3, scale, delta, BORDER_DEFAULT);
        Sobel(gray, grad_y, CV_16S, 0, 1, 3, scale, delta, BORDER_DEFAULT);
        Mat abs_grad_x, abs_grad_y;
        convertScaleAbs(grad_x, abs_grad_x);
        convertScaleAbs(grad_y, abs_grad_y);
        addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, outputImage);
    }
};

int main() {
    ofstream dataFile("performance_data.csv");
    if (!dataFile.is_open()) {
        cerr << "Failed to open data file for writing." << endl;
        return -1;
    }

    dataFile << "Image,Filter,ExecutionTime,MemoryUsage,SSIM,PSNR\n";
//    string imagePath = "./images";
//    string outputDir = "./out";
    string outputDir = "./test2014_out";

    if (!fs::exists(outputDir)) {
        fs::create_directory(outputDir);
    }


    string imagePath = "./test2014";
    if (!fs::exists(imagePath)) {
        cerr << "Image directory does not exist: " << imagePath << endl;
        return -1;
    }

    for (const auto& entry : fs::directory_iterator(imagePath)) {
        Mat image = imread(entry.path().string(), IMREAD_COLOR);
        if (image.empty()) {
            cout << "Could not open or find the image: " << entry.path() << endl;
            continue;
        }

        GaussianBlurFilter gBlur(9, 0);
        MedianFilter mFilter(5);
        SobelEdgeDetection sEdge(1, 0);

        Mat imageBlurred, imageMedianFiltered, imageSobelEdge;

        // Apply Gaussian Blur
        auto start = chrono::high_resolution_clock::now();
        size_t memBefore = getMemoryUsage();
        gBlur.apply(image, imageBlurred);
        size_t memAfter = getMemoryUsage();
        auto stop = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
        double memoryUsed = memAfter > memBefore ? (memAfter - memBefore) : 0;
        double ssim = getSSIM(image, imageBlurred);
        double psnr = cv::PSNR(image, imageBlurred);
        dataFile << entry.path().filename() << ",Gaussian," << duration.count() << "," << memoryUsed << "," << ssim << "," << psnr << "\n";

        // Apply Median Filter
        start = chrono::high_resolution_clock::now();
        memBefore = getMemoryUsage();
        mFilter.apply(image, imageMedianFiltered);
        memAfter = getMemoryUsage();
        stop = chrono::high_resolution_clock::now();
        duration = chrono::duration_cast<chrono::microseconds>(stop - start);
        memoryUsed = memAfter > memBefore ? (memAfter - memBefore) : 0;
        ssim = getSSIM(image, imageMedianFiltered);
        psnr = cv::PSNR(image, imageMedianFiltered);
        dataFile << entry.path().filename() << ",Median," << duration.count() << "," << memoryUsed << "," << ssim << "," << psnr << "\n";

        // Apply Sobel Edge Detection
        start = chrono::high_resolution_clock::now();
        memBefore = getMemoryUsage();
        sEdge.apply(image, imageSobelEdge);
        memAfter = getMemoryUsage();
        stop = chrono::high_resolution_clock::now();
        duration = chrono::duration_cast<chrono::microseconds>(stop - start);
        memoryUsed = memAfter > memBefore ? (memAfter - memBefore) : 0;

        if (imageSobelEdge.empty()) {
            cerr << "Sobel output is empty, check filter application." << endl;
            continue;  // Skip this iteration if Sobel failed
        }

        if (image.size() != imageSobelEdge.size() || image.type() != imageSobelEdge.type()) {
            cerr << "Error: Images must be of the same size and type for PSNR calculation." << endl;
            imageSobelEdge.convertTo(imageSobelEdge, image.type());
            resize(imageSobelEdge, imageSobelEdge, image.size());
        }

        try {
            ssim = getSSIM(image, imageSobelEdge);
            psnr = cv::PSNR(image, imageSobelEdge);
        } catch (const cv::Exception& e) {
            cerr << "Error calculating SSIM/PSNR: " << e.what() << endl;
            ssim = -1;  // Indicate failure
            psnr = -1;
        }

        dataFile << entry.path().filename() << ",Sobel," << duration.count() << "," << memoryUsed << "," << ssim << "," << psnr << "\n";

        cout << "Processed: " << entry.path().filename() << endl;
        string outputPath = outputDir + "/Original_" + entry.path().filename().string();
        imwrite(outputPath, image);
        outputPath = outputDir + "/Gaussian_" + entry.path().filename().string();
        imwrite(outputPath, imageBlurred);
        outputPath = outputDir + "/Median_" + entry.path().filename().string();
        imwrite(outputPath, imageMedianFiltered);
        outputPath = outputDir + "/Sobel_" + entry.path().filename().string();
        imwrite(outputPath, imageSobelEdge);
    }
    dataFile.close();
    cout << "Data recorded successfully." << endl;
    return 0;
}
