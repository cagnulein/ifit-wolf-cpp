#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <paddleocr.h>

using namespace cv;
using namespace std;

// File path
const string ocrfileName = "ocr-output.txt";
const string ocrlogFile = "ocr-logfile.txt";

int main() {
    // Take Zwift screenshot
    Mat screenshot;
    screenshot = imread("zwift_screenshot.jpg");

    // Scale image to 3000 x 2000
    resize(screenshot, screenshot, Size(3000, 2000));

    // Convert screenshot to a numpy array
    Mat screenshot_np = screenshot;

    // Crop image to incline area
    int screenwidth = screenshot.cols;
    int screenheight = screenshot.rows;
    int col1 = static_cast<int>(screenwidth / 3000.0 * 2800);
    int row1 = static_cast<int>(screenheight / 2000.0 * 75);
    int col2 = screenwidth;
    int row2 = static_cast<int>(screenheight / 2000.0 * 200);
    Mat cropped_np = screenshot_np(Range(row1, row2), Range(col1, col2));

    // Convert numpy array to PIL image
    Mat cropped_pil;
    cropped_np.convertTo(cropped_pil, CV_8UC3);

    // Convert PIL Image to a cv2 image
    Mat cropped_cv2;
    cvtColor(cropped_pil, cropped_cv2, COLOR_RGB2BGR);

    // Convert cv2 image to HSV
    Mat image;
    cropped_cv2.copyTo(image);
    cvtColor(cropped_cv2, image, COLOR_BGR2HSV);

    // Isolate white mask
    Scalar lower(0, 0, 159);
    Scalar upper(0, 0, 255);
    Mat mask0;
    inRange(image, lower, upper, mask0);
    Mat result0;
    bitwise_and(cropped_cv2, cropped_cv2, result0, mask0);

    // Isolate yellow mask
    lower = Scalar(24, 239, 241);
    upper = Scalar(24, 253, 255);
    Mat mask1;
    inRange(image, lower, upper, mask1);
    Mat result1;
    bitwise_and(cropped_cv2, cropped_cv2, result1, mask1);

    // Isolate orange mask
    lower = Scalar(8, 191, 243);
    upper = Scalar(8, 192, 243);
    Mat mask2;
    inRange(image, lower, upper, mask2);
    Mat result2;
    bitwise_and(cropped_cv2, cropped_cv2, result2, mask2);

    // Isolate red mask
    lower = Scalar(0, 255, 255);
    upper = Scalar(10, 255, 255);
    Mat mask3;
    inRange(image, lower, upper, mask3);
    Mat result3;
    bitwise_and(cropped_cv2, cropped_cv2, result3, mask3);

    // Join colour masks
    Mat mask = mask0 + mask1 + mask2 + mask3;

    // Set output image to zero everywhere except mask
    Mat merge = image.clone();
    merge.setTo(0, mask == 0);

    // Convert to grayscale
    Mat gray;
    cvtColor(merge, gray, COLOR_BGR2GRAY);

    // Convert to black/white by threshold
    Mat bin;
    threshold(gray, bin, 30, 255, THRESH_BINARY);

    // Closing
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    Mat closing;
    morphologyEx(bin, closing, MORPH_CLOSE, kernel);

    // Invert black/white
    Mat inv;
    bitwise_not(closing, inv);

    // Apply average blur
    Mat averageBlur;
    blur(inv, averageBlur, Size(3, 3));

    // OCR image
    paddleocr::PaddleOCR ocr;
    auto result = ocr.ocr("zwiftImage.jpg", false, true, true, false, false, false, false, true, true);

    // Extract OCR text
    string ocr_text = "";
    for (const auto& line : result) {
        for (const auto& word : line) {
            ocr_text += word[1][0];
        }
    }

    // Remove all characters that are not "-" and integers from OCR text
    regex pattern("[^-\\d]+");
    ocr_text = regex_replace(ocr_text, pattern, "");
    string incline;
    if (!ocr_text.empty()) {
        incline = ocr_text;
    } else {
        incline = "None";
    }

    // Write OCR text to log file
    ofstream file(ocrlogFile, ios::app);
    if (file.is_open()) {
        string dt = to_string(time(0));
        file << dt << ", " << incline << "\n";
        file.close();
    }

    // Write OCR text to file
    ofstream outfile(ocrfileName);
    if (outfile.is_open()) {
        outfile << ocr_text;
        outfile.close();
    }

    return 0;
}
