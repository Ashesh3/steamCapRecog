#include <algorithm>
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

#include "main.hpp"
#include "imagereconstruct.hpp"
#include "image.hpp"
#include "segments.hpp"
#include "descriptors.hpp"
#include "classify.hpp"
#include "misc.hpp"

using namespace std;
using namespace cv;
namespace fs = boost::filesystem;

int main(int argc, char ** argv) {
    const int RESIZE_FACTOR = 2;
    const string DATA_PATH = "/workspace/steam/opencv-steam-captcha/data";
    const string OUTPUT_PATH = "/workspace/steam/opencv-steam-captcha/output/";
int notX=1;
    // 0, 1, 5, 6, I, O and S are never used
    const string ALLOWED_CHARS = "234789ABCDEFGHJKLMNPQRTUVWXYZ@&%";

    // Images
    Mat sourceImage, finalImage, histogramImage;
    Mat histogram;

    // Initialize character counters
    map < string, int > counter;
    for (int i = 0; i < ALLOWED_CHARS.length(); i++) {
      string letter(1, ALLOWED_CHARS[i]);
      counter[letter] = 0;
    }

    // Check if data folder exists
    fs::path folder(DATA_PATH);
    if (!exists(folder))
      return -1;

    // Create output folder structure
    if (!createFolderStructure(OUTPUT_PATH, ALLOWED_CHARS))
      return -1;

    fs::directory_iterator endItr;
    for (fs::directory_iterator itr(folder); itr != endItr; itr++) {
      string fullPath = itr -> path().string();
      string fileName = itr -> path().filename().string();

      // Skip all dot files
      if (fileName[0] == '.')
        continue;

      // Retrieve captcha string
      string captchaCode = boost::replace_all_copy(fileName, ".png", "");
      captchaCode = aliasToSpecialChar(captchaCode);

      // Load our base image
      sourceImage = imread(fullPath, CV_LOAD_IMAGE_GRAYSCALE);

      // Is it loaded?
      if (!sourceImage.data)
        return -1;

      // Resize the image by resize factor
      resize(sourceImage, sourceImage, Size(sourceImage.cols * RESIZE_FACTOR, sourceImage.rows * RESIZE_FACTOR));

      // Define our final image
      finalImage = sourceImage.clone();

      // Apply adaptive threshold
      //adaptiveThreshold(finalImage, finalImage, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 3, 1);

      // Use the thresholded image as a mask
      Mat tmp;
      sourceImage.copyTo(tmp, finalImage);
      tmp.copyTo(finalImage);
      tmp.release();

      // Let's calculate histogram for our image
      histogram = createHistogram(finalImage);

      // Calculate final threshold value
      int thresholdValue = getIdealThreshold(histogram);

      // Draw histogram image
      histogramImage = drawHistogram(histogram, thresholdValue);

      // Apply binary threshold
      //threshold(finalImage, finalImage, thresholdValue, 255, THRESH_BINARY);

      // Morphological closing
      Mat element = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
      dilate(finalImage, finalImage, element);
      erode(finalImage, finalImage, element);

	  
      // Segments
      int * segH = horizontalSegments(finalImage);
      int * segV = verticalSegments(finalImage);

      Mat segHImage = drawHorizontalSegments(segH, finalImage.rows, finalImage.cols);
      Mat segVImage = drawVerticalSegments(segV, finalImage.rows, finalImage.cols);

      // Create pairs
      vector < pair < int, int > > verticalPairs = filterVerticalPairs(createSegmentPairs(segV, finalImage.rows));
      vector < pair < int, int > > horizontalPairs = splitLarge(filterHorizontalPairs(createSegmentPairs(segH, finalImage.cols), finalImage.cols));

      // Get segment squares
      vector < Rectangle > squares = takeRectangles(shrinkRectangles(finalImage, getRectangles(verticalPairs, horizontalPairs)), 6);

      // Save the squares
      saveRectangles(sourceImage, squares, OUTPUT_PATH, captchaCode, counter);
      // Let's draw the rectangles
      drawRectangles(finalImage, squares);
      drawRectangles(sourceImage, squares);

      // Display the images if necessary

	  if(notX ==12){
       imshow("Final image", finalImage);
       imshow("Source image", sourceImage);
       imshow("HSeg", segHImage);
       imshow("VSeg", segVImage);
       imshow("Histogram", histogramImage);
       waitKey();
	   notX=2;
	}
      sourceImage.release();
      finalImage.release();
    }

    Mat trainingData, classLabels;

    // Create training data
    getSimpleTrainingData(trainingData, classLabels, OUTPUT_PATH, "G", "Y", 10);

    // This part is highly simplified and was mostly done just to test
    // classification based on two characters with the highest frequency
    CvSVMParams params;
    params.svm_type = CvSVM::C_SVC;
    params.kernel_type = CvSVM::LINEAR;
    params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

    CvSVM SVM;
    SVM.train(trainingData, classLabels, Mat(), Mat(), params);

    int success = 0;
	int fail=0;
    for (int i = 0; i < 10; i++) {

      Mat letterImage = imread(OUTPUT_PATH + "2/" + to_string(i) + ".png", CV_LOAD_IMAGE_GRAYSCALE);
      float result = classify(SVM, letterImage);
      if (result == 1) {
        cout << "2 classified as: 2" << endl;
        success++;
      } else {
		cout << "!!! 2 misrecognised" << endl;

        // cout << "FAIL G" << endl; 
 fail++;
      }
	  letterImage.release();

	  letterImage = imread(OUTPUT_PATH + "3/" + to_string(i) + ".png", CV_LOAD_IMAGE_GRAYSCALE);
      result = classify(SVM, letterImage);
      if (result == 2) {
        cout << "3 classified as: 3" << endl;
        success++;
      } else {
		  		cout << "!!! 3 misrecognised" << endl;

        // cout << "FAIL G" << endl; 
 fail++;
      }
      letterImage.release();
	letterImage = imread(OUTPUT_PATH + "4/" + to_string(i) + ".png", CV_LOAD_IMAGE_GRAYSCALE);
      result = classify(SVM, letterImage);
      if (result == 3) {
        cout << "4 classified as: 4" << endl;
        success++;
      } else {
		  		cout << "!!! 4 misrecognised" << endl;

        // cout << "FAIL G" << endl; 
		
 fail++;
      }
      letterImage.release();
	  
	  
	  letterImage = imread(OUTPUT_PATH + "7/" + to_string(i) + ".png", CV_LOAD_IMAGE_GRAYSCALE);
      result = classify(SVM, letterImage);
      if (result == 4) {
        cout << "7 classified as: 7" << endl;
        success++;
      } else {
		  		cout << "!!! 7 misrecognised" << endl;

        // cout << "FAIL G" << endl; 
 fail++;
      }
      letterImage.release();
	  
	  letterImage = imread(OUTPUT_PATH + "8/" + to_string(i) + ".png", CV_LOAD_IMAGE_GRAYSCALE);
      result = classify(SVM, letterImage);
      if (result == 5) {
        cout << "8 classified as: 8" << endl;
        success++;
      } else {
		  		cout << "!!! 8 misrecognised" << endl;

        // cout << "FAIL G" << endl; 
 fail++;
      }
      letterImage.release();
	  
	  letterImage = imread(OUTPUT_PATH + "9/" + to_string(i) + ".png", CV_LOAD_IMAGE_GRAYSCALE);
      result = classify(SVM, letterImage);
      if (result == 6) {
        cout << "9 classified as: 9" << endl;
        success++;
      } else {
		  		cout << "!!! 9 misrecognised" << endl;

        // cout << "FAIL G" << endl; 
 fail++;
      }
      letterImage.release();
	  
	  letterImage = imread(OUTPUT_PATH + "and/" + to_string(i) + ".png", CV_LOAD_IMAGE_GRAYSCALE);
      result = classify(SVM, letterImage);
      if (result == 33) {
        cout << "& classified as: &" << endl;
        success++;
      } else {
		  		cout << "!!! & misrecognised" << endl;

        // cout << "FAIL G" << endl; 
 fail++;
      }
      letterImage.release();
	  
	  letterImage = imread(OUTPUT_PATH + "at/" + to_string(i) + ".png", CV_LOAD_IMAGE_GRAYSCALE);
      result = classify(SVM, letterImage);
      if (result == 34) {
        cout << "@ classified as: @" << endl;
        success++;
      } else {
		  		cout << "!!! @ misrecognised" << endl;

        // cout << "FAIL G" << endl; 
 fail++;
      }
      letterImage.release();
  

  	  letterImage = imread(OUTPUT_PATH + "pct/" + to_string(i) + ".png", CV_LOAD_IMAGE_GRAYSCALE);
      result = classify(SVM, letterImage);
      if (result == 35) {
        cout << "% classified as: %" << endl;
        success++;
      } else {
		  		cout << "!!! % misrecognised" << endl;

        // cout << "FAIL G" << endl; 
 fail++;
      }
      letterImage.release();
  

     
	  for (int tmp = 66; tmp <= 90; tmp++) {
        if (tmp == 73 || tmp == 79 || tmp == 83)
          continue;
        else {

          letterImage = imread(OUTPUT_PATH + char(tmp)+"/" + to_string(i) + ".png", CV_LOAD_IMAGE_GRAYSCALE);
          result = classify(SVM, letterImage);
          //cout<<"R: "<<result<<endl;

          if (result == tmp-66+8) {
            cout << "Letter "<<char(tmp)<<" classified as: "<<char(result-7+65) << endl;
            success++;
          } else {
            // cout << "FAIL Y" << endl;
			            cout << "!!!Letter "<<char(tmp)<<" classified as: "<<char(result-7+65) <<"[R:"<<result<<"]"<<endl;
						fail++;
          }
          letterImage.release();

        }

      }
	
	 
	 
      trainingData.release();
      classLabels.release();
}
float rate = ((success*100/(success+fail)));
      cout << "Success rate: " << success << "/" << (success+fail) <<" = "<< rate<<"%"<< endl;

      return 0;
    }