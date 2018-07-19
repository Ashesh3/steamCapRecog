#include <opencv2/opencv.hpp>

  using namespace cv;
using namespace std;

void getSimpleDescriptor(Mat & image, Mat & descriptor) {
  float * floatMatrix = (float * ) malloc(image.cols * image.rows * sizeof(float));

  for (int j = 0; j < image.rows; j++)
    for (int k = 0; k < image.cols; k++)
      floatMatrix[j * image.cols + k] = ((float) image.at <unsigned char>(j, k)) / 255.0;

  descriptor = Mat(1, image.cols * image.rows, CV_32FC1, floatMatrix);

  free(floatMatrix);
}

void getHOGDescriptor(Mat & image, Mat & descriptor) {
  vector <float>features;

  HOGDescriptor hog;
  hog.winSize = Size(32, 48);

  hog.compute(image, features, Size(8, 8), Size(0, 0));

  descriptor = Mat(1, (int) features.size(), CV_32FC1, features.data());

  features.clear();
}

void getSimpleTrainingData(Mat & trainingData, Mat & classLabels, string folder, string posLetter, string negLetter, int sampleSize) {
  Mat image;

  // Get dimensions first
  image = imread(folder + posLetter + "/0.png", CV_LOAD_IMAGE_GRAYSCALE);

  trainingData = Mat(sampleSize * 32, image.cols * image.rows, CV_32FC1);
  classLabels = Mat(sampleSize * 32, 1, CV_32FC1);

  image.release();

  for (int i = 0; i < sampleSize; i++) {
	      int n = 1;

    image = imread(folder + "2" + "/" + to_string(i) + ".png", CV_LOAD_IMAGE_GRAYSCALE);
    if (!image.data)
      break;
    for (int j = 0; j < image.rows; j++)
      for (int k = 0; k < image.cols; k++)
        trainingData.at <float>(i * 32, j * image.cols + k) = ((float) image.at <unsigned char>(j, k)) / 255.0;
    classLabels.at <float>(i * 32, 0) = n++;
    image.release();

    image = imread(folder + "3" + "/" + to_string(i) + ".png", CV_LOAD_IMAGE_GRAYSCALE);
    if (!image.data)
      break;

    for (int j = 0; j < image.rows; j++)
      for (int k = 0; k < image.cols; k++)
        trainingData.at <float>(i * 32 + 1, j * image.cols + k) = ((float) image.at <unsigned char>(j, k)) / 255.0;
    classLabels.at <float>(i * 32 + 1, 0) = n++;
    image.release();

    image = imread(folder + "4" + "/" + to_string(i) + ".png", CV_LOAD_IMAGE_GRAYSCALE);
    if (!image.data)
      break;

    for (int j = 0; j < image.rows; j++)
      for (int k = 0; k < image.cols; k++)
        trainingData.at <float>(i * 32 + 2, j * image.cols + k) = ((float) image.at <unsigned char>(j, k)) / 255.0;
    classLabels.at <float>(i * 32 + 2, 0) = n++;
    image.release();

    image = imread(folder + "7" + "/" + to_string(i) + ".png", CV_LOAD_IMAGE_GRAYSCALE);
    if (!image.data)
      break;

    for (int j = 0; j < image.rows; j++)
      for (int k = 0; k < image.cols; k++)
        trainingData.at <float>(i * 32 + 3, j * image.cols + k) = ((float) image.at <unsigned char>(j, k)) / 255.0;
    classLabels.at <float>(i * 32 + 3, 0) = n++;
    image.release();

    image = imread(folder + "8" + "/" + to_string(i) + ".png", CV_LOAD_IMAGE_GRAYSCALE);
    if (!image.data)
      break;

    for (int j = 0; j < image.rows; j++)
      for (int k = 0; k < image.cols; k++)
        trainingData.at <float>(i * 32 + 4, j * image.cols + k) = ((float) image.at <unsigned char>(j, k)) / 255.0;
    classLabels.at <float>(i * 32 + 4, 0) = n++;
    image.release();

    image = imread(folder + "9" + "/" + to_string(i) + ".png", CV_LOAD_IMAGE_GRAYSCALE);
    if (!image.data)
      break;

    for (int j = 0; j < image.rows; j++)
      for (int k = 0; k < image.cols; k++)
        trainingData.at <float>(i * 32 + 5, j * image.cols + k) = ((float) image.at <unsigned char>(j, k)) / 255.0;
    classLabels.at <float>(i * 32 + 5, 0) = n++;
    image.release();


	
	
    int tmpindx = 6;
    // n is 7 here [starts from 7 as A]
    for (int tmp = 65; tmp <= 90; tmp++) {
      if (tmp == 73 || tmp == 79 || tmp == 83)
	  {
		 n++;
       // continue;
	  }
      else {
        image = imread(folder + char(tmp) + "/" + to_string(i) + ".png", CV_LOAD_IMAGE_GRAYSCALE);
		//cout<<tmpindx<<"|"<<n<<" ;";
		//cout<<folder<<folder << char(tmp) << "/" << to_string(i) + ".png"<<endl;
        if (!image.data)
          continue;

        for (int j = 0; j < image.rows; j++)
          for (int k = 0; k < image.cols; k++)
            trainingData.at <float>(i * 32 + tmpindx, j * image.cols + k) = ((float) image.at <unsigned char> (j, k)) / 255.0;
        classLabels.at <float>(i * 32 + tmpindx, 0) = n++;
        image.release();

        tmpindx++;
		

      }

    }
	
    //n is 31 here [31 is  & ]
			//cout<<tmpindx<<"|"<<n<<" ;";

    image = imread(folder + "and" + "/" + to_string(i) + ".png", CV_LOAD_IMAGE_GRAYSCALE);
			//cout<<folder<< "and"<<"/" <<to_string(i) << ".png"<<endl;

    if (!image.data)
      break;
    for (int j = 0; j < image.rows; j++)
      for (int k = 0; k < image.cols; k++)
        trainingData.at <float>(i * 32 + tmpindx, j * image.cols + k) = ((float) image.at <unsigned char>(j, k)) / 255.0;
    classLabels.at <float>(i * 32 + tmpindx, 0) = n++;
    image.release();
tmpindx++;
		//cout<<tmpindx<<"|"<<n<<" ;";

    image = imread(folder + "at" + "/" + to_string(i) + ".png", CV_LOAD_IMAGE_GRAYSCALE);
				//cout<<folder<< "at"<<"/" <<to_string(i) << ".png"<<endl;

    if (!image.data)
      break;

    for (int j = 0; j < image.rows; j++)
      for (int k = 0; k < image.cols; k++)
        trainingData.at <float>(i * 32 + tmpindx, j * image.cols + k) = ((float) image.at <unsigned char>(j, k)) / 255.0;
    classLabels.at <float>(i * 32 + tmpindx, 0) = n++;
    image.release();
tmpindx++;
		//cout<<tmpindx<<"|"<<n<<" ;";

    image = imread(folder + "pct" + "/" + to_string(i) + ".png", CV_LOAD_IMAGE_GRAYSCALE);
				//cout<<folder<< "pct"<<"/" <<to_string(i) << ".png"<<endl;

    if (!image.data)
      break;

    for (int j = 0; j < image.rows; j++)
      for (int k = 0; k < image.cols; k++)
        trainingData.at <float>(i * 32 + tmpindx, j * image.cols + k) = ((float) image.at <unsigned char>(j, k)) / 255.0;
    classLabels.at <float>(i * 32 + tmpindx, 0) = n++;
    image.release();
	

  }
}

void getHOGTrainingData(Mat & trainingData, Mat & classLabels, string folder, string posLetter, string negLetter, int sampleSize) {
  Mat image;
  vector <float>features;

  trainingData = Mat(sampleSize * 2, 540, CV_32FC1);
  classLabels = Mat(sampleSize * 2, 1, CV_32FC1);

  for (int i = 0; i < sampleSize; i++) {
    HOGDescriptor hog;
    hog.winSize = Size(32, 48);

    image = imread(folder + posLetter + "/" + to_string(i) + ".png", CV_LOAD_IMAGE_GRAYSCALE);
    if (!image.data)
      break;
    hog.compute(image, features, Size(8, 8), Size(0, 0));

    for (int j = 0; j < features.size(); j++)
      trainingData.at <float>(i * 2, j) = features[j];

    classLabels.at <float>(i * 2, 0) = 1.0;

    image = imread(folder + negLetter + "/" + to_string(i) + ".png", CV_LOAD_IMAGE_GRAYSCALE);
    if (!image.data)
      break;
    hog.compute(image, features, Size(8, 8), Size(0, 0));

    for (int j = 0; j < features.size(); j++)
      trainingData.at <float>(i * 2 + 1, j) = features[j];

    classLabels.at <float>(i * 2 + 1, 0) = -1.0;
  }
}