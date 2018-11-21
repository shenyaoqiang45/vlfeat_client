#ifndef _VLFEAT_DEMO_H_
#define _VLFEAT_DEMO_H_

#include <cstdio>
#include <vector>
#include <iostream>
#include <fstream>
#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<time.h>


#include "vlfeat_common_header.h"

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include "opencv2/ml.hpp"

using namespace cv;
using namespace std;
using namespace cv::ml;

#define IMAGE_NEW_ROWS (100)
#define IMAGE_NEW_COLS (100)

#define NUM_WORDS (64)
#define NUM_SAMPLES_PER_SAMPLES (1000)
#define NUM_IMAGES (800)
#define NUM_DESCR_PER_IMAGES (NUM_WORDS*NUM_SAMPLES_PER_SAMPLES/NUM_IMAGES)
#define NUM_PCA_DIM (80)
#define DSIFT_DESCR_SIZE (128)

#define RANDM(x) (rand()%x)

void vlfeat_svm_demo();

void vlfeat_dsift_demo();

void vlfeat_printMat_demo();

void opencv_filestorage_demo();

int vlfeat_train_Encoder(const string &imgfolder);

int vlfeat_encode_images(const string& filename, vector<Mat>& features, vector<int>& labels);

int opencv_train_svm(const string& data_filename, const string& filename_to_save);

#endif