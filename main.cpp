#include "vlfeat_common_header.h"
#include "vlfeat_demo.h"

#define use _CRT_SECURE_NO_WARNINGS

int main(int argc, const char * argv[]) {
	//VL_PRINT("Hello world!\n");
	//vlfeat_svm_demo();
	//vlfeat_dsift_demo();
	//opencv_filestorage_demo();

	//vlfeat_train_Encoder("F:/nazhi_Daniel/datas/SIW/GMM/");
	//vector<Mat> features;
	//vector<int> labels;
	//(void)vlfeat_encode_images("F:/nazhi_Daniel/datas/SIW/GMM/gmm.txt", features, labels);

	string dataFile = "F:/nazhi_Daniel/datas/SIW/train_random.txt";
	/*step1 vlfeat_train_Encoder done*/
	//vlfeat_train_Encoder(dataFile);

	/*step2 opencv_train_svm*/
	//(void)opencv_train_svm(dataFile, "DSIFT_GMM_SVM_MODEL");
	(void)vlfeat_test_demo("spoof_1.jpg");
	return 0;
}