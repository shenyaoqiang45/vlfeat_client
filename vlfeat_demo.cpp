#include "vlfeat_demo.h"

/******************************** usefull operation begin *****************************************/
static std::vector<std::string> split(std::string str, std::string pattern)
{
	std::string::size_type pos;
	std::vector<std::string> result;

	str += pattern;//扩展字符串以方便操作
	int size = str.size();

	for (int i = 0; i < size; i++)
	{
		pos = str.find(pattern, i);
		if (pos < size)
		{
			std::string s = str.substr(i, pos - i);
			result.push_back(s);
			i = pos + pattern.size() - 1;
		}
	}
	return result;
}

Mat ConvertToMat(void *vec, int row, int col, int type)
{

	if (type == CV_32FC1)
	{
		float *data = (float*)vec;
		Mat retmat = Mat(row, col, CV_32FC1);
		int k = 0;
		for (int i = 0; i < row; ++i)
		{
			for (int j = 0; j<col; j++)
			{
				//printf("%f",vec[i]);

				//for (int j = 0; j < data.cols; ++j)
				//buffer[k++]= data.at<float>(i, j);
				retmat.at<float>(i, j) = data[k++];
			}
		}
		return retmat;
	}
	else
	{
		int k = 0;
		double *data = (double *)vec;
		Mat retmat = Mat(row, col, CV_64FC1);
		for (int i = 0; i < row; ++i)
		{
			for (int j = 0; j<col; j++)
			{
				//printf("%f",vec[i]);

				//for (int j = 0; j < data.cols; ++j)
				//buffer[k++]= data.at<float>(i, j);
				retmat.at<double>(i, j) = data[k++];
			}
			return retmat;
		}
	}
	// FILE *ft = fopen("testGMM.txt","w");
	//     for (int iter=0;iter<col;iter++)
	//       fprintf(ft,"%f %f\n ",retmat.at<float>(0,iter),vec[iter]);
	//     fclose(ft);
	//free(buffer);
	//printf("%d %d\n",row,col);
	//printf("%u %f\n",vec,vec[0]);
	//int k=0;

}

void ConvertToVec(Mat data, float *vec)
{
	int k = 0;
	for (int i = 0; i < data.rows; ++i)
	for (int j = 0; j < data.cols; ++j)
		vec[k++] = data.at<float>(i, j);
	return;
}

static  Mat ConvertToRowVectors(const vector<Mat> &data)
{
	Mat dst(static_cast<int>(data.size()), data[0].rows*data[0].cols, CV_32FC1);
	for (unsigned int i = 0; i < data.size(); i++)
	{
		Mat image_row = data[i].clone().reshape(1, 1);
		Mat row_i = dst.row(i);
		image_row.convertTo(row_i, CV_32F);
	}
	return dst;
}

void vlfeat_printMat(Mat &mat)
{
	FILE *ftest = fopen("vlfeat_printMat.txt", "w");
	for (int i = 0; i < mat.rows; i++)
	{
		for (int j = 0; j < mat.cols; j++)
		{
			cout << mat.at<float>(i, j) << "  ";
			fprintf(ftest, "%.5f  ", mat.at<float>(i, j));
		}
		cout << endl;
		fprintf(ftest, "\r\n");
	}
	fclose(ftest);
	cout << "Mat rows:" << mat.rows << endl;
	cout << "Mat cols:" << mat.cols << endl;
	return;
}

static void savePCAdata(PCA &pca, const string &file)
{
	FileStorage fs(file, FileStorage::WRITE); //创建XML文件  

	if (!fs.isOpened())
	{
		cerr << "failed to open " << file << endl;
	}
	fs << "eigenvalues" << pca.eigenvalues;
	fs << "eigenvectors" << pca.eigenvectors;
	fs << "mean" << pca.mean;
	fs.release();
	cout << "PCA model is saved.\n";
	return;
}

static void loadPCAdata(PCA &pca, const string &file)
{
	FileStorage f(file, FileStorage::READ);
	Mat eigenvalues, eigenvectors, mean;
	f["eigenvalues"] >> eigenvalues;
	f["eigenvectors"] >> eigenvectors;
	f["mean"] >> mean;
	f.release();

	pca.eigenvalues = eigenvalues;
	pca.eigenvectors = eigenvectors;
	pca.mean = mean;
	cout << "PCA model is loaded.\n";
	return;
}
/******************************** usefull operation begin *****************************************/

/******************************** vlfeat api begin *****************************************/
static void vleat_readImage(const string &path, float*vec)
{
	if (NULL == vec)
	{
		std::cout << "invalid input" << std::endl;
		return;
	}

	cv::Mat img1 = imread(path, IMREAD_COLOR);
	if (!img1.data)
	{
		std::cout << "Could not open or find the image, path:" << path << std::endl;
		return;
	}

	cv::Mat img2;
	cv::resize(img1, img2, Size(IMAGE_NEW_COLS, IMAGE_NEW_ROWS), INTER_LINEAR);

	cv::Mat img3;
	cv::cvtColor(img2, img3, cv::COLOR_BGR2GRAY);

	//cv::imshow("img", img3);
	//cv::waitKey(0);

	int k = 0;
	float scale = 1.0 / 255;
	for (int i = 0; i < IMAGE_NEW_COLS; ++i)
	{
		for (int j = 0; j < IMAGE_NEW_ROWS; ++j)
		{
			vec[k] = img3.at<unsigned char>(i, j) * scale;
			k++;
		}
	}
	return;
}

static  Mat vlfeat_convertToRowVectors(const vector<Mat> &data)
{
	int iRows = NUM_WORDS*NUM_SAMPLES_PER_SAMPLES;
	int iCols = DSIFT_DESCR_SIZE;
	Mat dst(iRows, iCols, CV_32F);

	int k = 0;
	for (int i = 0; i < data.size(); i++)
	{
		Mat oriMat = data[i];
		for (int j = 0; j < oriMat.rows; j++)
		{
			Mat oriRow = oriMat.row(j);
			Mat dstRow = dst.row(k);
			oriRow.convertTo(dstRow, CV_32F);
			k++;
			if (k >= iRows)
			{
				break;
				CV_DbgAssert(0);
			}
		}
	}

	return dst;
}


int vlfeat_train_Encoder(const string &filename)
{
	std::vector<cv::Mat> dsifts;
	std::vector<cv::String> filenames;
	//cv::glob(imgfolder, filenames);
	std::ifstream file(filename.c_str(), ifstream::in);
	std::vector<std::string> result;
	string line;
	int count = 0;

	if (!file)
	{
		string error_message = "No valid input file was given, please check the given filename.";
		CV_Error(Error::StsBadArg, error_message);
	}

	while (getline(file, line))
	{
		result = split(line.c_str(), " ");
		filenames.push_back(result[0]);
		count++;
		if (NUM_IMAGES == count)
		{
			break;
		}
	}
	file.close();

	int step = 3;
	int binSize = 8;
	VlDsiftFilter * vlf;
	/*malloc memory*/
	vlf = vl_dsift_new_basic(IMAGE_NEW_ROWS, IMAGE_NEW_COLS, step, binSize);
	int imgSize = IMAGE_NEW_ROWS*IMAGE_NEW_COLS*sizeof(float);
	float *grayImg = (float*)malloc(imgSize);
	if (NULL == grayImg)
	{
		vl_dsift_delete(vlf);
		return -1;
	}
	memset(grayImg, 0, imgSize);

	/*batch process images*/
	for (int i = 0; i < NUM_IMAGES; i++)
	{
		vleat_readImage(filenames[i].c_str(), grayImg);

		/*call processing function of vl*/
		vl_dsift_process(vlf, grayImg);
		int pointNum = vl_dsift_get_keypoint_num(vlf);
		Mat descrMat = Mat(NUM_DESCR_PER_IMAGES, DSIFT_DESCR_SIZE, CV_32FC1);
		srand((int)time(0));
		for (int j = 0; j < NUM_DESCR_PER_IMAGES; j++)
		{
			/*radnom select NUM_DESCR_PER_IMAGES points from pointNum*/
			//srand((int)time(0)); Warning:time is too short, and it leads to have the same randIndex.
			int randIndex = RANDM(pointNum);
			for (int k = 0; k < DSIFT_DESCR_SIZE; k++)
			{
				descrMat.at<float>(j, k) = vlf->descrs[randIndex*DSIFT_DESCR_SIZE + k];
			}

		}
		//vlfeat_printMat(descrMat);

		dsifts.push_back(descrMat);

		/*init var*/
		memset(grayImg, 0, imgSize);
	}


	cv::Mat dsiftMat = vlfeat_convertToRowVectors(dsifts);
	//vlfeat_printMat(dsiftMat);

	int dimension = DSIFT_DESCR_SIZE;
	int numClusters = NUM_WORDS;
	if (0)
	{
		/*Temporarily not applicable, due to unsatisfied effect*/
		/*save pca model*/
		PCA pca(dsiftMat, cv::Mat(), PCA::DATA_AS_ROW, NUM_PCA_DIM);
		savePCAdata(pca, ".\\DSIFT_PCA_MODEL.xml");
		dimension = NUM_PCA_DIM;
		Mat dsiftPCAMat = pca.project(dsiftMat);
	}

	Mat dsiftGMM = dsiftMat;
	/*gmm*/
	// create a new instance of a GMM object for float data
	VlGMM * gmm = vl_gmm_new(VL_TYPE_FLOAT, dimension, numClusters);
	// set the maximum number of EM iterations to 100
	vl_gmm_set_max_num_iterations(gmm, 100);
	// set the initialization to random selection
	vl_gmm_set_initialization(gmm, VlGMMKMeans);
	// cluster the data, i.e. learn the GMM
	vl_gmm_cluster(gmm, (void*)dsiftGMM.data, dsiftGMM.rows);

	float * means;
	float * covariances;
	float * priors;
	means = (float *)vl_gmm_get_means(gmm);
	covariances = (float *)vl_gmm_get_covariances(gmm);
	priors = (float *)vl_gmm_get_priors(gmm);

	Mat measMat = ConvertToMat(means, numClusters, dimension, CV_32FC1);
	Mat covarMat = ConvertToMat(covariances, numClusters, dimension, CV_32FC1);
	Mat priorsMat = ConvertToMat(priors, 1, numClusters, CV_32FC1);
	//vlfeat_printMat(priorsMat);

	/*save gmm model */
	FileStorage fs(".\\DSIFT_GMM_MODEL.xml", FileStorage::WRITE);
	fs << "numClusters" << numClusters;
	fs << "dimension" << dimension;
	fs << "measMat" << measMat;
	fs << "covarMat" << covarMat;
	fs << "priorsMat" << priorsMat;
	fs.release();
	// run fisher encoding
	// allocate space for the encoding
	//int encSize = sizeof(float)* 2 * dimension * numClusters;
	//char* enc = (char*)vl_malloc(encSize);
	//memset(enc, 0, encSize);
	//vl_size res = 0;
	//res = vl_fisher_encode
	//	(enc, VL_TYPE_FLOAT,
	//	measMat.data, dimension, numClusters,
	//	covarMat.data,
	//	priorsMat.data,
	//	dsifts[1].data, 1,
	//	VL_FISHER_FLAG_IMPROVED
	//	);

	//Mat encMat(6, 128, CV_32FC1);
	//memcpy(encMat.data, enc, encSize);
	//vlfeat_printMat(encMat);

	/*free memory*/
	free(grayImg);

	vl_dsift_delete(vlf);

	vl_gmm_delete(gmm);

	//vl_free(enc);
	return 0;
}

int vlfeat_encode_images(const string& filename, vector<Mat>& features, vector<int>& labels)
{
	std::ifstream file(filename.c_str(), ifstream::in);
	std::vector<std::string> result;
	string line;
	int iResult = 0;

	if (!file)
	{
		string error_message = "No valid input file was given, please check the given filename.";
		CV_Error(Error::StsBadArg, error_message);
	}
	/* load GMM model*/
	FileStorage fs(".\\DSIFT_GMM_MODEL.xml", FileStorage::READ);
	int dimension = 0;
	int numClusters = 0;
	Mat measMat;
	Mat covarMat;
	Mat priorsMat;
	fs["numClusters"] >> numClusters;
	fs["dimension"] >> dimension;
	fs["measMat"] >> measMat;
	fs["covarMat"] >> covarMat;
	fs["priorsMat"] >> priorsMat;
	fs.release();

	float sum = 0.0;
	for (int i = 0; i < priorsMat.rows; ++i)
	for (int j = 0; j < priorsMat.cols; ++j)
		sum = sum + priorsMat.at<float>(i, j);
	cout << "priorsMat sum:" << sum << endl;

	if (0)
	{
		/*Temporarily not applicable, due to unsatisfied effect*/
		/* load pca model*/
		PCA pca;
		loadPCAdata(pca, "DSIFT_PCA_MODEL.xml");
	}


	/*transform dsift to FV by GMM*/
	int step = 3;
	int binSize = 8;
	VlDsiftFilter * vlf;
	/*malloc memory*/
	vlf = vl_dsift_new_basic(IMAGE_NEW_ROWS, IMAGE_NEW_COLS, step, binSize);
	int imgSize = IMAGE_NEW_ROWS*IMAGE_NEW_COLS*sizeof(float);
	float *grayImg = (float*)malloc(imgSize);
	if (NULL == grayImg)
	{
		vl_dsift_delete(vlf);
		return -1;
	}
	memset(grayImg, 0, imgSize);

	int encSize = sizeof(float)* 2 * dimension * numClusters;
	int fishVecSize = 2 * dimension * numClusters;
	// allocate space for the encoding
	char* enc = (char*)vl_malloc(encSize);
	if (NULL == enc)
	{
		vl_dsift_delete(vlf);
		free(grayImg);
		return -1;
	}
	memset(enc, 0, encSize);
	while (getline(file, line))
	{
		result = split(line.c_str(), " ");
		
		iResult = atoi(result[1].c_str());
		vleat_readImage(result[0], grayImg);

		vl_dsift_process(vlf, grayImg);
		
		if (0 == vlf->numFrames)
		{
			iResult = 0;
			memset(grayImg, 0, imgSize);
			memset(enc, 0, encSize);
			cout << "vl_dsift_process failed " << endl;
			continue;
		}

		/*pca process*/
		Mat descrsMat = ConvertToMat(vlf->descrs, vlf->numFrames, vlf->descrSize, CV_32FC1);
		//Mat descrsFisherMat = pca.project(descrsMat);
		Mat descrsFisherMat = descrsMat;

		// run fisher encoding
		vl_size res = 0;
		res = vl_fisher_encode
			(enc, VL_TYPE_FLOAT,
			measMat.data, dimension, numClusters,
			covarMat.data,
			priorsMat.data,
			descrsFisherMat.data, vlf->numFrames,
			VL_FISHER_FLAG_IMPROVED
			);		

		/*save feature and labels*/
		labels.push_back(iResult);

		Mat FishVec(1, fishVecSize, CV_32FC1, Scalar(0));
		memcpy(FishVec.data, enc, encSize);
		features.push_back(FishVec);

		//vlfeat_printMat(FishVec);
		/*init var*/
		iResult = 0;
		memset(grayImg, 0, imgSize);
		memset(enc, 0, encSize);
	}

	/*free memory*/
	file.close();

	free(grayImg);

	vl_dsift_delete(vlf);

	vl_free(enc);
	return 0;
}
/********************************  vlfeat api end *****************************************/

/******************************** SVM begin *****************************************/
static Ptr<TrainData> prepare_train_data(const Mat& data, const Mat& responses, int ntrain_samples)
{
	Mat sample_idx = Mat::zeros(1, data.rows, CV_8U);
	Mat train_samples = sample_idx.colRange(0, ntrain_samples);
	train_samples.setTo(Scalar::all(1));

	int nvars = data.cols;
	Mat var_type(nvars + 1, 1, CV_8U);
	var_type.setTo(Scalar::all(VAR_ORDERED));
	var_type.at<uchar>(nvars) = VAR_CATEGORICAL;

	return TrainData::create(data, ROW_SAMPLE, responses,
		noArray(), sample_idx, noArray(), var_type);
}

static void test_and_save_classifier(const Ptr<StatModel>& model,
	const Mat& data, const Mat& responses,
	int ntrain_samples, int rdelta,
	const string& filename_to_save)
{
	int i, nsamples_all = data.rows;
	double train_hr = 0, test_hr = 0;

	// compute prediction error on train and test data
	for (i = 0; i < nsamples_all; i++)
	{
		Mat sample = data.row(i);

		float r = model->predict(sample);
		cout << r << endl;

		r = std::abs(r + rdelta - responses.at<int>(i)) <= FLT_EPSILON ? 1.f : 0.f;

		if (i < ntrain_samples)
			train_hr += r;
		else
			test_hr += r;
	}

	test_hr /= nsamples_all - ntrain_samples;
	train_hr = ntrain_samples > 0 ? train_hr / ntrain_samples : 1.;

	printf("Recognition rate: train = %.1f%%, test = %.1f%%\n",
		train_hr*100., test_hr*100.);

	if (!filename_to_save.empty())
	{
		model->save(filename_to_save);
	}
}

int opencv_train_svm(const string& data_filename, const string& filename_to_save)
{
	vector<Mat> features;	// vector to hold the images
	vector<int> labels;	// vector to hold the images
	int iSamples = 0;

	(void)vlfeat_encode_images(data_filename, features, labels);
	if (features.size() != labels.size())
	{
		return -1;
	}

   	iSamples = features.size();
	Mat responses = Mat(iSamples, 1, CV_32S);
	memcpy(responses.data, labels.data(), labels.size()*sizeof(int));
	Mat data = ConvertToRowVectors(features);

	Ptr<SVM> model;
	int nsamples_all = data.rows;
	int ntrain_samples = (int)(nsamples_all*0.7);

	// create classifier by using <data> and <responses>
	cout << "Training the classifier ...\n";
	Ptr<TrainData> tdata = prepare_train_data(data, responses, ntrain_samples);
	model = SVM::create();
	model->setType(SVM::C_SVC);
	model->setKernel(SVM::LINEAR);
	//model->setGamma(1);
	//model->setC(10);
	model->setC(1);
	model->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 1000, 1e-6));
	model->train(tdata);
	cout << endl;

	test_and_save_classifier(model, data, responses, ntrain_samples, 0, filename_to_save);
	return true;
}
/******************************** SVM end *****************************************/

/******************************** demo begin *****************************************/
void vlfeat_svm_demo()
{
	vl_size const numData = 4;
	vl_size const dimension = 2;
	double x[dimension * numData] = {
		0.0, -0.5,
		0.6, -0.3,
		0.0, 0.5,
		0.6, 0.0 };
	double y[numData] = { 1, 1, -1, 1 };
	double lambda = 0.01;
	double const *model;
	double bias;
	VlSvm * svm = vl_svm_new(VlSvmSolverSgd,
		x, dimension, numData,
		y,
		lambda);
	vl_svm_train(svm);
	model = vl_svm_get_model(svm);
	bias = vl_svm_get_bias(svm);
	printf("model w = [ %f , %f ] , bias b = %f \n",
		model[0],
		model[1],
		bias);
	vl_svm_delete(svm);
	return;
}

void vlfeat_dsift_demo()
{
	cv::Mat img1;
	img1 = imread("F:/nazhi_Daniel/C++/fish.jpg", IMREAD_COLOR);
	if (!img1.data)         // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return;
	}

	int iCols = img1.cols;
	int iRows = img1.rows;
	float scale = 1.0;

	if (iRows > IMAGE_NEW_ROWS)
	{
		scale = IMAGE_NEW_ROWS*1.0 / iRows;
		iCols = (int)(scale * iCols);
		iRows = IMAGE_NEW_ROWS;
	}

	cv::Mat img2;
	cv::resize(img1, img2, Size(iCols, iRows), INTER_LINEAR);

	cv::Mat img3;
	cv::cvtColor(img2, img3, cv::COLOR_BGR2GRAY);

	cv::imshow("test", img3);
	cv::waitKey(0);

	int k = 0;
	float *grayImg = (float*)malloc(iRows*iCols*sizeof(float));
	for (int i = 0; i < iRows; ++i)
	{
		for (int j = 0; j < iCols; ++j)
		{
			grayImg[k] = img3.at<unsigned char>(i, j);
			grayImg[k] = grayImg[k] / 255;
			k++;
		}
	}

	/*get dsift feature*/
	VlDsiftFilter * vlf;
	int step = 3;
	int binSize = 8;
	vlf = vl_dsift_new_basic(iRows, iCols, step, binSize);
	/*call processing function of vl*/
	vl_dsift_process(vlf, grayImg);
	int pointNum, descrSize;
	pointNum = vl_dsift_get_keypoint_num(vlf);
	descrSize = vl_dsift_get_descriptor_size(vlf);
	//cout << pointNum << "   " << descrSize << endl; 

	/*get frame  and scale*/
	const float* descriptor;
	VlDsiftKeypoint* framescale;
	descriptor = vl_dsift_get_descriptors(vlf);
	framescale = (VlDsiftKeypoint*)vl_dsift_get_keypoints(vlf);

	/*print*/
	//cout << "***********************the first descriptor is:"<<endl;
	//cout << "x: "<<vlf->frames[0].x<<endl;
	//cout << "y: "<<vlf->frames[0].y<<endl;
	//cout << "s: "<<vlf->frames[0].s<<endl;
	//cout << "norm: "<<vlf->frames[0].norm<<endl;
	//for(int i=0; i < descrSize; i++)
	//{
	//    cout << vlf->descrs[i] << endl;
	//}

	/*free memory*/
	free(grayImg);

	vl_dsift_delete(vlf);
	return;
}

void vlfeat_printMat_demo()
{
	cv::Mat mat(3, 4, CV_32FC1);
	cv::Mat mat2(3, 4, CV_32FC1, Scalar(2));

	for (int i = 0; i < mat.rows; i++)
	{
		for (int j = 0; j < mat.cols; j++)
		{
			mat.at<float>(i, j) = i*mat.cols + j;
		}
	}
	memcpy(mat2.data, mat.data, mat.rows*mat.cols*sizeof(float));
	vlfeat_printMat(mat2);
	return;
}

void opencv_filestorage_demo()
{
	Mat mat = Mat::eye(Size(12, 12), CV_8UC1);
	FileStorage fs(".\\vocabulary.xml", FileStorage::WRITE);
	fs << "vocabulary" << mat;
	fs.release();

	FileStorage fsOut(".\\vocabulary.xml", FileStorage::READ);
	Mat mat_vocabulary;
	fsOut["vocabulary"] >> mat_vocabulary;
	cout << mat_vocabulary << endl;
	fs.release();
	return;
}


Ptr<SVM> vlfeat_load_SVM_model(const string& filename_to_load)
{
	Ptr<SVM> model = StatModel::load<SVM>(filename_to_load);
	if (model.empty())
		cout << "Could not read the classifier " << filename_to_load << endl;
	else
		cout << "The classifier " << filename_to_load << " is loaded.\n";

	return model;
}

void vlfeat_load_GMM_model(void **pvlf, GMM_PARA_S &stGmm)
{
	/* load GMM model*/
	FileStorage fs("DSIFT_GMM_MODEL.xml", FileStorage::READ);
	fs["numClusters"] >> stGmm.numClusters;
	fs["dimension"] >> stGmm.dimension;
	fs["measMat"] >>stGmm.measMat;
	fs["covarMat"] >> stGmm.covarMat;
	fs["priorsMat"] >> stGmm.priorsMat;    
	fs.release();

	int step = 3;
	int binSize = 8;
	/*malloc memory*/
	*pvlf = vl_dsift_new_basic(IMAGE_NEW_ROWS, IMAGE_NEW_COLS, step, binSize);
	return;
}

void vlfeat_free_GMM_model(void *vlf)
{
	vl_dsift_delete((VlDsiftFilter *)vlf);
	return;
}

float vlfeat_test_demo(const string &path)
{
	Ptr<SVM> model = vlfeat_load_SVM_model("DSIFT_GMM_SVM_MODEL");
	VlDsiftFilter *vlf;
	GMM_PARA_S stGmm;
	vlfeat_load_GMM_model((void**)&vlf, stGmm);
	/*transform dsift to FV by GMM*/

	int dimension = stGmm.dimension;
	int numClusters = stGmm.numClusters;
	int imgSize = IMAGE_NEW_ROWS*IMAGE_NEW_COLS*sizeof(float);
	float *grayImg = (float*)malloc(imgSize);
	if (NULL == grayImg)
	{
		return -1;
	}
	memset(grayImg, 0, imgSize);

	int encSize = sizeof(float)* 2 * dimension * numClusters;
	int fishVecSize = 2 * dimension * numClusters;
	// allocate space for the encoding
	char* enc = (char*)vl_malloc(encSize);
	if (NULL == enc)
	{
		vl_dsift_delete(vlf);
		free(grayImg);
		return -1;
	}
	memset(enc, 0, encSize);

	vleat_readImage(path, grayImg);

	vl_dsift_process(vlf, grayImg);

	// run fisher encoding
	vl_size res = 0;
	res = vl_fisher_encode
		(enc, VL_TYPE_FLOAT,
		stGmm.measMat.data, dimension, numClusters,
		stGmm.covarMat.data,
		stGmm.priorsMat.data,
		vlf->descrs, vlf->numFrames,
		VL_FISHER_FLAG_IMPROVED
		);

	Mat FishVec(1, fishVecSize, CV_32FC1, Scalar(0));
	memcpy(FishVec.data, enc, encSize);

	float r = model->predict(FishVec);

	/*free memory*/
	free(grayImg);

	vl_free(enc);

	vlfeat_free_GMM_model(vlf);
	return r;
}
/******************************** demo end *****************************************/