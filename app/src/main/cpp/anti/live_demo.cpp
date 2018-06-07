#include <iostream>  
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
//#include "opencv2/imgproc/imgproc.hpp"  
//#include "opencv2/objdetect/objdetect.hpp"
#include <string>
#include <fstream>
#include <string>
#include "antiSpofModel.h"
#include "featureExtractor.h"
#include "faceDetector.h"
//#include "ml.hpp"
#include "linear.h"
#include "../vl/fisher.h"

//#include<ml/ml.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/objdetect/objdetect.hpp>

using namespace std;
using namespace cv;
using namespace cv::ml;

#define max(a,b) a>b?a:b
#define min(a,b) a>b?b:a
#define Random(x) (rand() % x)

int featureDim = 384;
int numPcaBase = 300;
int numGmmClusters = 128;

#define DETECT_BUFFER_SIZE 0x20000


string int2Str(int i)
{
	if (i == 0)
		return "0";
	string result = "";
	while (i>0)
	{
		int res = i % 10;
		char s = '0' + res;
		result = s + result;
		i /= 10;
	}

	return result;
}

void load_svm(svmModel*& svm, std::string filename)
{
	ifstream fin(filename.c_str());
	if (!fin)
	{
		std::cout << "Unable to open file: svm.dat " << endl;
		exit(1);
	}

	int nr_feature;
	int n;
	int nr_class;
	double bias;
	parameter & param = svm->param;
	svm->label = NULL;
	std::string cmd;
	while (1)
	{
		fin >> cmd;
		if (cmd == "solver_type")
		{
			fin >> cmd;
			for (int i = 0; solver_type_table[i]; i++)
			{
				if (solver_type_table[i] == cmd)
				{
					param.solver_type = i;
					break;
				}
			}
		}
		else if (cmd == "nr_class")
		{
			fin >> nr_class;
			svm->nr_class = nr_class;
		}
		else if (cmd == "nr_feature")
		{
			fin >> nr_feature;
			svm->nr_feature = nr_feature;
		}
		else if (cmd == "bias")
		{
			fin >> bias;
			svm->bias = bias;

		}
		else if (cmd == "w")
		{
			nr_feature = svm->nr_feature;
			if (svm->bias >= 0)
				n = nr_feature + 1;
			else
				n = nr_feature;
			int w_size = n;
			int nr_w;
			if (nr_class == 2 && param.solver_type != MCSVM_CS)
				nr_w = 1;
			else
				nr_w = nr_class;
			svm->w = Malloc(double, w_size*nr_w);
			for (int i = 0; i < w_size; i++)
			{
				for (int j = 0; j < nr_w; j++)
				{
					fin >> svm->w[i*nr_w + j];
				}
			}
			break;
		}
		else if (cmd == "label")
		{
			int nr_class = svm->nr_class;
			svm->label = Malloc(int, nr_class);
			for (int i = 0; i < nr_class; i++)
			{
				fin >> svm->label[i];
			}
		
		}
		else
		{
			cout << "unknown text in model file: " << cmd << endl;
			free(svm->label);
			free(svm->w);
			return;
		}
	}
	fin.close();
}


void load_pca(cv::PCA& pca, std::string filename)
{
	ifstream fin(filename.c_str());
	if (!fin)
	{
		std::cout << "Unable to open file: pca.dat " << endl;
		exit(1);
	}
	string cmd;
	while (1)
	{
		fin >> cmd;
		if (cmd == "eigenvectors")
		{
			cv::Mat eigenvectors(numPcaBase, featureDim, CV_64F);
			for (int r = 0; r < eigenvectors.rows; r++)
			{
				for (int c = 0; c < eigenvectors.cols; c++)
				{
					fin >> eigenvectors.at<double>(r, c);
				}
			}
			pca.eigenvectors = eigenvectors;
		}
		else if (cmd == "eigenvalues")
		{
			cv::Mat eigenvalues(numPcaBase, 1, CV_64F);
			for (int r = 0; r < eigenvalues.rows; r++)
			{
				for (int c = 0; c < eigenvalues.cols; c++)
				{
					fin >> eigenvalues.at<double>(r, c);
				}
			}
			pca.eigenvalues = eigenvalues;

		}
		else if (cmd == "mean")
		{
			cv::Mat mean(featureDim, 1, CV_64F);
			for (int r = 0; r< mean.rows; r++)
			{
				for (int c = 0; c < mean.cols; c++)
				{
					fin >> mean.at<double>(r, c);
				}
			}
			pca.mean = mean;
			break;

		}
		else
		{
			cout << "unknown text in pca.dat: " << cmd << endl;
			exit(1);
		}

	}
	fin.close();

}

void load_gmm(VlGMM*& gmm, string filename)
{

	

	ifstream fin(filename.c_str());
	if (!fin)
	{
		std::cout << "Unable to open file: gmm.dat " << endl;
		exit(1);
	}
	string cmd;
	while (1)
	{
		fin >> cmd;
		if (cmd == "gmmPriors")
		{
			gmm = vl_gmm_new(VL_TYPE_DOUBLE, numPcaBase, numGmmClusters);
			vl_gmm_set_initialization(gmm, VlGMMCustom);

			double * gmmPriors = (double *)vl_calloc(numGmmClusters, sizeof(double));
			for (int i = 0; i < numGmmClusters; i++)
			{
				fin >> gmmPriors[i];
			}
			vl_gmm_set_priors(gmm, gmmPriors);

		}
		else if (cmd == "gmmMeans")
		{
			double * gmmMeans = (double *)vl_calloc(numGmmClusters*numPcaBase, sizeof(double));
			for (int i = 0; i < numGmmClusters*numPcaBase; i++)
			{
				fin >> gmmMeans[i];
			}
			vl_gmm_set_means(gmm, gmmMeans);

		}
		else if (cmd == "gmmCovariances")
		{
			double * gmmCovariances = (double *)vl_calloc(numGmmClusters*numPcaBase, sizeof(double));
			for (int i = 0; i < numGmmClusters*numPcaBase; i++)
			{
				fin >> gmmCovariances[i];
			}
			vl_gmm_set_covariances(gmm, gmmCovariances);
			break;

		}
		else
		{
			cout << "unknown text in model file: " << cmd << endl;
			vl_gmm_delete(gmm);
			exit(1);
		}
	}
	fin.close();
}


void gmm_coding(cv::Mat&in, cv::Mat&fv, VlGMM*& gmm)
{
	vl_size dimension = numPcaBase;
	vl_size numClusters = numGmmClusters;
	double * enc = (double *)vl_malloc(sizeof(double)* 2 * dimension*numClusters);

	int localFeatureNum = in.rows;
	int localFeatureDim = in.cols;

	double * data = (double *)vl_malloc(sizeof(double)*localFeatureNum*localFeatureDim);
	for (int i = 0; i < in.cols; i++)
	{
		for (int j = 0; j<in.rows; j++)
		{
			data[i*in.rows + j] = in.at<double>(j, i);
		}
	}
	vl_fisher_encode(enc, VL_TYPE_DOUBLE,
		vl_gmm_get_means(gmm), dimension, numClusters,
		vl_gmm_get_covariances(gmm),
		vl_gmm_get_priors(gmm),
		data, in.rows,
		VL_FISHER_FLAG_NORMALIZED | VL_FISHER_FLAG_SQUARE_ROOT | VL_FISHER_FLAG_IMPROVED);
	fv = cv::Mat(2 * dimension*numClusters, 1, CV_64F);
	for (int i = 0; i<2 * dimension*numClusters; i++)
	{
		fv.at<double>(i) = enc[i];
	}
	vl_free(enc);
	vl_free(data);

}

void code_into_sparse(struct feature_node * &x, cv::Mat &code)
{
	cv::Mat enc = code.t();

	int cnt = 0;
	for (int i = 0; i < enc.rows; ++i)
	{
		if (fabs(enc.at<double>(i)) > 1e-6)
		{
			++cnt;
		}
	}

	x = Malloc(struct feature_node, cnt + 1);

	cnt = 0;
	for (int i = 0; i< enc.rows; ++i)
	{
		if (fabs(enc.at<double>(i)) > 1e-6)
		{
			x[cnt].value = enc.at<double>(i);
			x[cnt].index = i + 1;
			++cnt;
		}
	}

	x[cnt].index = -1;
}

float predictWithScore(struct model *& svm, cv::Mat& code)
{
	assert(!code.empty());
	code = code.t();
	struct feature_node *x;
	code_into_sparse(x, code);
	double label = predictScore(svm, x);
	free(x);
	return label;
}

int predictLabel(struct model *& svm, cv::Mat& code)
{
	assert(!code.empty());
	code = code.t();
	struct feature_node *x;
	code_into_sparse(x, code);
	double label = predict(svm, x);
	free(x);
	return int(label);
}

vector<string> readTXT(string filename)
{
	ifstream in(filename.c_str());
	vector<string>result;
	string line;

	while (getline(in, line))
	{
		result.push_back(line);
	}
	in.close();
	return result;
}

void rotateFace(Mat& src, Point& pt, double angle, Mat& dst)
{
	Mat r = getRotationMatrix2D(pt, angle, 1.0);
	warpAffine(src, dst, r, Size(src.cols, src.rows));
}

Mat cropFacesBasedOnEye(Mat _image, cv::Point leftEye, cv::Point rightEye,
	float offset, int outputWidth, int outputHeight)
{
	int offset_h = floor(offset * outputWidth);
	int offset_v = floor(offset * outputHeight);

	int eyegap_h = rightEye.x - leftEye.x;
	int eyegap_v = rightEye.y - leftEye.y;

	float eye_distance = sqrt(pow(eyegap_h, 2) + pow(eyegap_v, 2));
	float eye_reference = outputWidth - 2 * offset_h;
	float scale = eye_distance / eye_reference;

	//rotate original around the left eye
	cv::Mat rotatedImage;
	if (eyegap_v != 0)
	{
		double rotation = atan2f((float)eyegap_v, (float)eyegap_h);
		double degree = rotation * 180 / CV_PI;
		rotateFace(_image, leftEye, degree, rotatedImage);
	}

	//crop the rotated image
	cv::Point crop_xy(leftEye.x - scale*offset_h, leftEye.y - scale*offset_v);
	cv::Size crop_size(outputWidth*scale, outputHeight*scale);
	cv::Rect crop_area(crop_xy, crop_size);
	cv::Mat cropFace;
	if (eyegap_v == 0)
		cropFace = _image(crop_area);
	else
		cropFace = rotatedImage(crop_area);

	//resize the face
	cv::resize(cropFace, cropFace, cv::Size(outputWidth, outputHeight));
	Mat croppedGray;
	//cv::cvtColor(cropFace, croppedGray, CV_BGR2GRAY);
	//cv::equalizeHist(croppedGray, croppedGray);
	return cropFace;
}



void TestWithCapture()
{
	string svm_model = "svmModel.dat";
	string pca_model = "pca.dat";
	string gmm_model = "gmm.dat";

	svmModel* svm = Malloc(svmModel, 1);
	cv::PCA pca;
	VlGMM* gmm;
	

	load_svm(svm, svm_model);
	load_pca(pca, pca_model);
	load_gmm(gmm, gmm_model);



	std::string  face_cascade_name = "haarcascade_frontalface_alt.xml";
	faceDetector face_cascade(face_cascade_name);

	featureExtractor extractor;

	//VideoWriter writer("text.avi", CV_FOURCC('M', 'J', 'P', 'G'), 25.0, Size(640, 480));

	//VideoCapture capture("C:\\Users\\zsm\\Desktop\\CBSR-Antispoofing\\train_release\\7\\4.avi");
	VideoCapture capture(3);
	
	//capture.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	//capture.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
	double rate = 25.0;
	Size videoSize(640, 480);
	cv::Mat frame;
	vector<Rect> faces;
	int label;
	bool isFirstFrame = true;

	if (capture.isOpened())
	{
		while (true)
		{
			capture >> frame;
			if (isFirstFrame)
			{
				isFirstFrame = false;
				continue;
			}
			if (frame.empty())
				break;

			vector<Rect>faces = face_cascade.detect(frame);
			Point leftEye, rightEye;
			//vector<Rect>faces = detectFaceRect(frame, &leftEye, &rightEye);
			if (faces.size() > 0)
			{
				for (size_t i = 0; i < faces.size(); i++)
				{
					int xo = faces[i].x + faces[i].width / 2;
					int yo = faces[i].y + faces[i].height / 2;
					int L = int(max(faces[i].width, faces[i].height)*1.2);
					cv::Rect rect(max(0, xo - L / 2), max(0, yo - L / 2), min(L, frame.cols - 1 - xo + faces[i].width / 2), min(L, frame.rows - 1 - yo + faces[i].height / 2));
					cv::Mat face = frame(rect);
					cv::resize(face, face, Size(64, 64), 0, 0, INTER_NEAREST);
					cv::Mat  Feature = extractor.findSurfDescriptor(face);
					cv::Mat pcaCode = pca.project(Feature);
					cv::Mat gmmCode;
					gmm_coding(pcaCode, gmmCode, gmm);
					label = predictLabel(svm, gmmCode);
					cv::rectangle(frame, faces[i], cv::Scalar(0, 255, 0), 2);
					if (label == 1)
					{
						cv::putText(frame, "Geninue", cv::Point(faces[i].x, faces[i].y), CV_FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
					}
					else
					{
						cv::putText(frame, "Fake", cv::Point(faces[i].x, faces[i].y), CV_FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
					}


				}
			}
			cv::imshow("Video", frame);
			//writer << frame;
			if (waitKey(1) == 27)
			{
				break;
			}
		}
	}
	else
	{
		vl_gmm_delete(gmm);
		free_and_destroy_model(&svm);
		cout << "--��!�� No capture id : 0 " << endl;
		system("pause");
		exit(1);
	}
	vl_gmm_delete(gmm);
	free_and_destroy_model(&svm);
}


void TestWithImages()
{
	//string svm_model = ".\\surf11_pca300_balance\\svm_0.dat";
	string svm_model = "mySvmModel_fake0_3_real0_3.dat";
	//string svm_model = "mySvmModel_fake0_2_real0_2.dat";
	string pca_model = "pca.dat";
	string gmm_model = "gmm.dat";

	svmModel* svm = Malloc(svmModel, 1);
	cv::PCA pca;
	VlGMM* gmm;


	load_svm(svm, svm_model);
	load_pca(pca, pca_model);
	load_gmm(gmm, gmm_model);


	

	//std::string  face_cascade_name = "haarcascade_frontalface_alt.xml";
	//faceDetector face_cascade(face_cascade_name);
	featureExtractor extractor;

	//VideoWriter writer("text.avi", CV_FOURCC('M', 'J', 'P', 'G'), 25.0, Size(640, 480));


	//VideoCapture capture(2);




	cv::Mat frame;
	vector<Rect> faces;
	int label;
	//bool isFirstFrame = true;

	int frameCount = 0, geninueError = 0, fakeError = 0,withoutFaceCount=0,correctCount=0;

	string realDir = "C:\\Users\\zsm\\Desktop\\given_image\\TestData\\real\\";
	string fakeDir = "C:\\Users\\zsm\\Desktop\\given_image\\TestData\\fake\\";

	vector<string>realImageNames = readTXT(realDir+"fileName.txt");
	vector<string>fakeImageNames= readTXT(fakeDir + "fileName.txt");

	for (int i = 0; i < realImageNames.size(); i++)
	{
		frame = imread(realImageNames[i]);
		//cv::resize(frame, frame, Size(frame.rows*0.8, frame.cols*0.8), 0, 0, INTER_NEAREST);

		if (frame.empty())
		{
			cout << "��ȡʧ�ܣ�";
			continue;
		}
		frameCount++;
		//std::cout << "frame ID:" << frameCount << "  ";
		//faces = face_cascade.detect(frame);

		Point leftEye, rightEye;

		//faces = detectFaceRect(frame, &leftEye, &rightEye);


		if (faces.size() > 0)
		{
			//for (size_t i = 0; i < faces.size(); i++)
			//{
			int i = 0;
			int xo = faces[i].x + faces[i].width / 2;
			int yo = faces[i].y + faces[i].height / 2;
			int L = int(max(faces[i].width, faces[i].height)*1.2);
			cv::Rect rect(max(0, xo - L / 2), max(0, yo - L / 2), min(L, frame.cols - 1 - xo + faces[i].width / 2), min(L, frame.rows - 1 - yo + faces[i].height / 2));


			//cv::Mat face = frame(rect);
			//cv::resize(face, face, Size(64, 64), 0, 0, INTER_NEAREST);
			Mat frontal = frame.clone();
			Mat face = cropFacesBasedOnEye(frontal, leftEye, rightEye, 0.2, 64, 64);


			cv::Mat  Feature = extractor.findSurfDescriptor(face);


			cv::Mat pcaCode = pca.project(Feature);



			cv::Mat gmmCode;
			gmm_coding(pcaCode, gmmCode, gmm);

			//namedWindow("feature");
			//imshow("feature", gmmCode);

			label = predictLabel(svm, gmmCode);
			//cv::rectangle(frame, faces[i], cv::Scalar(0, 255, 0), 2);
			if (label == 1)
			{
				correctCount++;
				//cv::putText(frame, "Geninue", cv::Point(faces[i].x, faces[i].y), CV_FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
			}
			else
			{
				geninueError++;
				//cv::putText(frame, "Fake", cv::Point(faces[i].x, faces[i].y), CV_FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
			}


			//	}

		}
		else
		{
			withoutFaceCount++;
		}
	}

	
	for (int i = 0; i < fakeImageNames.size(); i++)
	{
		frame = imread(fakeImageNames[i]);
		//cv::resize(frame, frame, Size(frame.rows*0.8, frame.cols*0.8), 0, 0, INTER_NEAREST);

		if (frame.empty())
		{
			cout << "��ȡʧ�ܣ�";
			continue;
		}
		frameCount++;
		//std::cout << "frame ID:" << frameCount << "  ";
		//faces = face_cascade.detect(frame);

		Point leftEye, rightEye;

		//faces = detectFaceRect(frame, &leftEye, &rightEye);


		if (faces.size() > 0)
		{
			//for (size_t i = 0; i < faces.size(); i++)
			//{
			int i = 0;
			int xo = faces[i].x + faces[i].width / 2;
			int yo = faces[i].y + faces[i].height / 2;
			int L = int(max(faces[i].width, faces[i].height)*1.2);
			cv::Rect rect(max(0, xo - L / 2), max(0, yo - L / 2), min(L, frame.cols - 1 - xo + faces[i].width / 2), min(L, frame.rows - 1 - yo + faces[i].height / 2));


			//cv::Mat face = frame(rect);
			//cv::resize(face, face, Size(64, 64), 0, 0, INTER_NEAREST);
			Mat frontal = frame.clone();
			Mat face = cropFacesBasedOnEye(frontal, leftEye, rightEye, 0.2, 64, 64);


			cv::Mat  Feature = extractor.findSurfDescriptor(face);


			cv::Mat pcaCode = pca.project(Feature);



			cv::Mat gmmCode;
			gmm_coding(pcaCode, gmmCode, gmm);

			//namedWindow("feature");
			//imshow("feature", gmmCode);

			label = predictLabel(svm, gmmCode);
			//cv::rectangle(frame, faces[i], cv::Scalar(0, 255, 0), 2);
			if (label == 1)
			{
				fakeError++;
				//cv::putText(frame, "Geninue", cv::Point(faces[i].x, faces[i].y), CV_FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
			}
			else
			{
				correctCount++;
				//cv::putText(frame, "Fake", cv::Point(faces[i].x, faces[i].y), CV_FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
			}


			//	}

		}
		else
		{
			withoutFaceCount++;
		}
	}
	//cv::imshow("Video", frame);
	//waitKey(0);
	frame.release();
	frame = NULL;

	cout << "ͼƬ����:" << frameCount << endl;
	cout << "����ͼ���������" << geninueError << endl;
	cout << "����ͼ���������" << fakeError << endl;


	vl_gmm_delete(gmm);
	free_and_destroy_model(&svm);
}
void Test()
{
	//string svm_model = ".\\surf11_pca300_balance\\svm_0.dat";
	string svm_model = "svmModel.dat";
	string pca_model = "pca.dat";
	string gmm_model = "gmm.dat";

	svmModel* svm = Malloc(svmModel, 1);
	cv::PCA pca;
	VlGMM* gmm;


	load_svm(svm, svm_model);
	load_pca(pca, pca_model);
	load_gmm(gmm, gmm_model);



	//std::string  face_cascade_name = "haarcascade_frontalface_alt.xml";
	//faceDetector face_cascade(face_cascade_name);
	featureExtractor extractor;

	//VideoWriter writer("text.avi", CV_FOURCC('M', 'J', 'P', 'G'), 25.0, Size(640, 480));


	//VideoCapture capture(2);




	cv::Mat frame;
	vector<Rect> faces;
	int label;
	//bool isFirstFrame = true;

	int frameCount = 0, geninueCount = 0, fakeCount = 0;



	frame = imread("C:\\Users\\zsm\\Desktop\\given_image\\TestData\\real\\3134.jpg");


	//cv::resize(frame, frame, Size(frame.rows*0.8, frame.cols*0.8), 0, 0, INTER_NEAREST);

	if (frame.empty())
	{
		cout << "��ȡʧ�ܣ�";
		return;
	}

	frameCount++;

	//std::cout << "frame ID:" << frameCount << "  ";
	//faces = face_cascade.detect(frame);

	Point leftEye, rightEye;

	//faces = detectFaceRect(frame,&leftEye,&rightEye);


	if (faces.size() > 0)
	{
		//for (size_t i = 0; i < faces.size(); i++)
		//{
		int i = 0;
		int xo = faces[i].x + faces[i].width / 2;
		int yo = faces[i].y + faces[i].height / 2;
		int L = int(max(faces[i].width, faces[i].height)*1.2);
		cv::Rect rect(max(0, xo - L / 2), max(0, yo - L / 2), min(L, frame.cols - 1 - xo + faces[i].width / 2), min(L, frame.rows - 1 - yo + faces[i].height / 2));

		
		//cv::Mat face = frame(rect);
		//cv::resize(face, face, Size(64, 64), 0, 0, INTER_NEAREST);
		Mat frontal = frame.clone();
		Mat face = cropFacesBasedOnEye(frontal,leftEye,rightEye,0.2,64,64);
		

		cv::Mat  Feature = extractor.findSurfDescriptor(face);


		cv::Mat pcaCode = pca.project(Feature);



		cv::Mat gmmCode;
		gmm_coding(pcaCode, gmmCode, gmm);

		namedWindow("feature");
		imshow("feature", gmmCode);

		label = predictLabel(svm, gmmCode);
		cv::rectangle(frame, faces[i], cv::Scalar(0, 255, 0), 2);
		if (label == 1)
		{
			geninueCount++;
			cv::putText(frame, "Geninue", cv::Point(faces[i].x, faces[i].y), CV_FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
		}
		else
		{
			fakeCount++;
			cv::putText(frame, "Fake", cv::Point(faces[i].x, faces[i].y), CV_FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
		}


		//	}

	}

	cv::imshow("Video", frame);
	waitKey(0);
	frame.release();
	frame = NULL;


	vl_gmm_delete(gmm);
	free_and_destroy_model(&svm);
}

unsigned char* getImageData(string filePath,int& width,int& heigh)
{
	Mat test = imread(filePath);
	int size = test.rows*test.cols * 3;
	int nl= test.cols * 3;

	width = test.cols;
	heigh = test.rows;

	unsigned char*result = new unsigned char[size];

	for (int i = 0; i < test.rows; i++)
	{
		uchar*ptr = test.ptr<uchar>(i);
		for (int j = 0; j < nl; j++)
		{
			result[i*nl + j] = ptr[j];
		}
	}

	return result;
}

float antiSpoofDetection(unsigned char*imgData,int width,int heigh)
{
	//string svm_model = ".\\surf11_pca300_balance\\svm_0.dat";
	string svm_model = "svmModel.dat";
	string pca_model = "pca.dat";
	string gmm_model = "gmm.dat";

	svmModel* svm = Malloc(svmModel, 1);
	cv::PCA pca;
	VlGMM* gmm;


	load_svm(svm, svm_model);
	load_pca(pca, pca_model);
	load_gmm(gmm, gmm_model);



	std::string  face_cascade_name = "haarcascade_frontalface_alt2.xml";
	faceDetector face_cascade(face_cascade_name);

	featureExtractor extractor;

	//VideoWriter writer("text.avi", CV_FOURCC('M', 'J', 'P', 'G'), 25.0, Size(640, 480));


	//VideoCapture capture(2);


	//Mat test = imread("C:\\Users\\zsm\\Desktop\\given_image\\TestData\\real\\3135.jpg");

	//width = test.cols;
	//heigh = test.rows;

	Mat frame = Mat(heigh,width,CV_8UC3);

	int nl = heigh;
	int nc = width * 3;

	for (int i = 0; i < nl; i++)
	{
		unsigned char*data = frame.ptr<uchar>(i);
		for (int j = 0; j < nc; j++)
		{
			data[j] = imgData[nc*i+j];
		}
	}

	//imshow("result",frame);
	//waitKey(0);
	//return 0;


	vector<Rect> faces;
	float result=-1;
	//bool isFirstFrame = true;

	//cv::resize(frame, frame, Size(frame.rows*0.8, frame.cols*0.8), 0, 0, INTER_NEAREST);

	if (frame.empty())
	{
		//cout << "��ȡʧ�ܣ�";
		return -1;
	}

	

	//std::cout << "frame ID:" << frameCount << "  ";
	faces = face_cascade.detect(frame);

	//Point leftEye, rightEye;

	//faces = detectFaceRect(frame, &leftEye, &rightEye);


	if (faces.size() > 0)
	{
		//for (size_t i = 0; i < faces.size(); i++)
		//{
		int i = 0;
		int xo = faces[i].x + faces[i].width / 2;
		int yo = faces[i].y + faces[i].height / 2;
		int L = int(max(faces[i].width, faces[i].height)*1.2);
		cv::Rect rect(max(0, xo - L / 2), max(0, yo - L / 2), min(L, frame.cols - 1 - xo + faces[i].width / 2), min(L, frame.rows - 1 - yo + faces[i].height / 2));


		cv::Mat face = frame(rect);
		cv::resize(face, face, Size(64, 64), 0, 0, INTER_NEAREST);
		//Mat frontal = frame.clone();
		//Mat face = cropFacesBasedOnEye(frontal, leftEye, rightEye, 0.2, 64, 64);



		cv::Mat  Feature = extractor.findSurfDescriptor(face);
		cv::Mat pcaCode = pca.project(Feature);
		cv::Mat gmmCode;
		gmm_coding(pcaCode, gmmCode, gmm);

		//namedWindow("feature");
		//imshow("feature", gmmCode);

		result = predictWithScore(svm, gmmCode);

	}

	//cv::imshow("Video", frame);
	//waitKey(0);
	frame.release();
	frame = NULL;


	vl_gmm_delete(gmm);
	free_and_destroy_model(&svm);

	return result;
}



int main()
{
	
	int width = 0, heigh = 0;
	//getImageData用于读取图片数据，转换为unsigned char流
	string imagePath="/home/zsm/data/TestData/fake/WIN_20180424_09_49_29_Pro.jpg";
	uchar*imgData = getImageData(imagePath,width,heigh);

	//读入图片的数据流，进行反欺骗检测
	float result = antiSpoofDetection(imgData,width,heigh);
	cout << result << endl;
	delete(imgData);
	
	//Train();
	//Test();
	//TestWithImages();
	//TestWithCapture();

	return 0;
}