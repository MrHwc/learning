#include <opencv2\opencv.hpp>
#include <fstream>
using namespace std;
using namespace cv;

//void main()//�궨
//{ 
//	
//	Size board_size = Size(9, 6);            /****    �������ÿ�С��еĽǵ���       ****/
//	vector<Point2f> corners;                  /****    ����ÿ��ͼ���ϼ�⵽�Ľǵ�       ****/
//	vector<vector<Point2f>>  corners_Seq;    /****  �����⵽�����нǵ�       ****/
//	vector<Mat>  image_Seq;
//	int count = 0;
//	int img_count = 0;
//	string imgname;
//	ifstream fin("C:/Users/Administrator/Desktop/cam_pic/pic.txt");
//	Mat src;
//	while (getline(fin, imgname))
//	{
//		img_count++;
//		imgname = "C:/Users/Administrator/Desktop/cam_pic/" + imgname;
//		src = imread(imgname);
//
//		Mat img_gray;
//		cvtColor(src, img_gray, CV_BGR2GRAY);
//		bool patternfound = findChessboardCorners(src, board_size, corners, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE +
//			CALIB_CB_FAST_CHECK);
//		if (!patternfound)
//		{
//			cout << "not found" << endl;
//			break;
//		}
//		else
//		{
//			cornerSubPix(img_gray, corners, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.01));
//			Mat img_temp = src.clone();
//			for (int i = 0; i < corners.size(); ++i)
//			{
//				circle(img_temp, corners[i], 2, Scalar(0, 0, 255), 2, 8, 0);
//			}
//			/*imshow("img_temp", img_temp);
//			waitKey(0);*/
//			count += corners.size();
//			corners_Seq.push_back(corners);
//		}
//	}
//
//	Size square_size = Size(26, 26);//
//	vector<vector<Point3f>>  object_Points;
//	//Mat img_points = Mat(1, count, CV_32FC2, Scalar::all(0));////
//	//vector<int>  point_counts;/////
//	Mat intrinsic_matrix = Mat(3, 3, CV_32FC1, Scalar::all(0));
//	Mat distortion_coeffs = Mat(1, 5, CV_32FC1, Scalar::all(0));
//	vector<cv::Mat> rotation_vectors;
//	vector<cv::Mat> translation_vectors;
//	
//	for (int n = 0; n < img_count; ++n)
//	{
//		vector<Point3f> tempPointSet_1;
//		for (int i = 0; i < board_size.height; i++)
//		{
//			for (int j = 0; j < board_size.width; j++)
//			{
//				Point3f tempPoint;
//				tempPoint.x = i*square_size.width;
//				tempPoint.y = j*square_size.height;
//				tempPoint.z = 0;
//				tempPointSet_1.push_back(tempPoint);
//			}
//		}
//		object_Points.push_back(tempPointSet_1);
//	}
//	calibrateCamera(object_Points, corners_Seq, src.size(), intrinsic_matrix, distortion_coeffs, rotation_vectors, translation_vectors, 0);
//	
//
//	double total_err = 0.0;                   /* ����ͼ���ƽ�������ܺ� */
//	double err = 0.0;                        /* ÿ��ͼ���ƽ����� */
//	vector<Point2f>  image_points2;
//	for (int n = 0; n < img_count; ++n)
//	{
//		vector<Point3f> tempPointSet = object_Points[n];
//		/****    ͨ���õ������������������Կռ����ά���������ͶӰ���㣬�õ��µ�ͶӰ��     ****/
//		projectPoints(tempPointSet, rotation_vectors[n], translation_vectors[n], intrinsic_matrix, distortion_coeffs, image_points2);
//		/* �����µ�ͶӰ��;ɵ�ͶӰ��֮������*/
//		vector<Point2f> tempImagePoint = corners_Seq[n];
//		Mat tempImagePointMat = Mat(1, tempImagePoint.size(), CV_32FC2);
//		Mat image_points2Mat = Mat(1, image_points2.size(), CV_32FC2);
//		for (size_t i = 0; i != tempImagePoint.size(); i++)
//		{
//			image_points2Mat.at<Vec2f>(0, i) = Vec2f(image_points2[i].x, image_points2[i].y);
//			tempImagePointMat.at<Vec2f>(0, i) = Vec2f(tempImagePoint[i].x, tempImagePoint[i].y);
//		}
//		err = norm(image_points2Mat, tempImagePointMat, NORM_L2);
//		total_err += err /= 54;//point_counts[0]
//		cout << "��" << n + 1 << "��ͼ���ƽ����" << err << "����" << endl;
//	}
//	cout << "����ƽ����" << total_err /img_count<< "����" << endl << endl;
//	FileStorage fs("calibration.xml", FileStorage::WRITE);
//	fs << "intrinsic_matrix" << intrinsic_matrix;
//	fs << "distortion_coeffs" << distortion_coeffs;
//	fs.release();
//	waitKey(0);
//}

//void main()//У��
//{
//	FileStorage fs;
//	fs.open("calibration.xml", FileStorage::READ);
//	Mat intrinsic_matrix = Mat(3, 3, CV_32FC1, Scalar::all(0));
//	Mat distortion_coeffs = Mat(1, 5, CV_32FC1, Scalar::all(0));
//	fs["intrinsic_matrix"] >> intrinsic_matrix;
//	fs["distortion_coeffs"] >> distortion_coeffs;
//	Mat src = imread("C:/Users/Administrator/Desktop/cam_pic/6.jpg");
//	imshow("src", src);
//
//	Mat mapx = Mat(src.size(), CV_32FC1);
//	Mat mapy = Mat(src.size(), CV_32FC1);
//	Mat R = Mat::eye(3, 3, CV_32F);
//
//	initUndistortRectifyMap(intrinsic_matrix, distortion_coeffs, R, getOptimalNewCameraMatrix(intrinsic_matrix, distortion_coeffs, src.size(), 1, src.size(), 0), src.size(), CV_32FC1, mapx, mapy);
//	Mat dst = src.clone();
//	remap(src, dst, mapx, mapy, INTER_LINEAR);
//	imshow("dst",dst);
//	waitKey(0);
//}

void main()
{
	Mat src = imread("x.jpg");
	Mat gray;
	cvtColor(src,gray,CV_BGR2GRAY);
	MatND dsthist;
	int dims = 1;
	float hranges[] = { 0, 255 };
	const float *ranges[] = {hranges};
	int size = 256;
	int channels = 0;

	calcHist(&gray,1,&channels,Mat(),dsthist,dims,&size,ranges);
	int scale = 1;
	Mat dstimage(size*scale, size, CV_8U, Scalar(0));

	double minValue = 0;
	double maxValue = 0;
	minMaxLoc(dsthist, &minValue, &maxValue, 0, 0);

	int hpt = saturate_cast<int>(0.9*size);
	for (int i = 0; i < 256; ++i)
	{
		float binValue = dsthist.at<float>(i);
		int realValue = saturate_cast<int>(binValue*hpt / maxValue);
		rectangle(dstimage,Point(i*scale,size-1),Point((i+1)*scale-1,size-realValue),Scalar(255));
	}
	imshow("test",dstimage);
	cout << dstimage.size() << endl;
	waitKey(0);
}