// ObjectDetect.cpp : 定义控制台应用程序的入口点。
//
//#include "stdafx.h"
#include "ObjDetTrack.h"
#include <tchar.h>

char filename[100];
char windowname[100];
char image_name[25];

int _tmain(int argc, _TCHAR* argv[])
{
	//测试多张图片
	/*	for (int i = 1; i <= 50; i++)
	{
	sprintf(filename, "E:\\video\\pd\\%d.jpg", i);
	sprintf(windowname, "window%d.jpg", i);
	Mat pScr = imread(filename, 1);
	cvNamedWindow(windowname, CV_WINDOW_AUTOSIZE);
	imshow(windowname, pScr);

	vector<ObjResult> m_objects;
	ObjDet m_objDet;
	int flag = m_objDet.Initial();
	m_objects.clear();
	flag = m_objDet.ProcessSingleImg(pScr, m_objects);

	IplImage qImg;
	Mat imgDetectResult;
	qImg = IplImage(imgDetectResult); // cv::Mat -> IplImage
	sprintf(image_name, "%s%d%s", "..\\jieguo\\image", ++i, ".jpg");
	cvSaveImage(image_name, &qImg);
	}*/

	//测试单张图片
//	Mat firstImg = imread("E:\\video\\pd\\12.jpg");
//	imshow("yt", firstImg);
	vector<ObjResult> m_objects;

	ObjDet m_objDet;
	int flag = m_objDet.Initial();
	m_objects.clear();

	//VideoCapture camera(1);
	VideoCapture camera(0);
	Mat cameraFrame;
	if (!camera.isOpened())
	{
		cout << "ERROR:Could not access the camera or video!" << endl;
		exit(1);
	}
	int CAMERA_CHECK_ITERATIONS = 10;
	while (true)
	{
		camera >> cameraFrame;
		if (!cameraFrame.empty())
		{
#ifdef SHOW_DEBUG_INFO
			time_t startcal = clock();
#endif
			flag = m_objDet.ProcessSingleImg(cameraFrame, m_objects);

#ifdef SHOW_DEBUG_INFO
			time_t endcal = clock();
			cout << "计算一张图片所花费的时间：" << endcal - startcal << endl;
#endif
		}
		else
		{
			cout << "::: Accessing camera :::" << endl;
			if (CAMERA_CHECK_ITERATIONS > 0)
				CAMERA_CHECK_ITERATIONS--;
			else
				break;
		}
		int c = waitKey(33);

		if (c == 27)
		{
			break;
		}
	}

	getchar();
	//cvWaitKey(0);
	return 0;
}