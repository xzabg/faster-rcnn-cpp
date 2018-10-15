#pragma once
#include <stdafx.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

class TCPClient
{
public:
	TCPClient();
	virtual ~TCPClient();
public:
	//主循环
	void run(char* imagename);
	//发送数据
	bool SendData(unsigned short nOpcode, const char* pDataBuffer, const unsigned int& nDataSize);
	
private:
	SOCKET		mServerSocket;
	sockaddr_in	mServerAddr;

	char		m_cbSendBuf[NET_PACKET_SIZE];
};