// stdafx.h : 标准系统包含文件的包含文件，
// 或是经常使用但不常更改的
// 特定于项目的包含文件
//

#pragma once

#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0501
#endif

#include "targetver.h"

#include <stdio.h>
#include <tchar.h>
#include <iostream>

#include <WinSock2.h>
#pragma comment(lib,"ws2_32.lib")

#include "NetDefine.h"

#define SERVER_PORT		20000
#define SERVER_IP		"101.201.211.87"//服务器
//#define SERVER_IP		"192.168.1.114"
//#define SERVER_IP		"192.168.0.102"//169.254.169.148"//我的电脑不插网线
//#define SERVER_IP		"127.0.0.1"//我的电脑有网线
//#define SERVER_IP		"192.168.1.131"//我的台式机


class WinSocketSystem
{
public:
	WinSocketSystem()
	{
		int iResult = WSAStartup(MAKEWORD(2, 2), &wsaData);
		if (iResult != NO_ERROR)
		{
			exit(-1);
		}
	}
	~WinSocketSystem()
	{
		WSACleanup();
	}
protected:
	WSADATA wsaData;
};

static WinSocketSystem g_winsocketsystem;
// TODO:  在此处引用程序需要的其他头文件