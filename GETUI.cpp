// test_x64.cpp : 定义控制台应用程序的入口点。
#define Message B_Message
#include "IGtPush.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#ifdef WIN32
#include <windows.h>
#endif

#include<GETUI.h>

using namespace std;

void LinkTemplateDemo(LinkTemplate* templ)
{
	templ->t.appId = appId;
	templ->t.appKey = appKey;
	//通知栏标题
	templ->title = "请注意现在家里有人进来了";
	//通知栏内容
	templ->text = "快点开看看是谁";
	//通知栏显示本地图片
	templ->logo = "";
	//通知栏显示网络图标，如无法读取，则显示本地默认图标，可为空
	templ->logoUrl = "";

	static int num = 224;
	char image_name[100];
	sprintf(image_name, "http://101.201.211.87:8080/tu/%d.jpg", num++);


	//打开的链接地址
	//templ->url = "http://101.201.211.87:8080/xzabg/224.jpg";
	templ->url = image_name;

	//templ->t.duration_start="2015-07-10 18:00:00";
	//templ->t.duration_end="2015-07-10 19:00:00";
	//接收到消息是否响铃，GT_ON：响铃 GT_OFF：不响铃
	templ->isRing = GT_ON;
	//接收到消息是否震动，GT_ON：震动 GT_OFF：不震动
	templ->isVibrate = GT_ON;
	//接收到消息是否可清除，GT_ON：可清除 GT_OFF：不可清除
	templ->isClearable = GT_ON;
}


void tosingletest() {

	//准备数据
	Message msg = { 0 };
	msg.isOffline = 0;//是否离线下发
	msg.offlineExpireTime = 1000 * 3600 * 2;//离线下发有效期 毫秒
	msg.pushNetWorkType = 0;//0不限 1wifi 2:4G/3G/2G
	SingleMessage singleMsg = { 0 };
	singleMsg.msg = msg;

	//目标用户
	Target target = { 0 };
	target.appId = appId;
	target.clientId = cid;
	//target.alias = "test";
	IPushResult result = { 0 };


	//TransmissionTemplate tmpl = { 0 };
	//TransmissionTemplateDemo(&tmpl);
	//result = pushMessageToSingle(appKey, &singleMsg, &tmpl, Transmission, &target);

	LinkTemplate tmpl = { 0 };
	LinkTemplateDemo(&tmpl);
	result = pushMessageToSingle(appKey, &singleMsg, &tmpl, Link, &target);

	printResult(result);
}

static void printResult(IPushResult &result) {
	cout << "print result:-------------" << endl;
	for (int i = 0; i < result.size; i++) {
		cout << result.entry[i].key << ": " << result.entry[i].value << endl;
	}
	cout << "print end:----------------" << endl;
}