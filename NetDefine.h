#pragma once
#define NET_PACKET_DATA_SIZE 100000
#define NET_PACKET_SIZE (sizeof(NetPacketHeader)+NET_PACKET_DATA_SIZE)+5000
//网络数据包包头
struct NetPacketHeader
{
	unsigned int		wDataSize;		//数据包大小，包含封包头和封包数据大小
	unsigned short		wOpcode;		//操作码
};
/// 网络操作码
enum eNetOpcode
{
	NET_TEST1 = 1,
};
/// 测试1的网络数据包定义
struct NetPacket_Test1
{
	char	arrMessage[NET_PACKET_DATA_SIZE];
};