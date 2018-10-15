#include "ObjDetTrack.h"
#include "TCPClient.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define Message B_Message
#include "IGtPush.h"
#include"GETUI.h"

void nms_use_gpu(int m_per_nms_topN, ObjResult* iter, float nms_overlap_thres, int *output);

bool compare1(const ObjResult& Info1, const ObjResult& Info2)
{
	return Info1.fg_prob > Info2.fg_prob;
}

ObjDet::ObjDet()
{
	m_data_buf = NULL;
	m_proposal_net_ = NULL;
	m_classify_net_ = NULL;

	m_blob_bbox = NULL;
	m_blob_prob = NULL;
}

ObjDet::~ObjDet()
{
}

/*功能说明：
1、完成检测类本身的必要初始化
*/
int ObjDet::Initial()
{
	//相关参数
	m_per_nms_topN = 6000;
	//m_per_nms_topN = 3000;

	m_after_nms_topN = 300;
	m_use_gpu = true;
	m_min_input_side = 600.0f;

	//m_nms_overlap_thres = 0.75;
	m_nms_overlap_thres = 0.7;
	//m_CONF_THRESH = 0.3;
	m_CONF_THRESH = 0.6;
	m_NMS_THRESH = 0.35;
	//m_prop_min_width = 16;
	m_prop_min_width = 15;
	m_prop_max_width = 270;
	m_prop_max_height = 240;

	//设置硬件模式
	//Caffe::set_mode(Caffe::CPU);
	Caffe::set_mode(Caffe::GPU);

	//设定基准锚点
	m_ori_anchors_cnt = 9;
	m_ori_anchors_vec.clear();

	//这些值直接来自训练结果
	m_ori_anchors_vec.push_back(DhpRect(-83, -39, 100, 56));
	m_ori_anchors_vec.push_back(DhpRect(-175, -87, 192, 104));
	m_ori_anchors_vec.push_back(DhpRect(-359, -183, 376, 200));

	m_ori_anchors_vec.push_back(DhpRect(-55, -55, 72, 72));
	m_ori_anchors_vec.push_back(DhpRect(-119, -119, 136, 136));
	m_ori_anchors_vec.push_back(DhpRect(-247, -247, 264, 264));

	m_ori_anchors_vec.push_back(DhpRect(-35, -79, 52, 96));
	m_ori_anchors_vec.push_back(DhpRect(-79, -167, 96, 184));
	m_ori_anchors_vec.push_back(DhpRect(-167, -343, 184, 360));

	m_anchor_stride = 16;

	//载入训练参数
	int flag = Load();

	return OBJDETTRACK_SUCESS;
}

/*功能说明：
1、完成模型参数的载入
*/
int ObjDet::Load()
{
	string modelPath = "D:\\CaffeProgram\\6FT\\model\\faster_rcnn_VOC0712_ZF\\";

	string proposal_model_file = modelPath + "proposal_test.prototxt";
	string proposal_weights_file = modelPath + "proposal_final.final";

	string classify_model_file = modelPath + "detection_test.prototxt";
	string classify_weights_file = modelPath + "detection_final.final";

	m_proposal_net_ = boost::shared_ptr<Net<float> >(new Net<float>(proposal_model_file, caffe::TEST));
	m_proposal_net_->CopyTrainedLayersFrom(proposal_weights_file);

	m_classify_net_ = boost::shared_ptr<Net<float> >(new Net<float>(classify_model_file, caffe::TEST));
	m_classify_net_->CopyTrainedLayersFrom(classify_weights_file);

	return OBJDETTRACK_SUCESS;
}

//为后续步骤准备好输入数据
int ObjDet::PrepInputData()
{
	m_oriWidth = m_oriImg.cols;
	m_oriHeight = m_oriImg.rows;
	m_oriImg32f = cv::Mat(m_oriHeight, m_oriWidth, CV_32FC3, cv::Scalar(0, 0, 0));

	float min_side = min(m_oriWidth, m_oriHeight);
	m_img_scale = m_min_input_side / min_side;

	//能保证最短边是600？？
	m_resizedHeight = int(m_oriHeight * m_img_scale);
	m_resizedWidth = int(m_oriWidth * m_img_scale);

	//初始图像减去平均值
	for (int h = 0; h < m_oriHeight; ++h)
	{
		for (int w = 0; w < m_oriWidth; ++w)
		{
			int index = 3 * (h * m_oriWidth + w);
			((float*)m_oriImg32f.data)[index + 0] = float(m_oriImg.data[index + 0]) - 125.0f;
			((float*)m_oriImg32f.data)[index + 1] = float(m_oriImg.data[index + 1]) - 125.0f;
			((float*)m_oriImg32f.data)[index + 2] = float(m_oriImg.data[index + 2]) - 125.0f;
		}
	}
	//缩放
	m_resizedImg32f = cv::Mat(m_resizedHeight, m_resizedWidth, CV_32FC3, cv::Scalar(0, 0, 0));
	cv::resize(m_oriImg32f, m_resizedImg32f, cv::Size(m_resizedWidth, m_resizedHeight));

	//格外注意数据的存储顺序，是分层存储的
	//色彩通道的顺序是bgr
	m_data_buf = new float[m_resizedHeight * m_resizedWidth * 3];
	for (int h = 0; h < m_resizedHeight; ++h)
	{
		for (int w = 0; w < m_resizedWidth; ++w)
		{
			m_data_buf[(0 * m_resizedHeight + h)* m_resizedWidth + w] = ((float*)m_resizedImg32f.data)[3 * (h * m_resizedWidth + w) + 0];
			m_data_buf[(1 * m_resizedHeight + h)* m_resizedWidth + w] = ((float*)m_resizedImg32f.data)[3 * (h * m_resizedWidth + w) + 1];
			m_data_buf[(2 * m_resizedHeight + h)* m_resizedWidth + w] = ((float*)m_resizedImg32f.data)[3 * (h * m_resizedWidth + w) + 2];
		}
	}
	return OBJDETTRACK_SUCESS;
}

int ObjDet::ClearInputData()
{
	if (m_data_buf)
	{
		delete[] m_data_buf;
	}
	m_data_buf = NULL;

	return OBJDETTRACK_SUCESS;
}

//启动proposalnet部分的运行，让网络驱动起来
int ObjDet::ProposalNetRun()
{
	//设置数据
	m_proposal_net_->blob_by_name("data")->Reshape(1, 3, m_resizedHeight, m_resizedWidth);
	m_proposal_net_->blob_by_name("data")->set_cpu_data(m_data_buf);

#ifdef SHOW_DEBUG_INFO
	time_t start1 = clock();
#endif

	m_proposal_net_->ForwardFrom(0);

#ifdef SHOW_DEBUG_INFO
	time_t end1 = clock();
	cout << "ForwardFrom time: " << end1 - start1 << endl;
#endif

	m_blob_bbox = m_proposal_net_->blob_by_name("proposal_bbox_pred");
	const float* bbox_delt = m_blob_bbox->cpu_data();

	m_blob_prob = m_proposal_net_->blob_by_name("proposal_cls_prob");

	return OBJDETTRACK_SUCESS;
}

int ObjDet::ProposalExtract()
{
	m_objVec.clear();

	m_propWidth = m_blob_bbox->width();
	m_propHeight = m_blob_bbox->height();
	//注意，通道数需要除以4
	m_propChannel = m_blob_bbox->channels() / 4;

	//bbox_delt原来的扫描顺序——51、39、36、1
	//为了保持跟matlab的代码一致，便于调试，这里也改成先扫描列方向
	for (int w = 0; w < m_propWidth; w++)
	{
		for (int h = 0; h < m_propHeight; h++)
		{
			for (int n = 0; n < m_propChannel; n++)
			{
				//中心位置的偏移量
				float x = m_blob_bbox->data_at(0, 4 * n + 0, h, w);
				float y = m_blob_bbox->data_at(0, 4 * n + 1, h, w);

				//宽高的回归量
				float width = m_blob_bbox->data_at(0, 4 * n + 2, h, w);
				float height = m_blob_bbox->data_at(0, 4 * n + 3, h, w);

				//中心位置的偏移量
				ObjResult temp;
				temp.bbox.size = cv::Size2f(width, height);
				temp.bbox.center = cv::Point2f(x, y);

				//前景和背景的概率
				//重点注意——h那个维度比较诡异，等于将原来的9幅图像进行了高度方向的拼接
				temp.bg_prob = m_blob_prob->data_at(0, 0, n * m_propHeight + h, w);
				temp.fg_prob = m_blob_prob->data_at(0, 1, n * m_propHeight + h, w);

				m_objVec.push_back(temp);
			}
		}
	}
	return OBJDETTRACK_SUCESS;
}

//重点注意——输出的结果，其实是按照图像的列扫描的
int ObjDet::CalcPropsalAnchors()
{
	m_prop_anchors_vec.clear();
	m_prop_anchors_cnt = m_propWidth * m_propHeight * m_propChannel;

	for (int w = 0; w < m_propWidth; w++)
	{
		//列扫描
		for (int h = 0; h < m_propHeight; h++)
		{
			for (int n = 0; n < m_propChannel; n++)
			{
				//计算共享层中的坐标
				//重点注意——输出的结果，其实是按照图像的列扫描的
				int xOrigin = w;
				int yOrigin = h;

				//应该偏移的位置
				int xOffset = xOrigin * m_anchor_stride;
				int yOffset = yOrigin * m_anchor_stride;

				DhpRect temp = { 0, 0, 0, 0 };

				temp.m_left = xOffset + m_ori_anchors_vec[n].m_left;
				temp.m_top = yOffset + m_ori_anchors_vec[n].m_top;
				temp.m_right = xOffset + m_ori_anchors_vec[n].m_right;
				temp.m_bottom = yOffset + m_ori_anchors_vec[n].m_bottom;

				m_prop_anchors_vec.push_back(temp);
			}
		}
	}
	return OBJDETTRACK_SUCESS;
}

int ObjDet::ProposalRegression()
{
	//计算回归之后的矩形框的位置

	for (int w = 0; w < m_propWidth; w++)
	{
		//列扫描
		for (int h = 0; h < m_propHeight; h++)
		{
			for (int n = 0; n < m_propChannel; n++)
			{
				int index = ((w * m_propHeight) + h) * m_propChannel + n;

				float anchors_x = (m_prop_anchors_vec[index].m_left + m_prop_anchors_vec[index].m_right) / 2;
				float anchors_y = (m_prop_anchors_vec[index].m_top + m_prop_anchors_vec[index].m_bottom) / 2;

				float anchors_width = m_prop_anchors_vec[index].m_right - m_prop_anchors_vec[index].m_left;
				float anchors_height = m_prop_anchors_vec[index].m_bottom - m_prop_anchors_vec[index].m_top;

				//新的中心位置
				float x = m_objVec[index].bbox.center.x * anchors_width + anchors_x;
				float y = m_objVec[index].bbox.center.y * anchors_height + anchors_y;

				//新的尺寸
				float width = anchors_width * exp(m_objVec[index].bbox.size.width);
				float height = anchors_height * exp(m_objVec[index].bbox.size.height);

				m_objVec[index].left = x - width / 2;
				m_objVec[index].right = x + width / 2;
				m_objVec[index].top = y - height / 2;
				m_objVec[index].bottom = y + height / 2;
			}
		}
	}
	return OBJDETTRACK_SUCESS;
}

//对检测到的候选框进行边界检查，将超出边界的
int ObjDet::ProposalBoundDetect()
{
	//检查是否超出边界
	int propCnt = m_objVec.size();

	for (int i = 0; i < propCnt; i++)
	{
		//为了便于计算，将尺寸缩放到原来的图像对应的尺寸（500*375）
		m_objVec[i].left /= m_img_scale;
		m_objVec[i].right /= m_img_scale;
		m_objVec[i].top /= m_img_scale;
		m_objVec[i].bottom /= m_img_scale;

		float left = m_objVec[i].left;
		float top = m_objVec[i].top;
		float right = m_objVec[i].right;
		float bottom = m_objVec[i].bottom;

		if (left < 0)
		{
			left = 0;
		}
		if (top < 0)
		{
			top = 0;
		}
		if (right > m_oriWidth - 1)
		{
			right = m_oriWidth - 1;
		}
		if (bottom > m_oriHeight - 1)
		{
			bottom = m_oriHeight - 1;
		}

		m_objVec[i].left = left;
		m_objVec[i].right = right;
		m_objVec[i].top = top;
		m_objVec[i].bottom = bottom;
	}

	return OBJDETTRACK_SUCESS;
}

//滤除小尺寸候选框
int ObjDet::RemoveSmallBox()
{
	vector<ObjResult> tempVec;
	int propCnt = m_objVec.size();

	for (int i = 0; i < propCnt; i++)
	{
		float width = m_objVec[i].right - m_objVec[i].left;
		float height = m_objVec[i].bottom - m_objVec[i].top;
		if (width >= m_prop_min_width && height >= m_prop_min_width)
		{
			tempVec.push_back(m_objVec[i]);
		}
	}
	m_objVec = tempVec;
	return OBJDETTRACK_SUCESS;
}

//比较前景和背景概率大小，进行过滤
int ObjDet::RemoveBackBox()
{
	vector<ObjResult> tempVec;
	for (int i = 0; i < m_objVec.size(); i++)
	{
		if (m_objVec[i].fg_prob > m_objVec[i].bg_prob)
		{
			tempVec.push_back(m_objVec[i]);
		}
	}
	m_objVec = tempVec;
	return OBJDETTRACK_SUCESS;
}


int ObjDet::NonMaximalSuppression()
{
	//根据置信度从高到低进行排序
	std::sort(m_objVec.begin(), m_objVec.end(), compare1);

	//对数目进行限定
#ifdef SHOW_DEBUG_INFO
	time_t startnms = clock();
#endif

	vector<ObjResult> temp;
	if (m_objVec.size() > m_per_nms_topN)
	{
		for (int i = 0; i < m_per_nms_topN; i++)
		{
			temp.push_back(m_objVec[i]);
		}
		m_objVec = temp;
	}

	//非极值抑制：原理非常简单——按照置信度，从前往后遍历，去掉有较大交叠面积的就行了

	//GPU版本
	ObjResult* iter = &m_objVec[0];
	int output[12000] = { 0 };
	nms_use_gpu(m_per_nms_topN, iter, m_nms_overlap_thres, output);

	//CPU版本
	//for (vector<ObjResult>::iterator iter1 = m_objVec.begin(); iter1 != m_objVec.end() - 1;)
	//{
	//	//第一个矩形框的面积
	//	float left1 = iter1->left;
	//	float right1 = iter1->right;
	//	float top1 = iter1->top;
	//	float bottom1 = iter1->bottom;
	//	float area1 = (right1 - left1) * (bottom1 - top1);

	//	for (vector<ObjResult>::iterator iter2 = iter1 + 1; iter2 != m_objVec.end();)
	//	{
	//		float left2 = iter2->left;
	//		float right2 = iter2->right;
	//		float top2 = iter2->top;
	//		float bottom2 = iter2->bottom;
	//		float area2 = (right2 - left2) * (bottom2 - top2);

	//		//计算交叠面积比例
	//		float newLeft = max(left1, left2);
	//		float newRight = min(right1, right2);
	//		float newTop = max(top1, top2);
	//		float newBottom = min(bottom1, bottom2);

	//		//不相交
	//		if (newLeft > newRight || newTop > newBottom)
	//		{
	//			iter2++;
	//			continue;
	//		}

	//		float area3 = (newRight - newLeft) * (newBottom - newTop);
	//		float ratio = area3 / (area1 + area2 - area3);

	//		//此处阈值设置的比较低
	//		if (ratio > m_nms_overlap_thres)
	//		{
	//			//如果是最末尾一个元素，需要额外注意，因为erase在此时返回值不对
	//			if (iter2 == m_objVec.end() - 1)
	//			{
	//				m_objVec.pop_back();
	//				iter2 = m_objVec.end();
	//			}
	//			else
	//			{
	//				iter2 = m_objVec.erase(iter2);//vector::erase()：从指定容器删除指定位置的元素或某段范围内的元素,只想删除的下一个元素
	//			}
	//		}
	//		else
	//		{
	//			iter2++;
	//		}
	//	}

	//	//判断一下iter1是否为最后一个元素
	//	if (iter1 == m_objVec.end() - 1)
	//	{
	//		break;
	//	}
	//	else
	//	{
	//		iter1++;
	//	}
	//}


	//将置信度最高的300个保留下来
	//对数目进行限定

	//GPU部分
	temp.clear();
	if (m_objVec.size() > m_after_nms_topN)
	{
		for (int i = 0; output[i] != 0; i++)
			temp.push_back(m_objVec[output[i] - 1]);
		m_objVec = temp;
	}

	//CPU部分
	//temp.clear();
	//if (m_objVec.size() > m_after_nms_topN)
	//{
	//	for (int i = 0; i < m_after_nms_topN; i++)
	//	{
	//		temp.push_back(m_objVec[i]);
	//	}
	//	m_objVec = temp;
	//}

#ifdef SHOW_DEBUG_INFO
	time_t endnms = clock();
	cout << "nms所花费的时间：" << endnms - startnms << endl;
#endif
	return OBJDETTRACK_SUCESS;
}

int ObjDet::ShowPropsalResult()
{
	//将检测框绘制出来
	Mat imgDetectResult = m_oriImg.clone();
	for (int i = 0; i < m_objVec.size(); i++)
	{
		int left = m_objVec[i].left;
		int right = m_objVec[i].right;
		int top = m_objVec[i].top;
		int bottom = m_objVec[i].bottom;

		cv::rectangle(imgDetectResult, cv::Rect(left, top, right - left, bottom - top), cv::Scalar(0, 0, 255), 1);
	}
	namedWindow("proposal", 0);
	imshow("proposal", imgDetectResult);
	waitKey(1);

	return OBJDETTRACK_SUCESS;
}

int ObjDet::PrepareClassifyData()
{
	//开始进行识别
	//为data层设置数据
	//重点注意，这里的代码，其实已经是与模型文件发生耦合了，如果修改了模型文件，这里也需要相应的修改
	const boost::shared_ptr<caffe::Blob<float>> shared_output_blob = m_proposal_net_->blob_by_name("conv5");
	const boost::shared_ptr<caffe::Blob<float>> data_blob = m_classify_net_->blob_by_name("data");
	data_blob->ReshapeLike(*shared_output_blob);
	data_blob->CopyFrom(*shared_output_blob);

	//为rois层设置数据
	const boost::shared_ptr<caffe::Blob<float>> rois_blobs = m_classify_net_->blob_by_name("rois");

	//准备rois数据
	//这里的数据各个维度，是非常诡异的
	int roi_num = m_objVec.size();
	int roi_channel = 5;	//在坐标的基础上，再增加一个变量，用于表示批次
	int roi_height = 1;
	int roi_width = 1;

	vector<int> roisShape;
	roisShape.push_back(roi_num);
	roisShape.push_back(roi_channel);
	roisShape.push_back(roi_height);
	roisShape.push_back(roi_width);

	rois_blobs->Reshape(roisShape);

	//设置相应的回归框的位置，注意缩放
	float* roisdata = rois_blobs->mutable_cpu_data();
	for (int n = 0; n < roi_num; n++)
	{
		roisdata[n * roi_channel + 0] = 0;	//批次0
		roisdata[n * roi_channel + 1] = m_objVec[n].left * m_img_scale;
		roisdata[n * roi_channel + 2] = m_objVec[n].top * m_img_scale;
		roisdata[n * roi_channel + 3] = m_objVec[n].right * m_img_scale;
		roisdata[n * roi_channel + 4] = m_objVec[n].bottom * m_img_scale;
	}
	return OBJDETTRACK_SUCESS;
}

int ObjDet::ClassifyNetRun()
{
#ifdef SHOW_DEBUG_INFO
	time_t start5 = clock();
#endif

	m_classify_net_->ForwardFrom(0);

#ifdef SHOW_DEBUG_INFO
	time_t end5 = clock();
	cout << "分类的前向计算时间： " << end5 - start5 << endl;
#endif

	return OBJDETTRACK_SUCESS;
}

int ObjDet::ClassifyExtract()
{
	m_classifyVec.clear();

	boost::shared_ptr<caffe::Blob<float>> prob_blob = m_classify_net_->blob_by_name("cls_prob");
	int prob_num = prob_blob->num();
	int prob_channel = prob_blob->channels();//36
	int prob_height = prob_blob->height();
	int prob_width = prob_blob->width();
	int prob_cnt = prob_blob->count();
	const float* probData = prob_blob->cpu_data();

	boost::shared_ptr<caffe::Blob<float>> pred_blob = m_classify_net_->blob_by_name("bbox_pred");
	int box_num = pred_blob->num();
	int box_channel = pred_blob->channels();
	int box_height = pred_blob->height();
	int box_width = pred_blob->width();
	int box_cnt = pred_blob->count();
	const float* boxData = pred_blob->cpu_data();

	//进行分类，从21个类别中找到概率最高的那个
	//概率数据的存储顺序——先图像宽度方向，然后高度方向，再然后才是21个概率值
	//probnum是检测框的数目
	//这里没有再管原来的图像尺寸的存储顺序了，直接把概率结果当做二维数组来处理
	//是一个probnum*21*1的三维矩阵
	for (int i = 0; i < prob_num; i++)
	{
		float maxScore = -1.0f;
		int maxIndex = -1;
		//第0个是背景的概率，不考虑
		for (int j = 1; j < prob_channel; j++)
		{
			if (probData[i * prob_channel + j] > maxScore)
			{
				maxScore = probData[i * prob_channel + j];
				maxIndex = j;
			}
		}
		ObjResult temp;
		temp.fg_prob = maxScore;
		temp.classIndex = maxIndex;

		//存储位置,这里存储的只是回归量，精确地位置需要结合propVec3
		//注意，这里采集出来的坐标数据，在方框的扫描顺序上，是先图像宽度，再图像高度，然后36个坐标
		//中心位置的偏移量
		//等价于一个prob_num*21*4的三维矩阵
		float x = boxData[i * box_channel + 4 * maxIndex + 0];
		float y = boxData[i * box_channel + 4 * maxIndex + 1];

		//宽高的回归量
		float width = boxData[i * box_channel + 4 * maxIndex + 2];
		float height = boxData[i * box_channel + 4 * maxIndex + 3];

		//中心位置的偏移量
		temp.bbox.size = cv::Size2f(width, height);
		temp.bbox.center = cv::Point2f(x, y);

		m_classifyVec.push_back(temp);
	}
	return OBJDETTRACK_SUCESS;
}

int ObjDet::ClassifyRegression()
{
	//计算精准的坐标位置
	//计算回归之后的矩形框的位置
	int classifyCnt = m_classifyVec.size();
	for (int i = 0; i < classifyCnt; i++)
	{
		int index = i;
		float width0 = m_objVec[index].right - m_objVec[index].left;
		float height0 = m_objVec[index].bottom - m_objVec[index].top;
		float x0 = (m_objVec[index].right + m_objVec[index].left) / 2;
		float y0 = (m_objVec[index].bottom + m_objVec[index].top) / 2;

		//候选框的尺寸是针对原图的，要缩放到600对应的尺寸
		width0 *= m_img_scale;
		height0 *= m_img_scale;
		x0 *= m_img_scale;
		y0 *= m_img_scale;

		//新的中心位置
		float x = m_classifyVec[index].bbox.center.x * width0 + x0;
		float y = m_classifyVec[index].bbox.center.y * height0 + y0;

		//新的尺寸
		float width = width0 * exp(m_classifyVec[index].bbox.size.width);
		float height = height0 * exp(m_classifyVec[index].bbox.size.height);

		m_classifyVec[index].left = x - width / 2;
		m_classifyVec[index].right = x + width / 2;
		m_classifyVec[index].top = y - height / 2;
		m_classifyVec[index].bottom = y + height / 2;
	}

	//为了便于计算，将尺寸缩放到原来的图像对应的尺寸（500*375）
	for (int i = 0; i < classifyCnt; i++)
	{
		m_classifyVec[i].left /= m_img_scale;
		m_classifyVec[i].right /= m_img_scale;
		m_classifyVec[i].top /= m_img_scale;
		m_classifyVec[i].bottom /= m_img_scale;
	}
	return OBJDETTRACK_SUCESS;
}

//边界检查

int ObjDet::ClassifyBoundDetect()
{
	//检查是否超出边界
	int classifyCnt = m_classifyVec.size();

	for (int i = 0; i < classifyCnt; i++)
	{
		float left = m_classifyVec[i].left;
		float top = m_classifyVec[i].top;
		float right = m_classifyVec[i].right;
		float bottom = m_classifyVec[i].bottom;

		if (left < 0)
		{
			left = 0;
		}
		if (top < 0)
		{
			top = 0;
		}
		if (right > m_oriWidth - 1)
		{
			right = m_oriWidth - 1;
		}
		if (bottom > m_oriHeight - 1)
		{
			bottom = m_oriHeight - 1;
		}

		m_classifyVec[i].left = left;
		m_classifyVec[i].right = right;
		m_classifyVec[i].top = top;
		m_classifyVec[i].bottom = bottom;
	}
	return OBJDETTRACK_SUCESS;
}

//滤除小尺寸候选框
int ObjDet::RemoveSmallClassifyBox()
{
	//过滤掉尺寸过小的框
	vector<ObjResult> temp;
	int classifyCnt = m_classifyVec.size();

	for (int i = 0; i < classifyCnt; i++)
	{
		float width = m_classifyVec[i].right - m_classifyVec[i].left;
		float height = m_classifyVec[i].bottom - m_classifyVec[i].top;
		if (width >= m_prop_min_width && height >= m_prop_min_width
			&& width <= m_prop_max_width && height <= m_prop_max_height)
		{
			temp.push_back(m_classifyVec[i]);
		}
	}
	m_classifyVec = temp;

	return OBJDETTRACK_SUCESS;
}

int ObjDet::NonMaximalSuppression_classify()
{
	//去掉低置信度的框
	vector<ObjResult> temp;
	int classifyCnt = m_classifyVec.size();

	for (int i = 0; i < classifyCnt; i++)
	{
		if (m_classifyVec[i].fg_prob > m_CONF_THRESH)
		{
			temp.push_back(m_classifyVec[i]);
		}
	}
	m_classifyVec = temp;
	if (0 == m_classifyVec.size())
	{
		return OBJDETTRACK_SUCESS;
	}

	//根据置信度从高到低进行排序
	std::sort(m_classifyVec.begin(), m_classifyVec.end(), compare1);

	//非极值抑制
	//原理非常简单——按照置信度，从前往后遍历，去掉有较大交叠面积的就行了
	for (vector<ObjResult>::iterator iter1 = m_classifyVec.begin(); iter1 != m_classifyVec.end() - 1;)
	{
		//第一个矩形框的面积
		float left1 = iter1->left;
		float right1 = iter1->right;
		float top1 = iter1->top;
		float bottom1 = iter1->bottom;
		float area1 = (right1 - left1) * (bottom1 - top1);

		for (vector<ObjResult>::iterator iter2 = iter1 + 1; iter2 != m_classifyVec.end();)
		{
			float left2 = iter2->left;
			float right2 = iter2->right;
			float top2 = iter2->top;
			float bottom2 = iter2->bottom;
			float area2 = (right2 - left2) * (bottom2 - top2);

			//计算交叠面积比例
			float newLeft = max(left1, left2);
			float newRight = min(right1, right2);
			float newTop = max(top1, top2);
			float newBottom = min(bottom1, bottom2);

			//不相交
			if (newLeft > newRight || newTop > newBottom)
			{
				iter2++;
				continue;
			}

			float area3 = (newRight - newLeft) * (newBottom - newTop);
			float ratio = area3 / (area1 + area2 - area3);

			//此处阈值设置的比较低
			//当前的值是0.9
			if (ratio > m_NMS_THRESH)
			{
				//如果是最末尾一个元素，需要额外注意，因为erase在此时返回值不对
				if (iter2 == m_classifyVec.end() - 1)
				{
					m_classifyVec.pop_back();
					iter2 = m_classifyVec.end();
				}
				else
				{
					iter2 = m_classifyVec.erase(iter2);
				}
			}
			else
			{
				iter2++;
			}
		}

		//判断一下iter1是否为最后一个元素
		if (iter1 == m_classifyVec.end() - 1)
		{
			break;
		}
		else
		{
			iter1++;
		}
	}

	vector<ObjResult> temp1;//判断是人才保留该候选框
	for (int i = 0; i < m_classifyVec.size(); i++)
	{
		if (m_classifyVec[i].classIndex == 15)
		{
			temp1.push_back(m_classifyVec[i]);
		}
	}
	m_classifyVec.clear();
	for (int i = 0; i < temp1.size(); i++)
	{
		m_classifyVec.push_back(temp1[i]);
	}

	return OBJDETTRACK_SUCESS;
}

int ObjDet::ShowClassifyResult()
{
	//将检测框绘制出来
	Mat imgDetectResult = m_oriImg.clone();
	for (int i = 0; i < m_classifyVec.size(); i++)
	{
		int left = m_classifyVec[i].left;
		int right = m_classifyVec[i].right;
		int top = m_classifyVec[i].top;
		int bottom = m_classifyVec[i].bottom;

		cv::rectangle(imgDetectResult, cv::Rect(left, top, right - left, bottom - top), cv::Scalar(0, 255, 0), 2);
	}
	namedWindow("imgDetectResult", 0);
	imshow("imgDetectResult", imgDetectResult);

	waitKey(1);
	/*Mat resizedimg;
	resize(imgDetectResult, resizedimg, Size(320,240));*/

	char image_name[25];
	static int num = 1;
	IplImage qImg;
	if (m_classifyVec.size() > 0)
	{
		qImg = IplImage(imgDetectResult); // cv::Mat -> IplImage
		//sprintf(image_name, "%s%d%s", "..\\jieguo\\image", num++, ".jpg");
		//sprintf(image_name, "..\\jieguo\\%d.jpg", num++);
		sprintf(image_name, "..\\jieguo\\%d.jpg",num++);
		cvSaveImage(image_name, &qImg);

		//上传到服务器
		TCPClient tcp;
		//IplImage qImg = IplImage(imgDetectResult);
		tcp.run(image_name);

		Result r = pushInit(host, appKey, masterSecret, "编码");
		if (r != SUCCESS){
			printf("pushInit for app failed: ret=%d\n", r);
			return -1;
		}
		////对单个用户的消息推送
		tosingletest();
		Sleep(10000);
	}

	return OBJDETTRACK_SUCESS;
}

int ObjDet::ProcessSingleImg(Mat& currImg, vector<ObjResult>& detResult)
{
	if (currImg.empty())
	{
		std::cout << "Can not get the image file !" << endl;
		return OBJDETTRACK_FAIL;
	}
	m_oriImg = currImg.clone();

	//准备输入数据
	int flag = PrepInputData();

	flag = ProposalNetRun();

	//从计算结果中提取出原始的候选框位置
	flag = ProposalExtract();

	//计算锚点位置
	flag = CalcPropsalAnchors();

	//对候选框进行回归，得到真实位置
	flag = ProposalRegression();

	flag = ProposalBoundDetect();

	//滤除小尺寸候选框
	flag = RemoveSmallBox();

	//滤除背景概率过高的候选框
	//flag = RemoveBackBox();

	//根据位置交迭的程度，滤除部分置信度较低的候选框
	flag = NonMaximalSuppression();

#ifdef SHOW_DEBUG_INFO
	//将最终的检测结果传出
	//flag = ShowPropsalResult();
#endif	

	//----------------------------------------------------

	//获取用于分类的数据
	flag = PrepareClassifyData();

	flag = ClassifyNetRun();

	flag = ClassifyExtract();

	flag = ClassifyRegression();

	flag = ClassifyBoundDetect();

	//	flag = RemoveSmallClassifyBox();

	flag = NonMaximalSuppression_classify();

	//-----------------------------------------------------

	flag = ClearInputData();

#ifdef SHOW_DEBUG_INFO
	//将最终的检测结果传出
	flag = ShowClassifyResult();
#endif	

	detResult.clear();
	detResult = m_classifyVec;

	return OBJDETTRACK_SUCESS;
}