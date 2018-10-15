#define SHOW_DEBUG_INFO	

#include <stdio.h>  // for snprintf
#include <string>
#include <vector>
#include <math.h>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

//#include "CompressiveTracker.h"
#include "caffe.hpp"
using namespace caffe;

#define max(a, b) (((a)>(b)) ? (a) :(b))
#define min(a, b) (((a)<(b)) ? (a) :(b))

#define OBJDETTRACK_SUCESS 1
#define OBJDETTRACK_FAIL -1

struct ObjResult
{
	float left;
	float top;
	float right;
	float bottom;

	float fg_prob;
	float bg_prob;

	cv::RotatedRect bbox;	//早期用于记录外框的回归的结果，此后不再使用，矩形框的位置直接存储四边
	int classIndex;
};

struct DhpRect
{
	DhpRect();
	DhpRect(float left, float top, float right, float bottom)
	{
		m_left = left;
		m_right = right;
		m_top = top;
		m_bottom = bottom;
	}

	float m_left;
	float m_right;
	float m_top;
	float m_bottom;
};

//////////////////////////////////////////////////////////////////////////

class ObjDet
{
public:
	ObjDet();
	~ObjDet();
	int ProcessSingleImg(Mat& currImg, vector<ObjResult>& detResult);
	int Initial();

private:

	int Load();
	int PrepInputData();
	int ClearInputData();

	//进行候选区域的检测
	int ProposalNetRun();

	//提取proposal检测结果
	int ProposalExtract();

	//计算最后的卷基层对应的锚点位置
	int CalcPropsalAnchors();

	//对proposal检测结果进行回归
	int ProposalRegression();

	//边界检查
	int ProposalBoundDetect();

	//滤除小尺寸候选框
	int RemoveSmallBox();

	//滤除背景概率过高的候选框
	int RemoveBackBox();

	int ShowPropsalResult();

	//----------------------------------------

	//准备好用于分类的数据
	int PrepareClassifyData();

	//对候选框进行非极值抑制
	int NonMaximalSuppression();

	//进行目标的识别
	int ClassifyNetRun();

	int ClassifyExtract();

	int ClassifyRegression();

	//边界检查
	int ClassifyBoundDetect();

	//滤除小尺寸候选框
	int RemoveSmallClassifyBox();

	int NonMaximalSuppression_classify();

	//显示结果
	int ShowClassifyResult();

private:

	boost::shared_ptr< Net<float> > m_proposal_net_;
	boost::shared_ptr< Net<float> > m_classify_net_;

	boost::shared_ptr<caffe::Blob<float>> m_blob_bbox;
	boost::shared_ptr<caffe::Blob<float>> m_blob_prob;

	Mat m_oriImg;
	Mat m_oriImg32f;
	Mat m_resizedImg32f;
	//最终用于作为输入数据的内存
	float* m_data_buf;

	int m_oriWidth;
	int m_oriHeight;
	int m_resizedWidth;
	int m_resizedHeight;

	//图像的缩放比例
	float m_img_scale;

	//候选框
	vector<ObjResult> m_objVec;
	//最终检测结果
	vector<ObjResult> m_classifyVec;

	//非极值抑制之前的候选框最大数目，按照置信度排序
	int m_per_nms_topN;
	//非最大值抑制时的窗口交叠比例
	float m_nms_overlap_thres;
	//最终的候选框的最大数目
	int m_after_nms_topN;
	int m_use_gpu;
	//检测时图像最小边的长度	
	float  m_min_input_side;

	//分类时置信度的阈值
	float m_CONF_THRESH;
	//分类结果的非极值抑制窗口交叠比例
	float m_NMS_THRESH;

	//区域生成过程中的各种参数，就是最后一个卷积层
	int m_propNum;
	int m_propChannel;
	int m_propHeight;
	int m_propWidth;

	//锚点基准坐标
	vector<DhpRect> m_ori_anchors_vec;
	int m_ori_anchors_cnt;
	vector<DhpRect> m_prop_anchors_vec;
	int m_prop_anchors_cnt;
	//计算锚点位置时，步长
	int m_anchor_stride;

	//候选框的最小尺寸
	int m_prop_min_width;
	int m_prop_max_width;
	int m_prop_max_height;

};