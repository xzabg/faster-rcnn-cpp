#include "ObjDetTrack.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>

#include <stdio.h>
#include <cstdlib>
#include <vector>

#define DIVUP(m,n) ((m)/(n) + ((m)%(n) > 0))
const int threadsPerBlock = (sizeof(unsigned long long) * 8);

/* 以下为调用GPU程序 */
__device__ inline float devIoU(ObjResult const * const a, ObjResult const * const b)
{
	float left = max( a->left ,b->left );
	float right = min( a->right ,b->right );
	float top = max( a->top ,b->top );
	float bottom = min(a->bottom ,b->bottom );

	float width = max(right - left  , 0.f);
	float height = max(bottom - top  , 0.f);
	float interS = width * height;
	float Sa = (a->right - a->left ) * (a->bottom - a->top );
	float Sb = (b->right - b->left ) * (b->bottom - b->top );

	return interS / (Sa + Sb - interS);
}
__global__ void nms_kernel(const int n_boxes, const float nms_overlap_thres, ObjResult *iter, unsigned long long *dev_mask)
{
	const int row_start = blockIdx.y, col_start = blockIdx.x;
	const int row_size = min(n_boxes - row_start * threadsPerBlock, threadsPerBlock), col_size = min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

	__shared__ ObjResult* block_boxes[threadsPerBlock];
	if (threadIdx.x < col_size)
	{
		block_boxes[threadIdx.x] = iter + threadsPerBlock * col_start + threadIdx.x;
		/*
		block_boxes[threadIdx.x]->left   = (iter + threadsPerBlock * col_start + threadIdx.x)->left;
		block_boxes[threadIdx.x]->right  = (iter + threadsPerBlock * col_start + threadIdx.x)->right;
		block_boxes[threadIdx.x]->bottom = (iter + threadsPerBlock * col_start + threadIdx.x)->bottom;
		block_boxes[threadIdx.x]->top    = (iter + threadsPerBlock * col_start + threadIdx.x)->top;	*/
	}
	__syncthreads();

	if (threadIdx.x < row_size)
	{
		const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
		const ObjResult *cur_box = iter + cur_box_idx;
		int i = 0;
		unsigned long long t = 0;
		int start = 0;
		if (row_start == col_start) start = threadIdx.x + 1;
		for (i = start; i < col_size; i++)
		{
			if (devIoU(cur_box, block_boxes[i]) > nms_overlap_thres)
			{
				t |= 1ULL << i;
			}
		}
		const int col_blocks = DIVUP(n_boxes, threadsPerBlock);
		dev_mask[cur_box_idx * col_blocks + col_start] = t;
	}
}

void nms_use_gpu(int m_per_nms_topN, ObjResult* iter, float nms_overlap_thres, int *output)
{
	const int col_blocks = DIVUP(m_per_nms_topN, threadsPerBlock);
	cudaError_t cudaStatus;
	ObjResult *objresult = NULL;
	unsigned long long *mask_dev = NULL;
	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);

	//Allocate GPU buffers
	//vector<ObjResult>::iterator iter;
	//cudaStatus = 
	cudaMalloc((void**)&objresult, m_per_nms_topN * sizeof(ObjResult));
	//cudaStatus = 
	cudaMemcpy(objresult, iter, m_per_nms_topN * sizeof(ObjResult), cudaMemcpyHostToDevice);

	//cudaStatus = 
	cudaMalloc((void**)&mask_dev, col_blocks * m_per_nms_topN * sizeof(unsigned long long));

	dim3 blocks(DIVUP(m_per_nms_topN, threadsPerBlock), DIVUP(m_per_nms_topN, threadsPerBlock));
	dim3 threads(threadsPerBlock);
	//nms_kernel <<<blocks, threads >>> (m_per_nms_topN, nms_overlap_thres, objresult, mask_dev);
	nms_kernel << <blocks, threads >> >(m_per_nms_topN, nms_overlap_thres, objresult, mask_dev);

	std::vector<unsigned long long> mask_host(m_per_nms_topN * col_blocks);
	cudaMemcpy(&mask_host[0], mask_dev, sizeof(unsigned long long) * m_per_nms_topN * col_blocks, cudaMemcpyDeviceToHost);

	std::vector<unsigned long long> remv(col_blocks);
	memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);

	std::vector<int> keep;
	keep.reserve(m_per_nms_topN);
	for (int i = 0; i < m_per_nms_topN; i++)
	{
		int nblock = i / threadsPerBlock;
		int inblock = i % threadsPerBlock;

		if (!(remv[nblock] & (1ULL << inblock)))
		{
			keep.push_back(i+1);  // to matlab's index

			unsigned long long *p = &mask_host[0] + i * col_blocks;
			for (int j = nblock; j < col_blocks; j++)
			{
				remv[j] |= p[j];
			}
		}
	}

	memcpy(output, &keep[0], (int)keep.size() * sizeof(int));

	cudaFree(objresult);
	cudaFree(mask_dev);
}