#include "hashSdf.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define ISZERO(x) x < 0.00000000001 && x > -0.00000000001
//dangerous code, be careful
#define GO_TO_ERROR(status, msg) if (status != cudaSuccess){fprintf(stderr, msg); goto Error;}


//candidate big prime 87719,175447, 350899, 701819, 1403641, 2807303
//define the hash parameter
#define HASH_P1 73856093LL
#define HASH_P2 19349669LL
#define HASH_P3 83492791LL
#define HASH_SIZE 175447LL
#define HASH_BUCKET_SIZE 20
//define hash sdf data structure
Voxel* _voxel_pool;
HashEntry* _hash_sdf;
int _pool_size;

__device__ inline double hashCode(short x, short y, short z)
{
	return (x * HASH_P1 + y * HASH_P2 + z * HASH_P3) % HASH_SIZE;
}

__device__ inline double hashCode(short position[3])
{
	return (position[0] * HASH_P1 + position[1] * HASH_P2 + position[2] * HASH_P3) % HASH_SIZE;
}

__global__ void initialHashEntry(HashEntry* table, int blockSize, int pool_size)
{
	int id = threadIdx.x*blockSize+blockIdx.x;
	if(id < pool_size)
	{
		HashEntry* bucket = table + (id * HASH_BUCKET_SIZE);
		for(int i = 0; i < HASH_BUCKET_SIZE; i++)
		{
			bucket[i].pointer = -1;
		}
	}
}

int createHashSdf()
{
	_pool_size = HASH_SIZE * HASH_BUCKET_SIZE;

	cudaError_t cudaStatus = cudaSuccess;
	cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&_voxel_pool, _pool_size * sizeof(Voxel));
    GO_TO_ERROR(cudaStatus,  "cudaMalloc failed!");

	cudaStatus = cudaMalloc((void**)&_hash_sdf, _pool_size * sizeof(HashEntry));
    GO_TO_ERROR(cudaStatus,  "cudaMalloc failed!");

	int blockSize = 256;
	int blockdim = (_pool_size / blockSize) + 1;
	initialHashEntry<<<blockdim, blockSize>>>(_hash_sdf, blockSize, _pool_size);

	cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

	return cudaStatus;

	Error:
    cudaFree(_voxel_pool);
	cudaFree(_hash_sdf);
    
    return cudaStatus;

}
