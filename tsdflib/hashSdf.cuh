#ifndef _CUDA_HASH_SDF_H
#define _CUDA_HASH_SDF_H

typedef unsigned char uchar;

struct Voxel {
	float sdf;
	uchar colorRGB[3];
	uchar weight;
};

struct HashEntry {
	short position[3];
	short offset;
	int pointer;
};

int createHashSdf();


#endif