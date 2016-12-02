#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "tsdf.cuh"

#include <stdio.h>

#define isnan(x) ((x) != (x)) 

//dangerous code, be careful
#define GO_TO_ERROR(status, msg) if (status != cudaSuccess){fprintf(stderr, msg); goto Error;}


float _minX;
float _minY;
float _minZ;
int _Xlen;
int _Ylen;
int _Zlen;
float _devide;
float _epsi;
float _thre;
float* _dists;
float* _weights;
float* _n_x;
float* _n_y;
float* _n_z;
float* _local;

__device__ double getProjectd(float x,float y,float z,float ox,float oy,float oz,float nx,float ny,float nz,float mx,float my,float mz)
{
	return ((x-ox)*nx + (y-oy)*ny + (z-oz)*nz)/(nx*mx+ny*my+nz*mz);
}

__global__ void getDepth(float ix,float iy,float devide, float minX,float minY,float minZ, int Xlen,int Ylen,int Zlen, int p_xlen, float* mc, float* p_dists, float* p_nx, float* p_ny, float* p_nz,float* dists, float* weights, float* n_x,float* n_y, float* n_z)
{
	int idy = threadIdx.x;
	int idx = blockIdx.x;
	//int idx = blockIdx.x;
	//int idy = blockIdx.y;
	int pid = (idy*p_xlen)+idx;
	float x = ix + idx*devide;
	float y = iy + idy*devide;

	float maxX = minX + devide*Xlen;
	float maxY = minY + devide*Ylen;
	float maxZ = minZ + devide*Zlen;

	p_dists[pid] = -10000;
	//求z值区间
	float min = -10000,max = 10000;
	if(mc[0*4+2] > 0)
	{
		float left = (minX - mc[0*4+0]*x - mc[0*4+1]*y - mc[0*4+3])/mc[0*4+2];
		if(left > min) min = left;
		float right = (maxX - mc[0*4+0]*x - mc[0*4+1]*y - mc[0*4+3])/mc[0*4+2];
		if(right < max) max = right;
	}else if(mc[0*4+2] < 0)
	{
		float left = (minX - mc[0*4+0]*x - mc[0*4+1]*y - mc[0*4+3])/mc[0*4+2];
		if(left < max) max = left;
		float right = (maxX - mc[0*4+0]*x - mc[0*4+1]*y - mc[0*4+3])/mc[0*4+2];
		if(right > min) min = right;
	}
	if(mc[1*4+2] > 0)
	{
		float left = (minY - mc[1*4+0]*x - mc[1*4+1]*y - mc[1*4+3])/mc[1*4+2];
		if(left > min) min = left;
		float right = (maxY - mc[1*4+0]*x - mc[1*4+1]*y - mc[1*4+3])/mc[1*4+2];
		if(right < max) max = right;
	}else if(mc[1*4+2] < 0)
	{
		float left = (minY - mc[1*4+0]*x - mc[1*4+1]*y - mc[1*4+3])/mc[1*4+2];
		if(left < max) max = left;
		float right = (maxY - mc[1*4+0]*x - mc[1*4+1]*y - mc[1*4+3])/mc[1*4+2];
		if(right > min) min = right;
	}
	if(mc[2*4+2] > 0)
	{
		float left = (minZ - mc[2*4+0]*x - mc[2*4+1]*y - mc[2*4+3])/mc[2*4+2];
		if(left > min) min = left;
		float right = (maxZ - mc[2*4+0]*x - mc[2*4+1]*y - mc[2*4+3])/mc[2*4+2];
		if(right < max) max = right;
	}else if(mc[2*4+2] < 0)
	{
		float left = (minZ - mc[2*4+0]*x - mc[2*4+1]*y - mc[2*4+3])/mc[2*4+2];
		if(left < max) max = left;
		float right = (maxZ - mc[2*4+0]*x - mc[2*4+1]*y - mc[2*4+3])/mc[2*4+2];
		if(right > min) min = right;
	}

	if(min <= max)
	{
		//获得第一个格子
		float fx = mc[0*4+0]*x + mc[0*4+1]*y + mc[0*4+2]*min + mc[0*4+3];
		float fy = mc[1*4+0]*x + mc[1*4+1]*y + mc[1*4+2]*min + mc[1*4+3];
		float fz = mc[2*4+0]*x + mc[2*4+1]*y + mc[2*4+2]*min + mc[2*4+3];

		float originx = mc[0*4+0]*x + mc[0*4+1]*y + mc[0*4+3];
		float originy = mc[1*4+0]*x + mc[1*4+1]*y + mc[1*4+3];
		float originz = mc[2*4+0]*x + mc[2*4+1]*y + mc[2*4+3];
		
		float d = 0;
		int ox = -1;
		int oy = -1;
		int oz = -1;
		float oldd = 0;

		float step = devide; //步长

		while(d <= max-min)
		{
			//扫描
			int xid = (fx-minX)/devide;
			int yid = (fy-minY)/devide;
			int zid = (fz-minZ)/devide;

			int id = ((zid*Ylen)+yid)*Xlen+xid;
			if(xid < 0 || xid >= Xlen || yid < 0 || yid >= Ylen || zid < 0 || zid >= Zlen)
			{
				ox = -1;
				oldd = 0;
			}
			else if(weights[id] <= 0)
			{
				ox = -1;
				oldd = 0;
			}
			else if(dists[id] <= 0)
			{
				ox = xid;
				oy = yid;
				oz = zid;
				oldd = d;
			}else if(dists[id] > 0 && ox >= 0){
				

				int oid = ((oz*Ylen)+oy)*Xlen+ox;
				//将两个格子的中心分别投影到扫描线上
				double d1 = getProjectd(minX+ox*devide,minY+oy*devide,minZ+oz*devide,originx,originy,originz,n_x[oid],n_y[oid],n_z[oid],mc[0*4+2],mc[1*4+2],mc[2*4+2]);
                if(oldd-d1 > devide || d1-oldd > devide) d1 = oldd; 
				double d2 = getProjectd(minX+xid*devide,minY+yid*devide,minZ+zid*devide,originx,originy,originz,n_x[id],n_y[id],n_z[id],mc[0*4+2],mc[1*4+2],mc[2*4+2]);
				if(d-d2 > devide || d2-d > devide) d2 = d;
				//double d1 = oldd;
				//double d2 = d;

				double rate = (dists[id]-dists[oid])*(-dists[oid]);
				double zerocross = d1+ rate*(d2-d1);

				float nx = n_x[oid] + rate*(n_x[oid] - n_x[id]);
				float ny = n_y[oid] + rate*(n_y[oid] - n_y[id]);
				float nz = n_z[oid] + rate*(n_z[oid] - n_z[id]);
				float n = sqrt(nx*nx+ny*ny+nz*nz);
				p_nx[pid] = nx/n;
				p_ny[pid] = ny/n;
				p_nz[pid] = nz/n;
				
				p_dists[pid] = min+zerocross;
				//p_nx[pid] = n_x[id];
				//p_ny[pid] = n_y[id];
				//p_nz[pid] = n_z[id];
				break;
			}else{
				ox = -1;
				oldd = 0;
			}

			d += step;
			fx+=mc[0*4+2]*step;
			fy+=mc[1*4+2]*step;
			fz+=mc[2*4+2]*step;
		}	
	}
}

__device__ float getData(float* data, int Xlen, int Ylen,int Zlen, int x, int y,int z)
{
	if(x >= 0 && x < Xlen && y >= 0 && y < Ylen && z >= 0 && z < Zlen)
	{
		return data[((z*Ylen)+y)*Xlen+x];
	}else{
		return 0;
	}
}


__device__ float getTriInterWeight(float* weight,int Xlen,int Ylen,int Zlen,int downX, int downY,int downZ, float xd, float yd, float zd)
{
	int upX = downX + 1;
	int upY = downY + 1;
	int upZ = downZ	+ 1;

	int size = Xlen*Ylen*Zlen;

	/*
	int ddd = ((downZ*Ylen)+downY)*Xlen+downX;
	int ddu = ((upZ*Ylen)+downY)*Xlen+downX;
	int dud = ((downZ*Ylen)+upY)*Xlen+downX;
	int duu = ((upZ*Ylen)+upY)*Xlen+downX;
	int udd = ((downZ*Ylen)+downY)*Xlen+upX;
	int udu = ((upZ*Ylen)+downY)*Xlen+upX;
	int uud = ((downZ*Ylen)+upY)*Xlen+upX;
	int uuu = ((upZ*Ylen)+upY)*Xlen+upX;
	*/

	float iw1 = (1-zd)*getData(weight,Xlen,Ylen,Zlen,downX,downY,downZ)+zd*getData(weight,Xlen,Ylen,Zlen,downX,downY,upZ);
	float iw2 = (1-zd)*getData(weight,Xlen,Ylen,Zlen,downX,upY,downZ)+zd*getData(weight,Xlen,Ylen,Zlen,downX,upY,upZ);
	float jw1 = (1-zd)*getData(weight,Xlen,Ylen,Zlen,upX,downY,downZ)+zd*getData(weight,Xlen,Ylen,Zlen,upX,downY,upZ);
	float jw2 = (1-zd)*getData(weight,Xlen,Ylen,Zlen,upX,upY,downZ)+zd*getData(weight,Xlen,Ylen,Zlen,upX,upY,upZ);

	float ww1 =(1-yd)*iw1 + yd*iw2;
	float ww2 =(1-yd)*jw1 + yd*jw2;

	float retw = (1-xd)*ww1 + xd*ww2;

	return retw;
}

__device__ float getTriInter(float* data,float* weight,int Xlen,int Ylen,int Zlen,int downX, int downY,int downZ, float xd, float yd, float zd)
{
	int upX = downX + 1;
	int upY = downY + 1;
	int upZ = downZ	+ 1;

	float i1 = (1-zd)*getData(data,Xlen,Ylen,Zlen,downX,downY,downZ)*getData(weight,Xlen,Ylen,Zlen,downX,downY,downZ)+zd*getData(data,Xlen,Ylen,Zlen,downX,downY,upZ)*getData(weight,Xlen,Ylen,Zlen,downX,downY,upZ);
	float i2 = (1-zd)*getData(data,Xlen,Ylen,Zlen,downX,upY,downZ)*getData(weight,Xlen,Ylen,Zlen,downX,upY,downZ)+zd*getData(data,Xlen,Ylen,Zlen,downX,upY,upZ)*getData(weight,Xlen,Ylen,Zlen,downX,upY,upZ);
	float j1 = (1-zd)*getData(data,Xlen,Ylen,Zlen,upX,downY,downZ)*getData(weight,Xlen,Ylen,Zlen,upX,downY,downZ)+zd*getData(data,Xlen,Ylen,Zlen,upX,downY,upZ)*getData(weight,Xlen,Ylen,Zlen,upX,downY,upZ);
	float j2 = (1-zd)*getData(data,Xlen,Ylen,Zlen,upX,upY,downZ)*getData(weight,Xlen,Ylen,Zlen,upX,upY,downZ)+zd*getData(data,Xlen,Ylen,Zlen,upX,upY,upZ)*getData(weight,Xlen,Ylen,Zlen,upX,upY,upZ);

	float iw1 = (1-zd)*getData(weight,Xlen,Ylen,Zlen,downX,downY,downZ)+zd*getData(weight,Xlen,Ylen,Zlen,downX,downY,upZ);
	float iw2 = (1-zd)*getData(weight,Xlen,Ylen,Zlen,downX,upY,downZ)+zd*getData(weight,Xlen,Ylen,Zlen,downX,upY,upZ);
	float jw1 = (1-zd)*getData(weight,Xlen,Ylen,Zlen,upX,downY,downZ)+zd*getData(weight,Xlen,Ylen,Zlen,upX,downY,upZ);
	float jw2 = (1-zd)*getData(weight,Xlen,Ylen,Zlen,upX,upY,downZ)+zd*getData(weight,Xlen,Ylen,Zlen,upX,upY,upZ);
    
	i1 = (iw1 == 0)? 0 : i1 / iw1;
	i2 = (iw2 == 0)? 0 : i2 / iw2;
	j1 = (jw1 == 0)? 0 : j1 / jw1;
	j2 = (jw2 == 0)? 0 : j2 / jw2;

	float w1 = i1*(1-yd)*iw1 + i2 *yd*iw2;
	float w2 = j1*(1-yd)*jw1 + j2 *yd*jw2;

	float ww1 =(1-yd)*iw1 + yd*iw2;
	float ww2 =(1-yd)*jw1 + yd*jw2;

	w1 = (ww1 == 0)? 0 : w1 / ww1;
	w2 = (ww2 == 0)? 0 : w2 / ww2;

	float ret = w1*(1-xd)*ww1 + w2 * xd*ww2;

	float retw = (1-xd)*ww1 + xd*ww2;

	ret  = (retw == 0)? 0 : ret / retw;

	return ret;
}

__global__ void changeSdfData(float* local, float* dists, float* weights, float* n_x,float* n_y, float* n_z, float* olocalc, float* odists, float* oweights, float* on_x,float* on_y, float* on_z, float minX,float minY,float minZ,int Xlen,int Ylen,int Zlen, float ominX,float ominY,float ominZ, int oXlen,int oYlen,int oZlen, float devide, float thre)
{
	int idx = threadIdx.x;
	int idy = blockIdx.x;
	int idz = blockIdx.y;

	float l_coreX = minX + ((float)idx + 0.5)*devide;
	float l_coreY = minY + ((float)idy + 0.5)*devide;
	float l_coreZ = minZ + ((float)idz + 0.5)*devide;

    float coreX = local[0*4+0]*l_coreX+local[0*4+1]*l_coreY+local[0*4+2]*l_coreZ + local[0*4+3]*1;
	float coreY = local[1*4+0]*l_coreX+local[1*4+1]*l_coreY+local[1*4+2]*l_coreZ + local[1*4+3]*1;
	float coreZ = local[2*4+0]*l_coreX+local[2*4+1]*l_coreY+local[2*4+2]*l_coreZ + local[2*4+3]*1;

	float rx = olocalc[0*4+0]*coreX+olocalc[0*4+1]*coreY+olocalc[0*4+2]*coreZ + olocalc[0*4+3]*1;
	float ry = olocalc[1*4+0]*coreX+olocalc[1*4+1]*coreY+olocalc[1*4+2]*coreZ + olocalc[1*4+3]*1;
	float rz = olocalc[2*4+0]*coreX+olocalc[2*4+1]*coreY+olocalc[2*4+2]*coreZ + olocalc[2*4+3]*1;

	int downX = (rx-ominX) / devide - 0.5;
	int downY = (ry-ominY) / devide - 0.5;
	int downZ = (rz-ominZ) / devide - 0.5;

	float xd = (rx - (ominX + ((float)downX + 0.5)*devide)) / devide;
	float yd = (ry - (ominY + ((float)downY + 0.5)*devide)) / devide;
	float zd = (rz - (ominZ + ((float)downZ + 0.5)*devide)) / devide;

	int id = ((idz*Ylen)+idy)*Xlen+idx;

	weights[id] = getTriInterWeight(oweights,oXlen, oYlen, oZlen, downX, downY , downZ,  xd,  yd,  zd);
	if(weights[id] <= 0)
	{
		weights[id] = 0;
		dists[id] = 100;
		n_x[id] = 0;
		n_y[id] = 0;
		n_z[id] = 0;
	}else{
		dists[id] = getTriInter(odists,oweights,oXlen, oYlen, oZlen, downX, downY , downZ,  xd,  yd,  zd);
		//bool a;
		//a = (odists[id] == dists[id]);
		float nx = getTriInter(on_x,oweights,oXlen, oYlen, oZlen, downX, downY , downZ,  xd,  yd,  zd);
		float ny = getTriInter(on_y,oweights,oXlen, oYlen, oZlen, downX, downY , downZ,  xd,  yd,  zd);
		float nz = getTriInter(on_z,oweights,oXlen, oYlen, oZlen, downX, downY , downZ,  xd,  yd,  zd);
		float n = sqrt(nx*nx+ny*ny+nz*nz);
		n_x[id] = nx/n;
		n_y[id] = ny/n;
		n_z[id] = nz/n;


	}

}

__global__ void fusion(float* ptsx, float* ptsy, float* ptsz, float* norx, float* nory, float* norz, float* mc, float* dists, float* weights, float* n_x,float* n_y, float* n_z, float* local,int nHalfXres,int nHalfYres, float fCoeffX, float fCoeffY, float minX,float minY,float minZ,int Xlen,int Ylen,int Zlen, float devide, float epsi, float thre, int width)
{
	int idx = threadIdx.x;
	int idy = blockIdx.x;
	int idz = blockIdx.y;
	
	float l_coreX = minX + ((float)idx + 0.5)*devide;
	float l_coreY = minY + ((float)idy + 0.5)*devide;
	float l_coreZ = minZ + ((float)idz + 0.5)*devide;

	float coreX = local[0*4+0]*l_coreX+local[0*4+1]*l_coreY+local[0*4+2]*l_coreZ + local[0*4+3]*1;
	float coreY = local[1*4+0]*l_coreX+local[1*4+1]*l_coreY+local[1*4+2]*l_coreZ + local[1*4+3]*1;
	float coreZ = local[2*4+0]*l_coreX+local[2*4+1]*l_coreY+local[2*4+2]*l_coreZ + local[2*4+3]*1;

	float rx,ry,rz;
	rx = mc[0*4+0]*coreX+mc[0*4+1]*coreY+mc[0*4+2]*coreZ + mc[0*4+3]*1;
	ry = mc[1*4+0]*coreX+mc[1*4+1]*coreY+mc[1*4+2]*coreZ + mc[1*4+3]*1;
	rz = mc[2*4+0]*coreX+mc[2*4+1]*coreY+mc[2*4+2]*coreZ + mc[2*4+3]*1;

	float pxf = fCoeffX * (rx / 0.001f) / (rz / 0.001f) + nHalfXres;
    float pyf = nHalfYres - fCoeffY * (ry / 0.001f) / (rz / 0.001f);

	int px = pxf + 0.5;
	int py = pyf + 0.5;

	float dist = 100;
	if(px < 0 || px >= 640 || py < 0 || py >= 480){
	}else if(isnan(norx[py*width+px])){
	}else{
		    float ptx = ptsx[py*width+px];
			float pty = ptsy[py*width+px];
			float ptz = ptsz[py*width+px];

			float prx = mc[0*4+0]*ptx+mc[0*4+1]*pty+mc[0*4+2]*ptz + mc[0*4+3]*1;
			float pry = mc[1*4+0]*ptx+mc[1*4+1]*pty+mc[1*4+2]*ptz + mc[1*4+3]*1;
			float prz = mc[2*4+0]*ptx+mc[2*4+1]*pty+mc[2*4+2]*ptz + mc[2*4+3]*1;

			float d_p = rz-prz;
			
			//求点到平面（近似）距离
			
			float a[3];
			float b[3];
			a[0] = ptsx[py*width+px]-coreX;
			a[1] = ptsy[py*width+px]-coreY;
			a[2] = ptsz[py*width+px]-coreZ;
			b[0] = norx[py*width+px];
			b[1] = nory[py*width+px];
			b[2] = norz[py*width+px];
			float d_n = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

			//float d_s = sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
			
			//深度：
			dist = d_p;
	}
	
	float weight = 0;
	if(dist < epsi && dist > -epsi) weight = 1;
	else if(dist > thre || dist < -thre) weight = 0;
	else{
		float diff;
	    if(dist > 0) diff = dist - epsi;
	    else diff = dist + epsi;
	    weight = pow(2.7182818284590452353602874713526624977572f,-500*diff*diff);
	}
	if(weight > 0)
	{
		int id = ((idz*Ylen)+idy)*Xlen+idx;
	    float wid = weights[id];
		dists[id] = (dists[id]*wid+dist*weight)/(wid+weight);
		float nx = (n_x[id]*wid+norx[py*width+px]*weight)/(wid+weight);
		float ny = (n_y[id]*wid+nory[py*width+px]*weight)/(wid+weight);
		float nz = (n_z[id]*wid+norz[py*width+px]*weight)/(wid+weight);
		float n = sqrt(nx*nx+ny*ny+nz*nz);
		n_x[id] = nx/n;
		n_y[id] = ny/n;
		n_z[id] = nz/n;
		weights[id] = wid+weight;
	}
}

__global__ void sample(float devide, float minX,float minY,float minZ, int Xlen,int Ylen,int Zlen,
	                          float sdevide, float sminX, float sminY, float sminZ, int sXlen, int sYlen, int sZlen,
							  float* absrate,float* dists, float* weights, float* n_x,float* n_y, float* n_z, dim3 blockSize)
{
	int idx = threadIdx.x*blockSize.x+blockIdx.x;
	int idy = threadIdx.y*blockSize.y+blockIdx.y;
	int idz = threadIdx.z*blockSize.z+blockIdx.z;

	if(idx >= sXlen || idy >= sYlen || idz >= sZlen) return;

	int id = ((idz*sYlen)+idy)*sXlen+idx;

	float s_coreX = sminX + ((float)idx + 0.5)*sdevide;
	float s_coreY = sminY + ((float)idy + 0.5)*sdevide;
	float s_coreZ = sminZ + ((float)idz + 0.5)*sdevide;

	int downX = (s_coreX-minX) / devide - 0.5;
	int downY = (s_coreY-minY) / devide - 0.5;
	int downZ = (s_coreZ-minZ) / devide - 0.5;

	float xd = (s_coreX - (minX + ((float)downX + 0.5)*devide)) / devide;
	float yd = (s_coreY - (minY + ((float)downY + 0.5)*devide)) / devide;
	float zd = (s_coreZ - (minZ + ((float)downZ + 0.5)*devide)) / devide;

	float w = getTriInterWeight(weights,Xlen, Ylen, Zlen, downX, downY , downZ,  xd,  yd,  zd);
	if(w <= 0)
	{
		absrate[id] = 1/100;
	}else{
		float d = getTriInter(dists,weights,Xlen, Ylen, Zlen, downX, downY , downZ,  xd,  yd,  zd);
		absrate[id] = 1/abs(d);
	}
}


int changeSdf(float minx, float miny, float minz, int xlen,int ylen, int zlen, float* local,float* olocalc)
{

	int size = xlen*ylen*zlen;

	float* n_dists = 0;
    float* n_weights = 0;
    float* n_n_x = 0;
    float* n_n_y = 0;
    float* n_n_z = 0;
	float* n_local = 0;

	float* o_localc = 0;

	cudaError_t cudaStatus = cudaSuccess;

    cudaStatus = cudaMalloc((void**)&n_dists, size * sizeof(float));
	GO_TO_ERROR(cudaStatus,  "cudaMalloc failed!");

	cudaStatus = cudaMalloc((void**)&n_weights, size * sizeof(float));
    GO_TO_ERROR(cudaStatus,  "cudaMalloc failed!");

	cudaStatus = cudaMalloc((void**)&n_n_x, size * sizeof(float));
    GO_TO_ERROR(cudaStatus,  "cudaMalloc failed!");

	cudaStatus = cudaMalloc((void**)&n_n_y, size * sizeof(float));
    GO_TO_ERROR(cudaStatus,  "cudaMalloc failed!");

	cudaStatus = cudaMalloc((void**)&n_n_z, size * sizeof(float));
    GO_TO_ERROR(cudaStatus,  "cudaMalloc failed!");

	cudaStatus = cudaMalloc((void**)&n_local, 4*4 * sizeof(float));
    GO_TO_ERROR(cudaStatus,  "cudaMalloc failed!");

	cudaStatus = cudaMalloc((void**)&o_localc, 4*4 * sizeof(float));
    GO_TO_ERROR(cudaStatus,  "cudaMalloc failed!");

	cudaStatus = cudaMemcpy(n_local, local, 4*4 * sizeof(float), cudaMemcpyHostToDevice);
    GO_TO_ERROR(cudaStatus,  "cudaMemcpy failed!");

	cudaStatus = cudaMemcpy(o_localc, olocalc, 4*4 * sizeof(float), cudaMemcpyHostToDevice);
    GO_TO_ERROR(cudaStatus,  "cudaMemcpy failed!");

	dim3 dimBlock(ylen,zlen);
	changeSdfData<<<dimBlock, xlen>>>(n_local, n_dists, n_weights, n_n_x, n_n_y, n_n_z, o_localc, _dists, _weights, _n_x, _n_y, _n_z, minx, miny, minz, xlen, ylen, zlen, _minX, _minY, _minZ, _Xlen, _Ylen, _Zlen, _devide, _thre);

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

	_minX = minx;
	_minY = miny;
	_minZ = minz;
	_Xlen = xlen;
	_Ylen = ylen;
	_Zlen = zlen;
	
	cudaFree(_dists);
	cudaFree(_weights);
	cudaFree(_n_x);
    cudaFree(_n_y);
    cudaFree(_n_z);
	cudaFree(_local);

	_dists = n_dists;
    _weights = n_weights;
    _n_x = n_n_x;
    _n_y = n_n_y;
    _n_z =  n_n_z;
	_local = n_local;

	return cudaStatus;

	Error:
    cudaFree(n_dists);
	cudaFree(n_weights);
	cudaFree(n_n_x);
    cudaFree(n_n_y);
    cudaFree(n_n_z);
	cudaFree(n_local);
    
    return cudaStatus;
}

int getAllDepths(float ix, float iy, float devide, int xl,int yl,float* mc, float* p_dists, float* p_nx, float* p_ny, float* p_nz)
{
    float *dev_pdists = 0;

	float *dev_pnx = 0;
    float *dev_pny = 0;
    float *dev_pnz = 0;

	float *dev_mc = 0;

	cudaError_t cudaStatus = cudaSuccess;

	cudaStatus = cudaMalloc((void**)&dev_pdists, xl*yl * sizeof(float));
    GO_TO_ERROR(cudaStatus,  "cudaMalloc failed!");

	cudaStatus = cudaMalloc((void**)&dev_pnx, xl*yl * sizeof(float));
    GO_TO_ERROR(cudaStatus,  "cudaMalloc failed!");

	cudaStatus = cudaMalloc((void**)&dev_pny, xl*yl * sizeof(float));
    GO_TO_ERROR(cudaStatus,  "cudaMalloc failed!");

	cudaStatus = cudaMalloc((void**)&dev_pnz, xl*yl * sizeof(float));
    GO_TO_ERROR(cudaStatus,  "cudaMalloc failed!");

	cudaStatus = cudaMalloc((void**)&dev_mc, 4*4 * sizeof(float));
    GO_TO_ERROR(cudaStatus,  "cudaMalloc failed!");

	cudaStatus = cudaMemcpy(dev_mc, mc, 4*4 * sizeof(float), cudaMemcpyHostToDevice);
    GO_TO_ERROR(cudaStatus,  "cudaMemcpy failed!");

	getDepth<<<xl, yl>>>(ix,iy,devide, _minX,_minY,_minZ, _Xlen,_Ylen,_Zlen, xl, dev_mc, dev_pdists, dev_pnx, dev_pny, dev_pnz,_dists, _weights, _n_x, _n_y, _n_z);

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

	cudaStatus = cudaMemcpy(p_dists, dev_pdists, xl*yl * sizeof(float), cudaMemcpyDeviceToHost);
    GO_TO_ERROR(cudaStatus,  "cudaMemcpy failed!");

	cudaStatus = cudaMemcpy(p_nx, dev_pnx, xl*yl * sizeof(float), cudaMemcpyDeviceToHost);
    GO_TO_ERROR(cudaStatus,  "cudaMemcpy failed!");

	cudaStatus = cudaMemcpy(p_ny, dev_pny, xl*yl * sizeof(float), cudaMemcpyDeviceToHost);
    GO_TO_ERROR(cudaStatus,  "cudaMemcpy failed!");

	cudaStatus = cudaMemcpy(p_nz, dev_pnz, xl*yl * sizeof(float), cudaMemcpyDeviceToHost);
    GO_TO_ERROR(cudaStatus,  "cudaMemcpy failed!");

	Error:
    cudaFree(dev_pdists);
	cudaFree(dev_pnx);
    cudaFree(dev_pny);
    cudaFree(dev_pnz);
	cudaFree(dev_mc);
    
    return cudaStatus;
}

int freeSdf()
{
	
	
    cudaError_t cudaStatus  = cudaFree(_dists);
	cudaFree(_weights);
	cudaFree(_n_x);
    cudaFree(_n_y);
    cudaFree(_n_z);
	cudaFree(_local);
    
    return cudaStatus;
}

int getSdf(float* dists ,float* weights,float* n_x,float* n_y,float* n_z,int size)
{
	
	cudaError_t cudaStatus = cudaSuccess;
	cudaStatus = cudaMemcpy(dists, _dists, size * sizeof(float), cudaMemcpyDeviceToHost);
    GO_TO_ERROR(cudaStatus,  "cudaMemcpy failed!");

	cudaStatus = cudaMemcpy(weights, _weights, size * sizeof(float), cudaMemcpyDeviceToHost);
    GO_TO_ERROR(cudaStatus,  "cudaMemcpy failed!");

	cudaStatus = cudaMemcpy(n_x, _n_x, size * sizeof(float), cudaMemcpyDeviceToHost);
    GO_TO_ERROR(cudaStatus,  "cudaMemcpy failed!");

	cudaStatus = cudaMemcpy(n_y, _n_y, size * sizeof(float), cudaMemcpyDeviceToHost);
    GO_TO_ERROR(cudaStatus,  "cudaMemcpy failed!");

	cudaStatus = cudaMemcpy(n_z, _n_z, size * sizeof(float), cudaMemcpyDeviceToHost);
    GO_TO_ERROR(cudaStatus,  "cudaMemcpy failed!");

	Error:
    cudaFree(_dists);
	cudaFree(_weights);
	cudaFree(_n_x);
    cudaFree(_n_y);
    cudaFree(_n_z);
    
    return cudaStatus;
}

int updateSdf(float* ptsx, float* ptsy, float* ptsz, float* norx, float* nory, float* norz,int size, float* mc,int width,int xres,int yres,float coeffx,float coeffy)
{
	float *dev_px = 0;
    float *dev_py = 0;
    float *dev_pz = 0;

	float *dev_nx = 0;
    float *dev_ny = 0;
    float *dev_nz = 0;

	float *dev_mc = 0;

	cudaError_t cudaStatus = cudaSuccess;

	cudaStatus = cudaMalloc((void**)&dev_px, size * sizeof(float));
    GO_TO_ERROR(cudaStatus,  "cudaMalloc failed!");

	cudaStatus = cudaMalloc((void**)&dev_py, size * sizeof(float));
    GO_TO_ERROR(cudaStatus,  "cudaMalloc failed!");

	cudaStatus = cudaMalloc((void**)&dev_pz, size * sizeof(float));
    GO_TO_ERROR(cudaStatus,  "cudaMalloc failed!");

	cudaStatus = cudaMalloc((void**)&dev_nx, size * sizeof(float));
    GO_TO_ERROR(cudaStatus,  "cudaMalloc failed!");

	cudaStatus = cudaMalloc((void**)&dev_ny, size * sizeof(float));
    GO_TO_ERROR(cudaStatus,  "cudaMalloc failed!");

	cudaStatus = cudaMalloc((void**)&dev_nz, size * sizeof(float));
    GO_TO_ERROR(cudaStatus,  "cudaMalloc failed!");

	cudaStatus = cudaMalloc((void**)&dev_mc, 4*4 * sizeof(float));
    GO_TO_ERROR(cudaStatus,  "cudaMalloc failed!");

	cudaStatus = cudaMemcpy(dev_px, ptsx, size * sizeof(float), cudaMemcpyHostToDevice);
    GO_TO_ERROR(cudaStatus,  "cudaMemcpy failed!");

	cudaStatus = cudaMemcpy(dev_py, ptsy, size * sizeof(float), cudaMemcpyHostToDevice);
    GO_TO_ERROR(cudaStatus,  "cudaMemcpy failed!");

	cudaStatus = cudaMemcpy(dev_pz, ptsz, size * sizeof(float), cudaMemcpyHostToDevice);
    GO_TO_ERROR(cudaStatus,  "cudaMemcpy failed!");

	cudaStatus = cudaMemcpy(dev_nx, norx, size * sizeof(float), cudaMemcpyHostToDevice);
    GO_TO_ERROR(cudaStatus,  "cudaMemcpy failed!");

	cudaStatus = cudaMemcpy(dev_ny, nory, size * sizeof(float), cudaMemcpyHostToDevice);
    GO_TO_ERROR(cudaStatus,  "cudaMemcpy failed!");

	cudaStatus = cudaMemcpy(dev_nz, norz, size * sizeof(float), cudaMemcpyHostToDevice);
    GO_TO_ERROR(cudaStatus,  "cudaMemcpy failed!");

	cudaStatus = cudaMemcpy(dev_mc, mc, 4*4 * sizeof(float), cudaMemcpyHostToDevice);
    GO_TO_ERROR(cudaStatus,  "cudaMemcpy failed!");

	dim3 dimBlock(_Ylen,_Zlen);
    fusion<<<dimBlock, _Xlen>>>(dev_px, dev_py, dev_pz, dev_nx, dev_ny, dev_nz, dev_mc, _dists, _weights, _n_x, _n_y, _n_z, _local ,xres,yres,coeffx,coeffy,_minX,_minY,_minZ,_Xlen,_Ylen,_Zlen,_devide, _epsi, _thre, width);

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
Error:
    cudaFree(dev_px);
	cudaFree(dev_py);
	cudaFree(dev_pz);
    cudaFree(dev_nx);
	cudaFree(dev_ny);
	cudaFree(dev_nz);
    cudaFree(dev_mc);

    return cudaStatus;
}

int initialSdf(float minx, float miny, float minz, int xlen,int ylen, int zlen, float devide, float epsi, float thre,float* dists, float* weights,float* nx,float* ny,float* nz,float* local)
{
	_minX = minx;
	_minY = miny;
	_minZ = minz;
	_Xlen = xlen;
	_Ylen = ylen;
	_Zlen = zlen;
	_devide = devide;
	_epsi = epsi;
	_thre = thre;

	int size = _Xlen*_Ylen*_Zlen;

	_dists = 0;
    _weights = 0;
    _n_x = 0;
    _n_y = 0;
    _n_z = 0;
	_local = 0;

	cudaError_t cudaStatus = cudaSuccess;
	cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&_dists, size * sizeof(float));
    GO_TO_ERROR(cudaStatus,  "cudaMalloc failed!");

	cudaStatus = cudaMalloc((void**)&_weights, size * sizeof(float));
    GO_TO_ERROR(cudaStatus,  "cudaMalloc failed!");

	cudaStatus = cudaMalloc((void**)&_n_x, size * sizeof(float));
    GO_TO_ERROR(cudaStatus,  "cudaMalloc failed!");

	cudaStatus = cudaMalloc((void**)&_n_y, size * sizeof(float));
    GO_TO_ERROR(cudaStatus,  "cudaMalloc failed!");

	cudaStatus = cudaMalloc((void**)&_n_z, size * sizeof(float));
    GO_TO_ERROR(cudaStatus,  "cudaMalloc failed!");

	cudaStatus = cudaMalloc((void**)&_local, 4*4 * sizeof(float));
    GO_TO_ERROR(cudaStatus,  "cudaMalloc failed!");

	cudaStatus = cudaMemcpy(_dists, dists, size * sizeof(float), cudaMemcpyHostToDevice);
    GO_TO_ERROR(cudaStatus,  "cudaMemcpy failed!");

	cudaStatus = cudaMemcpy(_weights, weights, size * sizeof(float), cudaMemcpyHostToDevice);
    GO_TO_ERROR(cudaStatus,  "cudaMemcpy failed!");

	cudaStatus = cudaMemcpy(_n_x, nx, size * sizeof(float), cudaMemcpyHostToDevice);
    GO_TO_ERROR(cudaStatus,  "cudaMemcpy failed!");

	cudaStatus = cudaMemcpy(_n_y, ny, size * sizeof(float), cudaMemcpyHostToDevice);
    GO_TO_ERROR(cudaStatus,  "cudaMemcpy failed!");

	cudaStatus = cudaMemcpy(_n_z, nz, size * sizeof(float), cudaMemcpyHostToDevice);
    GO_TO_ERROR(cudaStatus,  "cudaMemcpy failed!");

	cudaStatus = cudaMemcpy(_local, local, 4*4 * sizeof(float), cudaMemcpyHostToDevice);
    GO_TO_ERROR(cudaStatus,  "cudaMemcpy failed!");

	return cudaStatus;

	Error:
    cudaFree(_dists);
	cudaFree(_weights);
	cudaFree(_n_x);
    cudaFree(_n_y);
    cudaFree(_n_z);
    
    return cudaStatus;
}

int sampleSdfData(float sdevide, float sminX, float sminY, float sminZ, int sXlen, int sYlen, int sZlen, float* absrate)
{
	int size = sXlen*sYlen*sZlen;

	float *dev_absrate = 0;

	cudaError_t cudaStatus = cudaSuccess;
	cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_absrate, size * sizeof(float));
    GO_TO_ERROR(cudaStatus,  "cudaMalloc failed!");

	dim3 dimBlock((sXlen + 7)/8,(sYlen+7)/8, (sZlen+7)/8);
	dim3 dimThread(8,8,8);
	sample<<<dimBlock, dimThread>>>(_devide, _minX, _minY, _minZ, _Xlen, _Ylen, _Zlen,
	                          sdevide, sminX, sminY, sminZ, sXlen, sYlen, sZlen,
							  dev_absrate, _dists, _weights, _n_x, _n_y, _n_z, dimBlock);

	cudaStatus = cudaMemcpy(absrate, dev_absrate, size * sizeof(float), cudaMemcpyDeviceToHost);
	GO_TO_ERROR(cudaStatus, "cudaMemcpy failed!")

Error:

	cudaFree(dev_absrate);

	return cudaStatus;
}
