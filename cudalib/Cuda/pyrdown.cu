/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2011, Willow Garage, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include "internal.h"
#include "vector_math.hpp"
#include "containers/safe_call.hpp"

__global__ void pyrDownGaussKernel (const PtrStepSz<unsigned short> src, PtrStepSz<unsigned short> dst, float sigma_color)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dst.cols || y >= dst.rows)
        return;

    const int D = 5;

    int center = src.ptr (2 * y)[2 * x];

    int x_mi = max(0, 2*x - D/2) - 2*x;
    int y_mi = max(0, 2*y - D/2) - 2*y;

    int x_ma = min(src.cols, 2*x -D/2+D) - 2*x;
    int y_ma = min(src.rows, 2*y -D/2+D) - 2*y;

    float sum = 0;
    float wall = 0;

    float weights[] = {0.375f, 0.25f, 0.0625f} ;

    for(int yi = y_mi; yi < y_ma; ++yi)
        for(int xi = x_mi; xi < x_ma; ++xi)
        {
            int val = src.ptr (2*y + yi)[2*x + xi];

            if (abs (val - center) < 3 * sigma_color)
            {
                sum += val * weights[abs(xi)] * weights[abs(yi)];
                wall += weights[abs(xi)] * weights[abs(yi)];
            }
        }


    dst.ptr (y)[x] = static_cast<int>(sum /wall);
}

void pyrDown(const DeviceArray2D<unsigned short> & src, DeviceArray2D<unsigned short> & dst)
{
    dst.create (src.rows () / 2, src.cols () / 2);

    dim3 block (32, 8);
    dim3 grid (divUp (dst.cols (), block.x), divUp (dst.rows (), block.y));

    const float sigma_color = 30;

    pyrDownGaussKernel<<<grid, block>>>(src, dst, sigma_color);
    cudaSafeCall ( cudaGetLastError () );
};

__global__ void computeVmapKernel(const PtrStepSz<unsigned short> depth, PtrStep<float> vmap, float fx_inv, float fy_inv, float cx, float cy, float depthCutoff, float depthFactor)
{
    int u = threadIdx.x + blockIdx.x * blockDim.x;
    int v = threadIdx.y + blockIdx.y * blockDim.y;

    if(u < depth.cols && v < depth.rows)
    {
        float z = depth.ptr (v)[u] / depthFactor; // load and convert: mm -> meters

        if(z != 0 && z < depthCutoff)
        {
            float vx = z * (u - cx) * fx_inv;
            float vy = z * (v - cy) * fy_inv;
            float vz = z;

            vmap.ptr (v                 )[u] = vx;
            vmap.ptr (v + depth.rows    )[u] = vy;
            vmap.ptr (v + depth.rows * 2)[u] = vz;
        }
        else
        {
            vmap.ptr (v)[u] = __int_as_float(0x7fffffff); /*CUDART_NAN_F*/
        }
    }
}

void createVMap(const Intr& intr, const DeviceArray2D<unsigned short> & depth, DeviceArray2D<float> & vmap, const float depthCutoff, const float depthFactor)
{
    vmap.create (depth.rows () * 3, depth.cols ());

    dim3 block (32, 8);
    dim3 grid (1, 1, 1);
    grid.x = divUp (depth.cols (), block.x);
    grid.y = divUp (depth.rows (), block.y);

    float fx = intr.fx, cx = intr.cx;
    float fy = intr.fy, cy = intr.cy;

    computeVmapKernel<<<grid, block>>>(depth, vmap, 1.f / fx, 1.f / fy, cx, cy, depthCutoff, depthFactor);
    cudaSafeCall (cudaGetLastError ());
}

__global__ void computeNmapKernel(int rows, int cols, const PtrStep<float> vmap, PtrStep<float> nmap)
{
    int u = threadIdx.x + blockIdx.x * blockDim.x;
    int v = threadIdx.y + blockIdx.y * blockDim.y;

    if (u >= cols || v >= rows)
        return;

    if (u == cols - 1 || v == rows - 1)
    {
        nmap.ptr (v)[u] = __int_as_float(0x7fffffff); /*CUDART_NAN_F*/
        return;
    }

    float3 v00, v01, v10;
    v00.x = vmap.ptr (v  )[u];
    v01.x = vmap.ptr (v  )[u + 1];
    v10.x = vmap.ptr (v + 1)[u];

    if (!isnan (v00.x) && !isnan (v01.x) && !isnan (v10.x))
    {
        v00.y = vmap.ptr (v + rows)[u];
        v01.y = vmap.ptr (v + rows)[u + 1];
        v10.y = vmap.ptr (v + 1 + rows)[u];

        v00.z = vmap.ptr (v + 2 * rows)[u];
        v01.z = vmap.ptr (v + 2 * rows)[u + 1];
        v10.z = vmap.ptr (v + 1 + 2 * rows)[u];

        float3 r = normalized (cross (v01 - v00, v10 - v00));

        nmap.ptr (v       )[u] = r.x;
        nmap.ptr (v + rows)[u] = r.y;
        nmap.ptr (v + 2 * rows)[u] = r.z;
    }
    else
        nmap.ptr (v)[u] = __int_as_float(0x7fffffff); /*CUDART_NAN_F*/
}

__global__ void computeNmapKernel2(int rows, int cols, const PtrStep<float> vmap, PtrStep<float> nmap)
{
	int u = threadIdx.x + blockIdx.x * blockDim.x;
	int v = threadIdx.y + blockIdx.y * blockDim.y;

	if (u >= cols || v >= rows)
		return;

	if (u == 0 || v == 0 || u == cols - 1 || v == rows - 1)
	{
		nmap.ptr(v)[u] = __int_as_float(0x7fffffff); /*CUDART_NAN_F*/
		return;
	}

	float3 v00, v01, v10, v0f1, vf10;
	v00.x = vmap.ptr(v)[u];
	v01.x = vmap.ptr(v)[u + 1];
	v10.x = vmap.ptr(v + 1)[u];
	v0f1.x = vmap.ptr(v)[u - 1];
	vf10.x = vmap.ptr(v - 1)[u];

	if (!isnan(v00.x) && !isnan(v01.x) && !isnan(v10.x) && !isnan(v0f1.x) && !isnan(vf10.x))
	{
		v00.y = vmap.ptr(v + rows)[u];
		v01.y = vmap.ptr(v + rows)[u + 1];
		v10.y = vmap.ptr(v + 1 + rows)[u];
		v0f1.y = vmap.ptr(v + rows)[u - 1];
		vf10.y = vmap.ptr(v - 1 + rows)[u];

		v00.z = vmap.ptr(v + 2 * rows)[u];
		v01.z = vmap.ptr(v + 2 * rows)[u + 1];
		v10.z = vmap.ptr(v + 1 + 2 * rows)[u];
		v0f1.z = vmap.ptr(v + 2 * rows)[u - 1];
		vf10.z = vmap.ptr(v - 1 + 2 * rows)[u];

		float3 n0 = normalized(cross(v01 - v00, v10 - v00));
		float3 n1 = normalized(cross(v10 - v00, v0f1 - v00));
		float3 n2 = normalized(cross(v0f1 - v00, vf10 - v00));
		float3 n3 = normalized(cross(vf10 - v00, v01 - v00));
		float3 n;
		n.x = (n0.x + n1.x + n2.x + n3.x) / 4.0;
		n.y = (n0.y + n1.y + n2.y + n3.y) / 4.0;
		n.z = (n0.z + n1.z + n2.z + n3.z) / 4.0;

		nmap.ptr(v)[u] = n.x;
		nmap.ptr(v + rows)[u] = n.y;
		nmap.ptr(v + 2 * rows)[u] = n.z;
	}
	else
		nmap.ptr(v)[u] = __int_as_float(0x7fffffff); /*CUDART_NAN_F*/
}

void createNMap(const DeviceArray2D<float>& vmap, DeviceArray2D<float>& nmap)
{
    nmap.create (vmap.rows (), vmap.cols ());

    int rows = vmap.rows () / 3;
    int cols = vmap.cols ();

    dim3 block (32, 8);
    dim3 grid (1, 1, 1);
    grid.x = divUp (cols, block.x);
    grid.y = divUp (rows, block.y);

	computeNmapKernel2<<<grid, block>>>(rows, cols, vmap, nmap);
    cudaSafeCall (cudaGetLastError ());
}

__global__ void computePmapKernel(int rows, int cols, const PtrStep<float> vmap, PtrStep<float> pmap)
{
	int u = threadIdx.x + blockIdx.x * blockDim.x;
	int v = threadIdx.y + blockIdx.y * blockDim.y;

	if (u >= cols || v >= rows)
		return;

	if (u == 0 || v == 0 || u == cols - 1 || v == rows - 1)
	{
		pmap.ptr(v)[u] = __int_as_float(0x7fffffff); /*CUDART_NAN_F*/
		return;
	}

	float3 v00, v01, v10, v0f1, vf10;
	v00.x = vmap.ptr(v)[u];
	v01.x = vmap.ptr(v)[u + 1];
	v10.x = vmap.ptr(v + 1)[u];
	v0f1.x = vmap.ptr(v)[u - 1];
	vf10.x = vmap.ptr(v - 1)[u];

	if (!isnan(v00.x) && !isnan(v01.x) && !isnan(v10.x) && !isnan(v0f1.x) && !isnan(vf10.x))
	{
		v00.y = vmap.ptr(v + rows)[u];
		v01.y = vmap.ptr(v + rows)[u + 1];
		v10.y = vmap.ptr(v + 1 + rows)[u];
		v0f1.y = vmap.ptr(v + rows)[u - 1];
		vf10.y = vmap.ptr(v - 1 + rows)[u];

		v00.z = vmap.ptr(v + 2 * rows)[u];
		v01.z = vmap.ptr(v + 2 * rows)[u + 1];
		v10.z = vmap.ptr(v + 1 + 2 * rows)[u];
		v0f1.z = vmap.ptr(v + 2 * rows)[u - 1];
		vf10.z = vmap.ptr(v - 1 + 2 * rows)[u];

		float3 n0 = normalized(cross(v01 - v00, v10 - v00));
		float3 n1 = normalized(cross(v10 - v00, v0f1 - v00));
		float3 n2 = normalized(cross(v0f1 - v00, vf10 - v00));
		float3 n3 = normalized(cross(vf10 - v00, v01 - v00));
		float3 n = n0 + n1 + n2 + n3;
		float a = norm(n0) + norm(n1) + norm(n2) + norm(n3);
		float b = norm(n);
		pmap.ptr(v)[u] = a - b;
	}
	else
		pmap.ptr(v)[u] = __int_as_float(0x7fffffff); /*CUDART_NAN_F*/
}

void createPMap(const DeviceArray2D<float>& vmap, DeviceArray2D<float>& pmap)
{
	int rows = vmap.rows() / 3;
	int cols = vmap.cols();

	pmap.create(rows, cols);

	dim3 block(32, 8);
	dim3 grid(1, 1, 1);
	grid.x = divUp(cols, block.x);
	grid.y = divUp(rows, block.y);

	computePmapKernel<<<grid, block>>>(rows, cols, vmap, pmap);
	cudaSafeCall(cudaGetLastError());
}

__global__ void computeNmapAndPmapKernel(int rows, int cols, const PtrStep<float> vmap, PtrStep<float> nmap, PtrStep<float> pmap)
{
	int u = threadIdx.x + blockIdx.x * blockDim.x;
	int v = threadIdx.y + blockIdx.y * blockDim.y;

	if (u >= cols || v >= rows)
		return;

	if (u == 0 || v == 0 || u == cols - 1 || v == rows - 1)
	{
		nmap.ptr(v)[u] = __int_as_float(0x7fffffff); /*CUDART_NAN_F*/
		return;
	}

	float3 v00, v01, v10, v0f1, vf10;
	v00.x = vmap.ptr(v)[u];
	v01.x = vmap.ptr(v)[u + 1];
	v10.x = vmap.ptr(v + 1)[u];
	v0f1.x = vmap.ptr(v)[u - 1];
	vf10.x = vmap.ptr(v - 1)[u];

	if (!isnan(v00.x) && !isnan(v01.x) && !isnan(v10.x) && !isnan(v0f1.x) && !isnan(vf10.x))
	{
		v00.y = vmap.ptr(v + rows)[u];
		v01.y = vmap.ptr(v + rows)[u + 1];
		v10.y = vmap.ptr(v + 1 + rows)[u];
		v0f1.y = vmap.ptr(v + rows)[u - 1];
		vf10.y = vmap.ptr(v - 1 + rows)[u];

		v00.z = vmap.ptr(v + 2 * rows)[u];
		v01.z = vmap.ptr(v + 2 * rows)[u + 1];
		v10.z = vmap.ptr(v + 1 + 2 * rows)[u];
		v0f1.z = vmap.ptr(v + 2 * rows)[u - 1];
		vf10.z = vmap.ptr(v - 1 + 2 * rows)[u];

		float3 n0 = normalized(cross(v01 - v00, v10 - v00));
		float3 n1 = normalized(cross(v10 - v00, v0f1 - v00));
		float3 n2 = normalized(cross(v0f1 - v00, vf10 - v00));
		float3 n3 = normalized(cross(vf10 - v00, v01 - v00));
		float3 n = n0 + n1 + n2 + n3;

		pmap.ptr(v)[u] = norm(n0) + norm(n1) + norm(n2) + norm(n3) - norm(n);
		n = normalized(n);
		nmap.ptr(v)[u] = n.x;
		nmap.ptr(v + rows)[u] = n.y;
		nmap.ptr(v + 2 * rows)[u] = n.z;
	}
	else
	{
		pmap.ptr(v)[u] = __int_as_float(0x7fffffff);
		nmap.ptr(v)[u] = __int_as_float(0x7fffffff); /*CUDART_NAN_F*/
	}
}

void createNMapAndPMap(const DeviceArray2D<float>& vmap, DeviceArray2D<float>& nmap, DeviceArray2D<float>& pmap)
{
	int rows = vmap.rows() / 3;
	int cols = vmap.cols();

	nmap.create(vmap.rows(), vmap.cols());
	pmap.create(rows, cols);

	dim3 block(32, 8);
	dim3 grid(1, 1, 1);
	grid.x = divUp(cols, block.x);
	grid.y = divUp(rows, block.y);

	computeNmapAndPmapKernel<<<grid, block>>>(rows, cols, vmap, nmap, pmap);
	cudaSafeCall(cudaGetLastError());
}

__global__ void computePlanemapKernel(int rows, int cols, const PtrStep<float> vmap, const PtrStep<float> nmap, const float4 plane, const float max_dist,
	PtrStep<bool> planemap)
{
	int u = threadIdx.x + blockIdx.x * blockDim.x;
	int v = threadIdx.y + blockIdx.y * blockDim.y;

	if (u >= cols || v >= rows)
		return;

	bool is_true = planemap.ptr(v)[u];
	if (is_true)
		return;

	float3 v_curr, n_curr;
	v_curr.x = vmap.ptr(v)[u];
	n_curr.x = nmap.ptr(v)[u];

	if (!isnan(v_curr.x) && !isnan(n_curr.x))
	{
		v_curr.y = vmap.ptr(v + rows)[u];
		v_curr.z = vmap.ptr(v + 2 * rows)[u];

		n_curr.y = nmap.ptr(v + rows)[u];
		n_curr.z = nmap.ptr(v + 2 * rows)[u];
		
		float3 plane_n;
		plane_n.x = plane.x;
		plane_n.y = plane.y;
		plane_n.z = plane.z;

		float dist = abs(plane.x * v_curr.x + plane.y * v_curr.y + plane.z * v_curr.z + plane.w);
		float n_norm = acos(dot(plane_n, n_curr)) * 180 / 3.141592;
		planemap.ptr(v)[u] = dist < max_dist && n_norm < 20;
	}
	else
	{
		planemap.ptr(v)[u] = false;
	}
}

void createPlaneMap(const DeviceArray2D<float>& vmap, const DeviceArray2D<float>& nmap, const float4 plane, const float max_dist, DeviceArray2D<bool>& planemap)
{
	int rows = vmap.rows() / 3;
	int cols = vmap.cols();

	planemap.create(rows, cols);

	dim3 block(32, 8);
	dim3 grid(1, 1, 1);
	grid.x = divUp(cols, block.x);
	grid.y = divUp(rows, block.y);

	computePlanemapKernel <<<grid, block >>>(rows, cols, vmap, nmap, plane, max_dist, planemap);
	cudaSafeCall(cudaGetLastError());
}

__global__ void tranformMapsKernel(int rows, int cols, const PtrStep<float> vmap_src, const PtrStep<float> nmap_src,
                                   const Mat33 Rmat, const float3 tvec, PtrStepSz<float> vmap_dst, PtrStep<float> nmap_dst)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < cols && y < rows)
    {
        //vertexes
        float3 vsrc, vdst = make_float3 (__int_as_float(0x7fffffff), __int_as_float(0x7fffffff), __int_as_float(0x7fffffff));
        vsrc.x = vmap_src.ptr (y)[x];

        if (!isnan (vsrc.x))
        {
            vsrc.y = vmap_src.ptr (y + rows)[x];
            vsrc.z = vmap_src.ptr (y + 2 * rows)[x];

            vdst = Rmat * vsrc + tvec;

            vmap_dst.ptr (y + rows)[x] = vdst.y;
            vmap_dst.ptr (y + 2 * rows)[x] = vdst.z;
        }

        vmap_dst.ptr (y)[x] = vdst.x;

        //normals
        float3 nsrc, ndst = make_float3 (__int_as_float(0x7fffffff), __int_as_float(0x7fffffff), __int_as_float(0x7fffffff));
        nsrc.x = nmap_src.ptr (y)[x];

        if (!isnan (nsrc.x))
        {
            nsrc.y = nmap_src.ptr (y + rows)[x];
            nsrc.z = nmap_src.ptr (y + 2 * rows)[x];

            ndst = Rmat * nsrc;

            nmap_dst.ptr (y + rows)[x] = ndst.y;
            nmap_dst.ptr (y + 2 * rows)[x] = ndst.z;
        }

        nmap_dst.ptr (y)[x] = ndst.x;
    }
}

void tranformMaps(const DeviceArray2D<float>& vmap_src,
                  const DeviceArray2D<float>& nmap_src,
                  const Mat33& Rmat, const float3& tvec,
                  DeviceArray2D<float>& vmap_dst, DeviceArray2D<float>& nmap_dst)
{
    int cols = vmap_src.cols();
    int rows = vmap_src.rows() / 3;

    vmap_dst.create(rows * 3, cols);
    nmap_dst.create(rows * 3, cols);

    dim3 block(32, 8);
    dim3 grid(1, 1, 1);
    grid.x = divUp(cols, block.x);
    grid.y = divUp(rows, block.y);

    tranformMapsKernel<<<grid, block>>>(rows, cols, vmap_src, nmap_src, Rmat, tvec, vmap_dst, nmap_dst);
    cudaSafeCall(cudaGetLastError());
}

__global__ void copyMapsKernel(int rows, int cols, const float * vmap_src, const float * nmap_src,
                               PtrStepSz<float> vmap_dst, PtrStep<float> nmap_dst)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < cols && y < rows)
    {
        //vertexes
        float3 vsrc, vdst = make_float3 (__int_as_float(0x7fffffff), __int_as_float(0x7fffffff), __int_as_float(0x7fffffff));

        vsrc.x = vmap_src[y * cols * 4 + (x * 4) + 0];
        vsrc.y = vmap_src[y * cols * 4 + (x * 4) + 1];
        vsrc.z = vmap_src[y * cols * 4 + (x * 4) + 2];

        if(!(vsrc.z == 0))
        {
            vdst = vsrc;
        }

        vmap_dst.ptr (y)[x] = vdst.x;
        vmap_dst.ptr (y + rows)[x] = vdst.y;
        vmap_dst.ptr (y + 2 * rows)[x] = vdst.z;

        //normals
        float3 nsrc, ndst = make_float3 (__int_as_float(0x7fffffff), __int_as_float(0x7fffffff), __int_as_float(0x7fffffff));

        nsrc.x = nmap_src[y * cols * 4 + (x * 4) + 0];
        nsrc.y = nmap_src[y * cols * 4 + (x * 4) + 1];
        nsrc.z = nmap_src[y * cols * 4 + (x * 4) + 2];

        if(!(vsrc.z == 0))
        {
            ndst = nsrc;
        }

        nmap_dst.ptr (y)[x] = ndst.x;
        nmap_dst.ptr (y + rows)[x] = ndst.y;
        nmap_dst.ptr (y + 2 * rows)[x] = ndst.z;
    }
}

void copyMaps(const DeviceArray<float>& vmap_src,
              const DeviceArray<float>& nmap_src,
              DeviceArray2D<float>& vmap_dst,
              DeviceArray2D<float>& nmap_dst)
{
    int cols = vmap_dst.cols();
    int rows = vmap_dst.rows() / 3;

    vmap_dst.create(rows * 3, cols);
    nmap_dst.create(rows * 3, cols);

    dim3 block(32, 8);
    dim3 grid(1, 1, 1);
    grid.x = divUp(cols, block.x);
    grid.y = divUp(rows, block.y);

    copyMapsKernel<<<grid, block>>>(rows, cols, vmap_src, nmap_src, vmap_dst, nmap_dst);
    cudaSafeCall(cudaGetLastError());
}

__global__ void rearrangeMapKernel(int rows, int cols,
	const PtrStep<float> src, PtrStep<float> dst)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= cols || y >= rows)
		return;

	float3 n;
	n.x = src.ptr(y)[x];
	if (!isnan(n.x))
	{
		n.y = src.ptr(y + rows)[x];
		n.z = src.ptr(y + 2 * rows)[x];

		dst.ptr(y)[x * 3] = n.x;
		dst.ptr(y)[x * 3 + 1] = n.y;
		dst.ptr(y)[x * 3 + 2] = n.z;
	}
	else
	{
		dst.ptr(y)[x * 3] = 0;
		dst.ptr(y)[x * 3 + 1] = 0;
		dst.ptr(y)[x * 3 + 2] = 0;
	}
}

void rearrangeMap(const DeviceArray2D<float>& src,
	              DeviceArray2D<float>& dst)
{
	int cols = src.cols();
	int rows = src.rows() / 3;

	dst.create(rows, cols * 3);
	dim3 block(32, 8);
	dim3 grid(1, 1, 1);
	grid.x = divUp(cols, block.x);
	grid.y = divUp(rows, block.y);

	rearrangeMapKernel<<<grid, block>>>(rows, cols, src, dst);
	cudaSafeCall(cudaGetLastError());
}

__global__ void pyrDownKernelGaussF(const PtrStepSz<float> src, PtrStepSz<float> dst, float * gaussKernel)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dst.cols || y >= dst.rows)
        return;

    const int D = 5;

    float center = src.ptr (2 * y)[2 * x];

    int tx = min (2 * x - D / 2 + D, src.cols - 1);
    int ty = min (2 * y - D / 2 + D, src.rows - 1);
    int cy = max (0, 2 * y - D / 2);

    float sum = 0;
    int count = 0;

    for (; cy < ty; ++cy)
    {
        for (int cx = max (0, 2 * x - D / 2); cx < tx; ++cx)
        {
            if(!isnan(src.ptr (cy)[cx]))
            {
                sum += src.ptr (cy)[cx] * gaussKernel[(ty - cy - 1) * 5 + (tx - cx - 1)];
                count += gaussKernel[(ty - cy - 1) * 5 + (tx - cx - 1)];
            }
        }
    }
    dst.ptr (y)[x] = (float)(sum / (float)count);
}

template<bool normalize>
__global__ void resizeMapKernel(int drows, int dcols, int srows, const PtrStep<float> input, PtrStep<float> output)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= dcols || y >= drows)
        return;

    const float qnan = __int_as_float(0x7fffffff);

    int xs = x * 2;
    int ys = y * 2;

    float x00 = input.ptr (ys + 0)[xs + 0];
    float x01 = input.ptr (ys + 0)[xs + 1];
    float x10 = input.ptr (ys + 1)[xs + 0];
    float x11 = input.ptr (ys + 1)[xs + 1];

    if (isnan (x00) || isnan (x01) || isnan (x10) || isnan (x11))
    {
        output.ptr (y)[x] = qnan;
        return;
    }
    else
    {
        float3 n;

        n.x = (x00 + x01 + x10 + x11) / 4;

        float y00 = input.ptr (ys + srows + 0)[xs + 0];
        float y01 = input.ptr (ys + srows + 0)[xs + 1];
        float y10 = input.ptr (ys + srows + 1)[xs + 0];
        float y11 = input.ptr (ys + srows + 1)[xs + 1];

        n.y = (y00 + y01 + y10 + y11) / 4;

        float z00 = input.ptr (ys + 2 * srows + 0)[xs + 0];
        float z01 = input.ptr (ys + 2 * srows + 0)[xs + 1];
        float z10 = input.ptr (ys + 2 * srows + 1)[xs + 0];
        float z11 = input.ptr (ys + 2 * srows + 1)[xs + 1];

        n.z = (z00 + z01 + z10 + z11) / 4;

        if (normalize)
            n = normalized (n);

        output.ptr (y        )[x] = n.x;
        output.ptr (y + drows)[x] = n.y;
        output.ptr (y + 2 * drows)[x] = n.z;
    }
}

template<bool normalize>
void resizeMap(const DeviceArray2D<float>& input, DeviceArray2D<float>& output)
{
    int in_cols = input.cols ();
    int in_rows = input.rows () / 3;

    int out_cols = in_cols / 2;
    int out_rows = in_rows / 2;

    output.create (out_rows * 3, out_cols);

    dim3 block (32, 8);
    dim3 grid (divUp (out_cols, block.x), divUp (out_rows, block.y));
    resizeMapKernel<normalize><< < grid, block>>>(out_rows, out_cols, in_rows, input, output);
    cudaSafeCall ( cudaGetLastError () );
    cudaSafeCall (cudaDeviceSynchronize ());
}

void resizeVMap(const DeviceArray2D<float>& input, DeviceArray2D<float>& output)
{
    resizeMap<false>(input, output);
}

void resizeNMap(const DeviceArray2D<float>& input, DeviceArray2D<float>& output)
{
    resizeMap<true>(input, output);
}
