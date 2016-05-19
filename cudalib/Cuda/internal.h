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

#ifndef INTERNAL_HPP_
#define INTERNAL_HPP_

#include "containers/device_array.hpp"
#include <vector_types.h>
#include <cuda_runtime_api.h>

#if __CUDA_ARCH__ < 300
#define MAX_THREADS 512
#else
#define MAX_THREADS 1024
#endif

/** \brief Camera intrinsics structure
  */
struct Intr
{
    float fx, fy, cx, cy;
    Intr () : fx(0), fy(0), cx(0), cy(0) {}
    Intr (float fx_, float fy_, float cx_, float cy_) : fx(fx_), fy(fy_), cx(cx_), cy(cy_) {}

    Intr operator()(int level_index) const
    {
        int div = 1 << level_index;
        return (Intr (fx / div, fy / div, cx / div, cy / div));
    }
};

/** \brief 3x3 Matrix for device code
  */
struct Mat33
{
    float3 data[3];
};

struct jtjjtr
{
    //27 floats for each product (27)
    float aa, ab, ac, ad, ae, af, ag,
              bb, bc, bd, be, bf, bg,
                  cc, cd, ce, cf, cg,
                      dd, de, df, dg,
                          ee, ef, eg,
                              ff, fg;

    //Extra data needed (29)
    float residual, inliers;

    __device__ inline void add(const jtjjtr & a)
    {
        aa += a.aa;
        ab += a.ab;
        ac += a.ac;
        ad += a.ad;
        ae += a.ae;
        af += a.af;
        ag += a.ag;

        bb += a.bb;
        bc += a.bc;
        bd += a.bd;
        be += a.be;
        bf += a.bf;
        bg += a.bg;

        cc += a.cc;
        cd += a.cd;
        ce += a.ce;
        cf += a.cf;
        cg += a.cg;

        dd += a.dd;
        de += a.de;
        df += a.df;
        dg += a.dg;

        ee += a.ee;
        ef += a.ef;
        eg += a.eg;

        ff += a.ff;
        fg += a.fg;

        residual += a.residual;
        inliers += a.inliers;
    }
};

struct jtj
{
	//21 floats for each product (21)
	float aa, ab, ac, ad, ae, af,
	          bb, bc, bd, be, bf,
	              cc, cd, ce, cf,
	                  dd, de, df,
	                      ee, ef,
	                          ff;
	float a, b;

	__device__ inline void add(const jtj & other)
	{
		aa += other.aa;
		ab += other.ab;
		ac += other.ac;
		ad += other.ad;
		ae += other.ae;
		af += other.af;

		bb += other.bb;
		bc += other.bc;
		bd += other.bd;
		be += other.be;
		bf += other.bf;

		cc += other.cc;
		cd += other.cd;
		ce += other.ce;
		cf += other.cf;

		dd += other.dd;
		de += other.de;
		df += other.df;

		ee += other.ee;
		ef += other.ef;

		ff += other.ff;

		a += other.a;
		b += other.b;
	}
};

void estimateCombined(const Mat33& Rcurr, const float3& tcurr, const DeviceArray2D<float>& vmap_curr, const DeviceArray2D<float>& nmap_curr, const Mat33& Rprev_inv, const float3& tprev, const Intr& intr,
                      const DeviceArray2D<float>& vmap_g_prev, const DeviceArray2D<float>& nmap_g_prev, float distThres, float angleThres,
                      DeviceArray2D<float>& gbuf, DeviceArray<float>& mbuf, float* matrixA_host, float* vectorB_host, float * residual_host);

void icpStep(const Mat33& Rcurr,
             const float3& tcurr,
             const DeviceArray2D<float>& vmap_curr,
             const DeviceArray2D<float>& nmap_curr,
             const Mat33& Rprev_inv,
             const float3& tprev,
             const Intr& intr,
             const DeviceArray2D<float>& vmap_g_prev,
             const DeviceArray2D<float>& nmap_g_prev,
             float distThres,
             float angleThres,
             DeviceArray<jtjjtr> & sum,
             DeviceArray<jtjjtr> & out,
             float * matrixA_host,
             float * vectorB_host,
             float * residual_host,
             int threads, int blocks);

void icpStep2(const Mat33& Rcurr,
	const float3& tcurr,
	const DeviceArray2D<float>& vmap_curr,
	const DeviceArray2D<float>& nmap_curr,
	const DeviceArray2D<bool>& planemap_curr,
	const DeviceArray2D<float>& plane_inlier_curr,
	const Mat33& Rprev_inv,
	const float3& tprev,
	const Intr& intr,
	const DeviceArray2D<float>& vmap_g_prev,
	const DeviceArray2D<float>& nmap_g_prev,
	const DeviceArray2D<bool>& planemap_g_prev,
	const DeviceArray<float>& plane_n_prev,
	const DeviceArray<float>& plane_d_prev,
	const DeviceArray<float>& plane_lambda_prev,
	float distThres,
	float angleThres,
	DeviceArray<jtjjtr> & sum,
	DeviceArray<jtjjtr> & out,
	float * matrixA_host,
	float * vectorB_host,
	float * residual_host,
	int threads, int blocks);

void calcCorr(const Mat33& Rcurr,
	const float3& tcurr,
	const DeviceArray2D<float>& vmap_curr,
	const DeviceArray2D<float>& nmap_curr,
	const Intr& intr,
	const DeviceArray2D<float>& vmap_g_prev,
	const DeviceArray2D<float>& nmap_g_prev,
	float distThres,
	float angleThres,
	DeviceArray<jtj> & sum,
	DeviceArray<jtj> & out,
	double * matrixA_host,
	int * result,
	int threads, int blocks);

void pyrDown(const DeviceArray2D<unsigned short> & src, DeviceArray2D<unsigned short> & dst);
void createVMap(const Intr& intr, const DeviceArray2D<unsigned short> & depth, DeviceArray2D<float> & vmap, const float depthCutoff, const float depthFactor);
void createNMap(const DeviceArray2D<float>& vmap, DeviceArray2D<float>& nmap);
void createPMap(const DeviceArray2D<float>& vmap, DeviceArray2D<float>& pmap);
void createNMapAndPMap(const DeviceArray2D<float>& vmap, DeviceArray2D<float>& nmap, DeviceArray2D<float>& pmap);
void createPlaneMap(const DeviceArray2D<float>& vmap, const DeviceArray2D<float>& nmap, const float4 plane, const float max_dist, DeviceArray2D<bool>& planemap);
void tranformMaps(const DeviceArray2D<float>& vmap_src,
                  const DeviceArray2D<float>& nmap_src,
                  const Mat33& Rmat, const float3& tvec,
                  DeviceArray2D<float>& vmap_dst, DeviceArray2D<float>& nmap_dst);

void copyMaps(const DeviceArray<float>& vmap_src,
              const DeviceArray<float>& nmap_src,
              DeviceArray2D<float>& vmap_dst,
              DeviceArray2D<float>& nmap_dst);

void rearrangeMap(const DeviceArray2D<float>& src,
	              DeviceArray2D<float>& dst);

void resizeVMap(const DeviceArray2D<float>& input, DeviceArray2D<float>& output);
void resizeNMap(const DeviceArray2D<float>& input, DeviceArray2D<float>& output);

template<class D, class Matx> D& device_cast (Matx& matx)
{
    return(*reinterpret_cast<D*>(matx.data ()));
}

#endif /* INTERNAL_HPP_ */
