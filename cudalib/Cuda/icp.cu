#include "internal.h"
#include "vector_math.hpp"
#include "containers/safe_call.hpp"

#if __CUDA_ARCH__ < 300
__inline__ __device__
float __shfl_down(float val, int offset, int width = 32)
{
    static __shared__ float shared[MAX_THREADS];
    int lane = threadIdx.x % 32;
    shared[threadIdx.x] = val;
    __syncthreads();
    val = (lane + offset < width) ? shared[threadIdx.x + offset] : 0;
    __syncthreads();
    return val;
}
__inline__ __device__
float __shfl_down(int val, int offset, int width = 32)
{
	static __shared__ int shared[MAX_THREADS];
	int lane = threadIdx.x % 32;
	shared[threadIdx.x] = val;
	__syncthreads();
	val = (lane + offset < width) ? shared[threadIdx.x + offset] : 0;
	__syncthreads();
	return val;
}
#endif

#if __CUDA_ARCH__ < 350
template<typename T>
__device__ __forceinline__ T __ldg(const T* ptr)
{
    return *ptr;
}
#endif

__inline__  __device__ jtjjtr warpReduceSum(jtjjtr val)
{
    for(int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        val.aa += __shfl_down(val.aa, offset);
        val.ab += __shfl_down(val.ab, offset);
        val.ac += __shfl_down(val.ac, offset);
        val.ad += __shfl_down(val.ad, offset);
        val.ae += __shfl_down(val.ae, offset);
        val.af += __shfl_down(val.af, offset);
        val.ag += __shfl_down(val.ag, offset);

        val.bb += __shfl_down(val.bb, offset);
        val.bc += __shfl_down(val.bc, offset);
        val.bd += __shfl_down(val.bd, offset);
        val.be += __shfl_down(val.be, offset);
        val.bf += __shfl_down(val.bf, offset);
        val.bg += __shfl_down(val.bg, offset);

        val.cc += __shfl_down(val.cc, offset);
        val.cd += __shfl_down(val.cd, offset);
        val.ce += __shfl_down(val.ce, offset);
        val.cf += __shfl_down(val.cf, offset);
        val.cg += __shfl_down(val.cg, offset);

        val.dd += __shfl_down(val.dd, offset);
        val.de += __shfl_down(val.de, offset);
        val.df += __shfl_down(val.df, offset);
        val.dg += __shfl_down(val.dg, offset);

        val.ee += __shfl_down(val.ee, offset);
        val.ef += __shfl_down(val.ef, offset);
        val.eg += __shfl_down(val.eg, offset);

        val.ff += __shfl_down(val.ff, offset);
        val.fg += __shfl_down(val.fg, offset);

        val.residual += __shfl_down(val.residual, offset);
        val.inliers += __shfl_down(val.inliers, offset);
    }

    return val;
}

__inline__  __device__ jtjjtr blockReduceSum(jtjjtr val)
{
    static __shared__ jtjjtr shared[32];

    int lane = threadIdx.x % warpSize;

    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val);

    //write reduced value to shared memory
    if(lane == 0)
    {
        shared[wid] = val;
    }
    __syncthreads();

    const jtjjtr zero = {0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0};

    //ensure we only grab a value from shared memory if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : zero;

    if(wid == 0)
    {
        val = warpReduceSum(val);
    }

    return val;
}

__global__ void reduceSum(jtjjtr * in, jtjjtr * out, int N)
{
    jtjjtr sum = {0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0};

    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
    {
        sum.add(in[i]);
    }

    sum = blockReduceSum(sum);

    if(threadIdx.x == 0)
    {
        out[blockIdx.x] = sum;
    }
}

struct ICPReduction
{
    Mat33 Rcurr;
    float3 tcurr;

    PtrStep<float> vmap_curr;
    PtrStep<float> nmap_curr;
	PtrStep<bool> planemap_curr;

    Mat33 Rprev_inv;
    float3 tprev;

    Intr intr;

    PtrStep<float> vmap_g_prev;
    PtrStep<float> nmap_g_prev;
	PtrStep<bool> planemap_g_prev;

    float distThres;
    float angleThres;

    int cols;
    int rows;
    int N;

	int P;								// 平面的个数
	int P_inliner_count;				// 每个平面的内点个数
	PtrSz<float> plane_n_prev;			// 平面的法向量(a,b,c)
	PtrSz<float> plane_d_prev;			// 平面的d
	PtrStep<float> plane_inlier_curr;	// 每个内点的坐标
	PtrSz<float> plane_lambda_prev;

    jtjjtr * out;

    __device__ __forceinline__ bool
    search (int & x, int & y, float3& n, float3& d, float3& s) const
    {
        float3 vcurr;
        vcurr.x = vmap_curr.ptr (y       )[x];
        vcurr.y = vmap_curr.ptr (y + rows)[x];
        vcurr.z = vmap_curr.ptr (y + 2 * rows)[x];

        float3 vcurr_g = Rcurr * vcurr + tcurr;
        float3 vcurr_cp = Rprev_inv * (vcurr_g - tprev);         // prev camera coo space

        int2 ukr;         //projection
        ukr.x = __float2int_rn (vcurr_cp.x * intr.fx / vcurr_cp.z + intr.cx);      //4
        ukr.y = __float2int_rn (vcurr_cp.y * intr.fy / vcurr_cp.z + intr.cy);                      //4

        if(ukr.x < 0 || ukr.y < 0 || ukr.x >= cols || ukr.y >= rows || vcurr_cp.z < 0)
            return false;

		bool is_plane_point = planemap_g_prev.ptr(ukr.y)[ukr.x];
		if (P > 0 && is_plane_point)
			return false;

        float3 vprev_g;
        vprev_g.x = __ldg(&vmap_g_prev.ptr (ukr.y       )[ukr.x]);
        vprev_g.y = __ldg(&vmap_g_prev.ptr (ukr.y + rows)[ukr.x]);
        vprev_g.z = __ldg(&vmap_g_prev.ptr (ukr.y + 2 * rows)[ukr.x]);

        float3 ncurr;
        ncurr.x = nmap_curr.ptr (y)[x];
        ncurr.y = nmap_curr.ptr (y + rows)[x];
        ncurr.z = nmap_curr.ptr (y + 2 * rows)[x];

        float3 ncurr_g = Rcurr * ncurr;

        float3 nprev_g;
        nprev_g.x =  __ldg(&nmap_g_prev.ptr (ukr.y)[ukr.x]);
        nprev_g.y = __ldg(&nmap_g_prev.ptr (ukr.y + rows)[ukr.x]);
        nprev_g.z = __ldg(&nmap_g_prev.ptr (ukr.y + 2 * rows)[ukr.x]);

        float dist = norm (vprev_g - vcurr_g);
        float sine = norm (cross (ncurr_g, nprev_g));

        n = nprev_g;
        d = vprev_g;
        s = vcurr_g;

        return (sine < angleThres && dist <= distThres && !isnan (ncurr.x) && !isnan (nprev_g.x));
    }

    __device__ __forceinline__ jtjjtr
    getProducts(int & i) const
    {
        int y = i / cols;
        int x = i - (y * cols);

		float row[7] = { 0, 0, 0, 0, 0, 0, 0 };
		bool is_plane_point = (P > 0) && (planemap_curr.ptr(y)[x]);
		bool found_coresp = false;

		if (!is_plane_point)
		{
			float3 n_cp, d_cp, s_cp;
			found_coresp = search(x, y, n_cp, d_cp, s_cp);

			if (found_coresp)
			{
				s_cp = Rprev_inv * (s_cp - tprev);         // prev camera coo space
				d_cp = Rprev_inv * (d_cp - tprev);         // prev camera coo space
				n_cp = Rprev_inv * (n_cp);                // prev camera coo space

				*(float3*)&row[0] = n_cp;
				*(float3*)&row[3] = cross(s_cp, n_cp);
				row[6] = dot(n_cp, s_cp - d_cp);
			}
		}

        jtjjtr values = {row[0] * row[0],
                         row[0] * row[1],
                         row[0] * row[2],
                         row[0] * row[3],
                         row[0] * row[4],
                         row[0] * row[5],
                         row[0] * row[6],

                         row[1] * row[1],
                         row[1] * row[2],
                         row[1] * row[3],
                         row[1] * row[4],
                         row[1] * row[5],
                         row[1] * row[6],

                         row[2] * row[2],
                         row[2] * row[3],
                         row[2] * row[4],
                         row[2] * row[5],
                         row[2] * row[6],

                         row[3] * row[3],
                         row[3] * row[4],
                         row[3] * row[5],
                         row[3] * row[6],

                         row[4] * row[4],
                         row[4] * row[5],
                         row[4] * row[6],

                         row[5] * row[5],
                         row[5] * row[6],

                         row[6] * row[6],
                         found_coresp};

        return values;
    }

	__device__ __forceinline__ jtjjtr
		getPlaneProducts(int i) const
	{
		int y = i / P_inliner_count;
		int x = i - (y * P_inliner_count);

		float3 n_cp, s_cp;
		n_cp.x = plane_n_prev[y];
		n_cp.y = plane_n_prev[y + P];
		n_cp.z = plane_n_prev[y + 2 * P];

		s_cp.x = plane_inlier_curr.ptr(y)[x];
		s_cp.y = plane_inlier_curr.ptr(y + rows)[x];
		s_cp.z = plane_inlier_curr.ptr(y + 2 * rows)[x];
		s_cp = Rcurr * s_cp + tcurr;

		float d = plane_d_prev[y];
		float lambda = plane_lambda_prev[y];

		float row[7] = { 0, 0, 0, 0, 0, 0, 0 };

		s_cp = Rprev_inv * (s_cp - tprev);         // prev camera coo space
		d += dot(tprev, n_cp);						// prev camera coo space
		n_cp = n_cp * Rprev_inv;                // prev camera coo space
		

		*(float3*)&row[0] = n_cp;
		*(float3*)&row[3] = cross(s_cp, n_cp);
		row[6] = -dot(n_cp, s_cp) - d;

		jtjjtr values = { lambda * row[0] * row[0],
						  lambda * row[0] * row[1],
						  lambda * row[0] * row[2],
						  lambda * row[0] * row[3],
						  lambda * row[0] * row[4],
						  lambda * row[0] * row[5],
						  lambda * row[0] * row[6],

						  lambda * row[1] * row[1],
						  lambda * row[1] * row[2],
						  lambda * row[1] * row[3],
						  lambda * row[1] * row[4],
						  lambda * row[1] * row[5],
						  lambda * row[1] * row[6],

						  lambda * row[2] * row[2],
						  lambda * row[2] * row[3],
						  lambda * row[2] * row[4],
						  lambda * row[2] * row[5],
						  lambda * row[2] * row[6],

						  lambda * row[3] * row[3],
						  lambda * row[3] * row[4],
						  lambda * row[3] * row[5],
						  lambda * row[3] * row[6],

						  lambda * row[4] * row[4],
						  lambda * row[4] * row[5],
						  lambda * row[4] * row[6],

						  lambda * row[5] * row[5],
						  lambda * row[5] * row[6],

						  lambda * row[6] * row[6],
						  true };

		return values;
	}

    __device__ __forceinline__ void
    operator () () const
    {
        jtjjtr sum = {0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0};

        for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N + P * P_inliner_count; i += blockDim.x * gridDim.x)
        {
			if (i < N)
			{
				jtjjtr val = getProducts(i);
				sum.add(val);
			}
			else
			{
				jtjjtr val = getPlaneProducts(i - N);
				sum.add(val);
			}
        }

        sum = blockReduceSum(sum);

        if(threadIdx.x == 0)
        {
            out[blockIdx.x] = sum;
        }
    }
};

__global__ void icpKernel(const ICPReduction icp)
{
    icp();
}

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
             int threads, int blocks)
{
    int cols = vmap_curr.cols ();
    int rows = vmap_curr.rows () / 3;

    ICPReduction icp;

    icp.Rcurr = Rcurr;
    icp.tcurr = tcurr;

    icp.vmap_curr = vmap_curr;
    icp.nmap_curr = nmap_curr;

    icp.Rprev_inv = Rprev_inv;
    icp.tprev = tprev;

    icp.intr = intr;

    icp.vmap_g_prev = vmap_g_prev;
    icp.nmap_g_prev = nmap_g_prev;

    icp.distThres = distThres;
    icp.angleThres = angleThres;

    icp.cols = cols;
    icp.rows = rows;

    icp.N = cols * rows;
	icp.P = 0;
    icp.out = sum;

    icpKernel<<<blocks, threads>>>(icp);

    reduceSum<<<1, MAX_THREADS>>>(sum, out, blocks);

    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());

    float host_data[32];
    out.download((jtjjtr *)&host_data[0]);

    int shift = 0;
    for (int i = 0; i < 6; ++i)  //rows
    {
        for (int j = i; j < 7; ++j)    // cols + b
        {
            float value = host_data[shift++];
            if (j == 6)       // vector b
                vectorB_host[i] = value;
            else
                matrixA_host[j * 6 + i] = matrixA_host[i * 6 + j] = value;
        }
    }

    residual_host[0] = host_data[27];
    residual_host[1] = host_data[28];
}

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
	int threads, int blocks)
{
	int cols = vmap_curr.cols();
	int rows = vmap_curr.rows() / 3;

	ICPReduction icp;

	icp.Rcurr = Rcurr;
	icp.tcurr = tcurr;

	icp.vmap_curr = vmap_curr;
	icp.nmap_curr = nmap_curr;

	icp.Rprev_inv = Rprev_inv;
	icp.tprev = tprev;

	icp.intr = intr;

	icp.vmap_g_prev = vmap_g_prev;
	icp.nmap_g_prev = nmap_g_prev;

	icp.distThres = distThres;
	icp.angleThres = angleThres;

	icp.cols = cols;
	icp.rows = rows;

	icp.N = cols * rows;
	icp.out = sum;

	icp.P = plane_d_prev.size();
	icp.P_inliner_count = plane_inlier_curr.cols();
	icp.plane_n_prev = plane_n_prev;
	icp.plane_d_prev = plane_d_prev;
	icp.planemap_g_prev = planemap_g_prev;
	icp.plane_inlier_curr = plane_inlier_curr;
	icp.planemap_curr = planemap_curr;
	icp.plane_lambda_prev = plane_lambda_prev;

	icpKernel <<<blocks, threads >>>(icp);

	reduceSum <<<1, MAX_THREADS >>>(sum, out, blocks);

	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());

	float host_data[32];
	out.download((jtjjtr *)&host_data[0]);

	int shift = 0;
	for (int i = 0; i < 6; ++i)  //rows
	{
		for (int j = i; j < 7; ++j)    // cols + b
		{
			float value = host_data[shift++];
			if (j == 6)       // vector b
				vectorB_host[i] = value;
			else
				matrixA_host[j * 6 + i] = matrixA_host[i * 6 + j] = value;
		}
	}

	residual_host[0] = host_data[27];
	residual_host[1] = host_data[28];
}

__inline__  __device__ jtj warpReduceSum(jtj val)
{
	for (int offset = warpSize / 2; offset > 0; offset /= 2)
	{
		val.aa += __shfl_down(val.aa, offset);
		val.ab += __shfl_down(val.ab, offset);
		val.ac += __shfl_down(val.ac, offset);
		val.ad += __shfl_down(val.ad, offset);
		val.ae += __shfl_down(val.ae, offset);
		val.af += __shfl_down(val.af, offset);

		val.bb += __shfl_down(val.bb, offset);
		val.bc += __shfl_down(val.bc, offset);
		val.bd += __shfl_down(val.bd, offset);
		val.be += __shfl_down(val.be, offset);
		val.bf += __shfl_down(val.bf, offset);

		val.cc += __shfl_down(val.cc, offset);
		val.cd += __shfl_down(val.cd, offset);
		val.ce += __shfl_down(val.ce, offset);
		val.cf += __shfl_down(val.cf, offset);

		val.dd += __shfl_down(val.dd, offset);
		val.de += __shfl_down(val.de, offset);
		val.df += __shfl_down(val.df, offset);

		val.ee += __shfl_down(val.ee, offset);
		val.ef += __shfl_down(val.ef, offset);

		val.ff += __shfl_down(val.ff, offset);

		val.a += __shfl_down(val.a, offset);
		val.b += __shfl_down(val.b, offset);
	}

	return val;
}

__inline__  __device__ jtj blockReduceSum(jtj val)
{
	static __shared__ jtj shared[32];

	int lane = threadIdx.x % warpSize;

	int wid = threadIdx.x / warpSize;

	val = warpReduceSum(val);

	//write reduced value to shared memory
	if (lane == 0)
	{
		shared[wid] = val;
	}
	__syncthreads();

	const jtj zero = { 0, 0, 0, 0, 0, 0,
	                   0, 0, 0, 0, 0,
	                   0, 0, 0, 0,
	                   0, 0, 0,
	                   0, 0,
	                   0,
	                   0, 0};

	//ensure we only grab a value from shared memory if that warp existed
	val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : zero;

	if (wid == 0)
	{
		val = warpReduceSum(val);
	}

	return val;
}

__global__ void reduceSum(jtj * in, jtj * out, int N)
{
	jtj sum = { 0, 0, 0, 0, 0, 0,
	            0, 0, 0, 0, 0,
	            0, 0, 0, 0,
	            0, 0, 0,
	            0, 0,
	            0,
	            0, 0};

	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
	{
		sum.add(in[i]);
	}

	sum = blockReduceSum(sum);

	if (threadIdx.x == 0)
	{
		out[blockIdx.x] = sum;
	}
}

struct CorrCalculator
{
	Mat33 Rcurr;
	float3 tcurr;

	PtrStep<float> vmap_curr;
	PtrStep<float> nmap_curr;

	Intr intr;

	PtrStep<float> vmap_g_prev;
	PtrStep<float> nmap_g_prev;

	float distThres;
	float angleThres;

	int cols;
	int rows;
	int N;

	jtj * out;

	__device__ __forceinline__ jtj
		search(int & x, int & y) const
	{
		jtj ret = {0, 0, 0, 0, 0, 0,
		              0, 0, 0, 0, 0,
		                 0, 0, 0, 0,
		                    0, 0, 0,
		                       0, 0,
		                          0,
		                       0, 0};

		float3 vcurr;
		vcurr.x = vmap_curr.ptr(y)[x];
		if (isnan(vcurr.x))
			return ret;
		ret.a = 1;

		vcurr.y = vmap_curr.ptr(y + rows)[x];
		vcurr.z = vmap_curr.ptr(y + 2 * rows)[x];

		float3 vcurr_g = Rcurr * vcurr + tcurr;

		int2 ukr;         //projection
		ukr.x = __float2int_rn(vcurr_g.x * intr.fx / vcurr_g.z + intr.cx);      //4
		ukr.y = __float2int_rn(vcurr_g.y * intr.fy / vcurr_g.z + intr.cy);                      //4

		if (ukr.x < 0 || ukr.y < 0 || ukr.x >= cols || ukr.y >= rows || vcurr_g.z < 0)
			return ret;

		float3 vprev_g;
		vprev_g.x = __ldg(&vmap_g_prev.ptr(ukr.y)[ukr.x]);
		vprev_g.y = __ldg(&vmap_g_prev.ptr(ukr.y + rows)[ukr.x]);
		vprev_g.z = __ldg(&vmap_g_prev.ptr(ukr.y + 2 * rows)[ukr.x]);

		float3 ncurr;
		ncurr.x = nmap_curr.ptr(y)[x];
		ncurr.y = nmap_curr.ptr(y + rows)[x];
		ncurr.z = nmap_curr.ptr(y + 2 * rows)[x];

		float3 ncurr_g = Rcurr * ncurr;

		float3 nprev_g;
		nprev_g.x = __ldg(&nmap_g_prev.ptr(ukr.y)[ukr.x]);
		nprev_g.y = __ldg(&nmap_g_prev.ptr(ukr.y + rows)[ukr.x]);
		nprev_g.z = __ldg(&nmap_g_prev.ptr(ukr.y + 2 * rows)[ukr.x]);

		float dist = norm(vprev_g - vcurr_g);
		float sine = norm(cross(ncurr_g, nprev_g));

		if (sine < angleThres && dist <= distThres && !isnan(ncurr.x) && !isnan(nprev_g.x))
		{
			ret.b = 1;

			ret.aa = 1;
			ret.ab = 0;
			ret.ac = 0;
			ret.ad = 0;
			ret.ae = 2 * vcurr.z;
			ret.af = -2 * vcurr.y;

			ret.bb = 1;
			ret.bc = 0;
			ret.bd = -2 * vcurr.z;
			ret.be = 0;
			ret.bf = 2 * vcurr.x;

			ret.cc = 1;
			ret.cd = 2 * vcurr.y;
			ret.ce = -2 * vcurr.x;
			ret.cf = 0;

			ret.dd = 4 * (vcurr.y * vcurr.y + vcurr.z * vcurr.z);
			ret.de = -4 * vcurr.x * vcurr.y;
			ret.df = -4 * vcurr.x * vcurr.z;

			ret.ee = 4 * (vcurr.x * vcurr.x + vcurr.z * vcurr.z);
			ret.ef = -4 * vcurr.y * vcurr.z;

			ret.ff = 4 * (vcurr.x * vcurr.x + vcurr.y * vcurr.y);

/*			ret.aa = vcurr.y * vcurr.y + vcurr.z * vcurr.z;
			ret.ab = -vcurr.x * vcurr.y;
			ret.ac = -vcurr.x * vcurr.z;
			ret.ad = 0;
			ret.ae = -vcurr.z;
			ret.af = vcurr.y;

			ret.bb = vcurr.x * vcurr.x + vcurr.z * vcurr.z;
			ret.bc = vcurr.y * vcurr.z;
			ret.bd = vcurr.z;
			ret.be = 0;
			ret.bf = -vcurr.x;

			ret.cc = vcurr.x * vcurr.x + vcurr.y * vcurr.y;
			ret.cd = vcurr.y;
			ret.ce = vcurr.x;
			ret.cf = 0;

			ret.dd = 1;
			ret.de = 0;
			ret.df = 0;

			ret.ee = 1;
			ret.ef = 0;

			ret.ff = 1;*/
		}

		return ret;
	}

	__device__ __forceinline__ jtj
		search(int & x, int & y, PtrStep<int> pairs) const
	{
			jtj ret = { 0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0,
				0, 0, 0, 0,
				0, 0, 0,
				0, 0,
				0,
				0, 0 };

			float3 vcurr;
			vcurr.x = vmap_curr.ptr(y)[x];
			if (isnan(vcurr.x))
			{
				pairs.ptr(y)[x * 2] = -1;
				pairs.ptr(y)[x * 2 + 1] = -1;
				return ret;
			}
				
			ret.a = 1;

			vcurr.y = vmap_curr.ptr(y + rows)[x];
			vcurr.z = vmap_curr.ptr(y + 2 * rows)[x];

			float3 vcurr_g = Rcurr * vcurr + tcurr;

			int2 ukr;         //projection
			ukr.x = __float2int_rn(vcurr_g.x * intr.fx / vcurr_g.z + intr.cx);      //4
			ukr.y = __float2int_rn(vcurr_g.y * intr.fy / vcurr_g.z + intr.cy);                      //4

			if (ukr.x < 0 || ukr.y < 0 || ukr.x >= cols || ukr.y >= rows || vcurr_g.z < 0)
			{
				pairs.ptr(y)[x * 2] = -1;
				pairs.ptr(y)[x * 2 + 1] = -1;
				return ret;
			}
				

			float3 vprev_g;
			vprev_g.x = __ldg(&vmap_g_prev.ptr(ukr.y)[ukr.x]);
			vprev_g.y = __ldg(&vmap_g_prev.ptr(ukr.y + rows)[ukr.x]);
			vprev_g.z = __ldg(&vmap_g_prev.ptr(ukr.y + 2 * rows)[ukr.x]);

			float3 ncurr;
			ncurr.x = nmap_curr.ptr(y)[x];
			ncurr.y = nmap_curr.ptr(y + rows)[x];
			ncurr.z = nmap_curr.ptr(y + 2 * rows)[x];

			float3 ncurr_g = Rcurr * ncurr;

			float3 nprev_g;
			nprev_g.x = __ldg(&nmap_g_prev.ptr(ukr.y)[ukr.x]);
			nprev_g.y = __ldg(&nmap_g_prev.ptr(ukr.y + rows)[ukr.x]);
			nprev_g.z = __ldg(&nmap_g_prev.ptr(ukr.y + 2 * rows)[ukr.x]);

			float dist = norm(vprev_g - vcurr_g);
			float sine = norm(cross(ncurr_g, nprev_g));

			if (sine < angleThres && dist <= distThres && !isnan(ncurr.x) && !isnan(nprev_g.x))
			{
				ret.b = 1;

				ret.aa = 1;
				ret.ab = 0;
				ret.ac = 0;
				ret.ad = 0;
				ret.ae = 2 * vcurr.z;
				ret.af = -2 * vcurr.y;

				ret.bb = 1;
				ret.bc = 0;
				ret.bd = -2 * vcurr.z;
				ret.be = 0;
				ret.bf = 2 * vcurr.x;

				ret.cc = 1;
				ret.cd = 2 * vcurr.y;
				ret.ce = -2 * vcurr.x;
				ret.cf = 0;

				ret.dd = 4 * (vcurr.y * vcurr.y + vcurr.z * vcurr.z);
				ret.de = -4 * vcurr.x * vcurr.y;
				ret.df = -4 * vcurr.x * vcurr.z;

				ret.ee = 4 * (vcurr.x * vcurr.x + vcurr.z * vcurr.z);
				ret.ef = -4 * vcurr.y * vcurr.z;

				ret.ff = 4 * (vcurr.x * vcurr.x + vcurr.y * vcurr.y);

				/*			ret.aa = vcurr.y * vcurr.y + vcurr.z * vcurr.z;
				ret.ab = -vcurr.x * vcurr.y;
				ret.ac = -vcurr.x * vcurr.z;
				ret.ad = 0;
				ret.ae = -vcurr.z;
				ret.af = vcurr.y;

				ret.bb = vcurr.x * vcurr.x + vcurr.z * vcurr.z;
				ret.bc = vcurr.y * vcurr.z;
				ret.bd = vcurr.z;
				ret.be = 0;
				ret.bf = -vcurr.x;

				ret.cc = vcurr.x * vcurr.x + vcurr.y * vcurr.y;
				ret.cd = vcurr.y;
				ret.ce = vcurr.x;
				ret.cf = 0;

				ret.dd = 1;
				ret.de = 0;
				ret.df = 0;

				ret.ee = 1;
				ret.ef = 0;

				ret.ff = 1;*/

				pairs.ptr(y)[x * 2] = ukr.y;
				pairs.ptr(y)[x * 2 + 1] = ukr.x;
			}
			else
			{
				pairs.ptr(y)[x * 2] = -1;
				pairs.ptr(y)[x * 2 + 1] = -1;
			}

			return ret;
		}

	__device__ __forceinline__ void
		operator () () const
	{
		jtj sum = { 0, 0, 0, 0, 0, 0,
		            0, 0, 0, 0, 0,
		            0, 0, 0, 0,
		            0, 0, 0,
		            0, 0,
		            0,
		            0, 0};

		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
		{
			int y = i / cols;
			int x = i - (y * cols);
			sum.add(search(x, y));
		}

		sum = blockReduceSum(sum);

		if (threadIdx.x == 0)
		{
			out[blockIdx.x] = sum;
		}
	}

	__device__ __forceinline__ void
		operator () (PtrStep<int> pairs) const
	{
		jtj sum = { 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0,
			0, 0, 0, 0,
			0, 0, 0,
			0, 0,
			0,
			0, 0 };

		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
		{
			int y = i / cols;
			int x = i - (y * cols);
			sum.add(search(x, y, pairs));
		}

		sum = blockReduceSum(sum);

		if (threadIdx.x == 0)
		{
			out[blockIdx.x] = sum;
		}
	}
};

__global__ void calcCorrKernel(CorrCalculator cor)
{
	cor();
}

__global__ void calcCorrKernel2(CorrCalculator cor, PtrStep<int> pairs)
{
	cor(pairs);
}

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
	int threads, int blocks)
{
	int cols = vmap_curr.cols();
	int rows = vmap_curr.rows() / 3;

	CorrCalculator cor;

	cor.Rcurr = Rcurr;
	cor.tcurr = tcurr;

	cor.vmap_curr = vmap_curr;
	cor.nmap_curr = nmap_curr;

	cor.intr = intr;

	cor.vmap_g_prev = vmap_g_prev;
	cor.nmap_g_prev = nmap_g_prev;

	cor.distThres = distThres;
	cor.angleThres = angleThres;

	cor.cols = cols;
	cor.rows = rows;

	cor.N = cols * rows;
	cor.out = sum;

	calcCorrKernel <<<blocks, threads >>>(cor);

	reduceSum <<<1, MAX_THREADS >>>(sum, out, blocks);

	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());

	float host_data[23];
	out.download((jtj *)&host_data[0]);

	int shift = 0;
	for (int i = 0; i < 6; i++)
	{
		for (int j = i; j < 6; j++)
		{
			matrixA_host[j * 6 + i] = matrixA_host[i * 6 + j] = host_data[shift++];
		}
	}

	result[0] = host_data[21];
	result[1] = host_data[22];
}

void calcCorrWithPairs(const Mat33& Rcurr,
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
	DeviceArray2D<int> pairs,
	int threads, int blocks)
{
	int cols = vmap_curr.cols();
	int rows = vmap_curr.rows() / 3;

	CorrCalculator cor;

	cor.Rcurr = Rcurr;
	cor.tcurr = tcurr;

	cor.vmap_curr = vmap_curr;
	cor.nmap_curr = nmap_curr;

	cor.intr = intr;

	cor.vmap_g_prev = vmap_g_prev;
	cor.nmap_g_prev = nmap_g_prev;

	cor.distThres = distThres;
	cor.angleThres = angleThres;

	cor.cols = cols;
	cor.rows = rows;

	cor.N = cols * rows;
	cor.out = sum;

	calcCorrKernel2 <<<blocks, threads >>>(cor, pairs);

	reduceSum <<<1, MAX_THREADS >>>(sum, out, blocks);

	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());

	float host_data[23];
	out.download((jtj *)&host_data[0]);

	int shift = 0;
	for (int i = 0; i < 6; i++)
	{
		for (int j = i; j < 6; j++)
		{
			matrixA_host[j * 6 + i] = matrixA_host[i * 6 + j] = host_data[shift++];
		}
	}

	result[0] = host_data[21];
	result[1] = host_data[22];
}