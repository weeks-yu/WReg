/*
 * ICPOdometry.cpp
 *
 *  Created on: 17 Sep 2012
 *      Author: thomas
 */

#include "ICPOdometry.h"

ICPOdometry::ICPOdometry(int width,
                         int height,
                         float cx, float cy, float fx, float fy,
						 float depthFactor,
                         float distThresh,
                         float angleThresh)
: lastICPError(0),
  lastICPCount(width * height),
  lastA(Eigen::Matrix<double, 6, 6, Eigen::RowMajor>::Zero()),
  lastb(Eigen::Matrix<double, 6, 1>::Zero()),
  distThres_(distThresh),
  angleThres_(angleThresh),
  width(width),
  height(height),
  cx(cx), cy(cy), fx(fx), fy(fy),
  depthFactor_(depthFactor)
{
    sumData.create(MAX_THREADS);
    outData.create(1);

    intr.cx = cx;
    intr.cy = cy;
    intr.fx = fx;
    intr.fy = fy;

    iterations.reserve(NUM_PYRS);

    depth_tmp.resize(NUM_PYRS);

    vmaps_g_prev_.resize(NUM_PYRS);
    nmaps_g_prev_.resize(NUM_PYRS);
	planemaps_g_prev_.resize(NUM_PYRS);

    vmaps_curr_.resize(NUM_PYRS);
    nmaps_curr_.resize(NUM_PYRS);
	pmaps_curr_.resize(NUM_PYRS);
	planemaps_curr_.resize(NUM_PYRS);

    for (int i = 0; i < NUM_PYRS; ++i)
    {
        int pyr_rows = height >> i;
        int pyr_cols = width >> i;

        depth_tmp[i].create (pyr_rows, pyr_cols);

        vmaps_g_prev_[i].create (pyr_rows*3, pyr_cols);
        nmaps_g_prev_[i].create (pyr_rows*3, pyr_cols);
		planemaps_g_prev_[i].create(pyr_rows, pyr_cols);

        vmaps_curr_[i].create (pyr_rows*3, pyr_cols);
        nmaps_curr_[i].create (pyr_rows*3, pyr_cols);
		pmaps_curr_[i].create (pyr_rows,   pyr_cols);
		planemaps_curr_[i].create(pyr_rows, pyr_cols);
    }
}

ICPOdometry::~ICPOdometry()
{

}

void ICPOdometry::initICP(unsigned short * depth, const float depthCutoff)
{
    depth_tmp[0].upload(depth, sizeof(unsigned short) * width, height, width);

    for(int i = 1; i < NUM_PYRS; ++i)
    {
        pyrDown(depth_tmp[i - 1], depth_tmp[i]);
    }

    for(int i = 0; i < NUM_PYRS; ++i)
    {
        createVMap(intr(i), depth_tmp[i], vmaps_curr_[i], depthCutoff, depthFactor_);
        createNMap(vmaps_curr_[i], nmaps_curr_[i]);
    }

	createPMap(vmaps_curr_[0], pmaps_curr_[0]);

    cudaDeviceSynchronize();

	DeviceArray2D<float> dst;
	
	rearrangeMap(vmaps_curr_[0], dst);
	vmap_curr = cv::Mat(height, width, CV_32FC3);
	dst.download(vmap_curr.data, vmap_curr.step[0]);

	rearrangeMap(nmaps_curr_[0], dst);
	nmap_curr = cv::Mat(height, width, CV_32FC3);
	dst.download(nmap_curr.data, nmap_curr.step[0]);

	pmap_curr = cv::Mat(height, width, CV_32FC1);
	pmaps_curr_[0].download(pmap_curr.data, pmap_curr.step[0]);

	dst.release();
}

void ICPOdometry::initICPModel(unsigned short * depth,
                               const float depthCutoff,
                               const Eigen::Matrix4f & modelPose)
{
    depth_tmp[0].upload(depth, sizeof(unsigned short) * width, height, width);

    for(int i = 1; i < NUM_PYRS; ++i)
    {
        pyrDown(depth_tmp[i - 1], depth_tmp[i]);
    }

    for(int i = 0; i < NUM_PYRS; ++i)
    {
        createVMap(intr(i), depth_tmp[i], vmaps_g_prev_[i], depthCutoff, depthFactor_);
        createNMap(vmaps_g_prev_[i], nmaps_g_prev_[i]);
    }

    cudaDeviceSynchronize();

    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rcam = modelPose.topLeftCorner(3, 3);
    Eigen::Vector3f tcam = modelPose.topRightCorner(3, 1);

    Mat33 &  device_Rcam = device_cast<Mat33>(Rcam);
    float3& device_tcam = device_cast<float3>(tcam);

    for(int i = 0; i < NUM_PYRS; ++i)
    {
        tranformMaps(vmaps_g_prev_[i], nmaps_g_prev_[i], device_Rcam, device_tcam, vmaps_g_prev_[i], nmaps_g_prev_[i]);
    }

    cudaDeviceSynchronize();
}

void ICPOdometry::initPlanes(std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f>> planes_prev,
	std::vector<float> planes_lambda_prev,
	std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f>> planes_curr,
	std::vector<std::pair<int, int>> plane_corr_id,
	int plane_inliers_count,
	std::vector<std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>> plane_inliers_curr)
{
	for (int i = 0; i < plane_corr_id.size(); i++)
	{
		float4 p1, p2;
		p1.x = planes_prev[plane_corr_id[i].first](0);
		p1.y = planes_prev[plane_corr_id[i].first](1);
		p1.z = planes_prev[plane_corr_id[i].first](2);
		p1.w = planes_prev[plane_corr_id[i].first](3);

		p2.x = planes_curr[plane_corr_id[i].second](0);
		p2.y = planes_curr[plane_corr_id[i].second](1);
		p2.z = planes_curr[plane_corr_id[i].second](2);
		p2.w = planes_curr[plane_corr_id[i].second](3);

		for (int j = 0; j < NUM_PYRS; j++)
		{
			createPlaneMap(vmaps_g_prev_[j], nmaps_g_prev_[j], p1, 0.02, planemaps_g_prev_[j]);
			createPlaneMap(vmaps_curr_[j], nmaps_curr_[j], p2, 0.02, planemaps_curr_[j]);
			
		}
		cudaDeviceSynchronize();
	}
	
	plane_count_ = plane_corr_id.size();
	float * ntmp = new float[plane_count_ * 3];
	float * dtmp = new float[plane_count_];
	float * lambdatmp = new float[plane_count_];
	float * plane_inliers = new float[plane_count_ * plane_inliers_count * 3];
	for (int i = 0; i < plane_corr_id.size(); i++)
	{
		ntmp[i] = planes_prev[plane_corr_id[i].first](0);
		ntmp[i + plane_count_] = planes_prev[plane_corr_id[i].first](1);
		ntmp[i + 2 * plane_count_] = planes_prev[plane_corr_id[i].first](2);

		dtmp[i] = planes_prev[plane_corr_id[i].first](3);

		lambdatmp[i] = planes_lambda_prev[plane_corr_id[i].first];

		for (int j = 0; j < plane_inliers_count; j++)
		{
			plane_inliers[i * plane_inliers_count + j] = plane_inliers_curr[i][j](0);
			plane_inliers[i * plane_inliers_count + j + plane_count_ * plane_inliers_count] = plane_inliers_curr[i][j](1);
			plane_inliers[i * plane_inliers_count + j + 2 * plane_count_ * plane_inliers_count] = plane_inliers_curr[i][j](2);
		}
	}
	plane_n_prev_.create(plane_count_ * 3);
	plane_n_prev_.upload(ntmp, plane_count_ * 3);

	plane_d_prev_.create(plane_count_);
	plane_d_prev_.upload(dtmp, plane_count_);

	planes_lambda_prev_.create(plane_count_);
	planes_lambda_prev_.upload(lambdatmp, plane_count_);

	plane_inliers_curr_.create(plane_count_ * 3, plane_inliers_count);
	plane_inliers_curr_.upload(plane_inliers, sizeof(float) * plane_inliers_count, plane_count_ * 3, plane_inliers_count);

	cudaDeviceSynchronize();

	delete ntmp;
	delete dtmp;
	delete plane_inliers;
}

void ICPOdometry::getIncrementalTransformation(Eigen::Vector3f & trans, Eigen::Matrix<float, 3, 3, Eigen::RowMajor> & rot,
	Eigen::Vector3f & estimated_trans, Eigen::Matrix<float, 3, 3, Eigen::RowMajor> & estimated_rot, int threads, int blocks)
{
    iterations.push_back(10);
    iterations.push_back(5);
    iterations.push_back(4);

    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rprev = rot;
    Eigen::Vector3f tprev = trans;

    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rcurr = estimated_rot;
    Eigen::Vector3f tcurr = estimated_trans;

    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rprev_inv = Rprev.inverse();
    Mat33 & device_Rprev_inv = device_cast<Mat33>(Rprev_inv);
    float3& device_tprev = device_cast<float3>(tprev);

    cv::Mat resultRt = cv::Mat::eye(4, 4, CV_64FC1);

    for(int i = NUM_PYRS - 1; i >= 0; i--)
    {
        for(int j = 0; j < iterations[i]; j++)
        {
            Eigen::Matrix<float, 6, 6, Eigen::RowMajor> A_icp;
            Eigen::Matrix<float, 6, 1> b_icp;

            Mat33&  device_Rcurr = device_cast<Mat33> (Rcurr);
            float3& device_tcurr = device_cast<float3>(tcurr);

            DeviceArray2D<float>& vmap_curr = vmaps_curr_[i];
            DeviceArray2D<float>& nmap_curr = nmaps_curr_[i];

            DeviceArray2D<float>& vmap_g_prev = vmaps_g_prev_[i];
            DeviceArray2D<float>& nmap_g_prev = nmaps_g_prev_[i];

            float residual[2];

            icpStep(device_Rcurr,
                    device_tcurr,
                    vmap_curr,
                    nmap_curr,
                    device_Rprev_inv,
                    device_tprev,
                    intr(i),
                    vmap_g_prev,
                    nmap_g_prev,
                    distThres_,
                    angleThres_,
                    sumData,
                    outData,
                    A_icp.data(),
                    b_icp.data(),
                    &residual[0],
                    threads,
                    blocks);

            lastICPError = sqrt(residual[0]) / residual[1];
            lastICPCount = residual[1];

            Eigen::Matrix<double, 6, 1> result;
            Eigen::Matrix<double, 6, 6, Eigen::RowMajor> dA_icp = A_icp.cast<double>();
            Eigen::Matrix<double, 6, 1> db_icp = b_icp.cast<double>();

            lastA = dA_icp;
            lastb = db_icp;
            result = lastA.ldlt().solve(lastb);

            Eigen::Isometry3f incOdom;

            OdometryProvider::computeProjectiveMatrix(resultRt, result, incOdom);

            Eigen::Isometry3f currentT;
            currentT.setIdentity();
            currentT.rotate(Rprev);
            currentT.translation() = tprev;

            currentT = currentT * incOdom.inverse();

            tcurr = currentT.translation();
            Rcurr = currentT.rotation();
        }
    }

    trans = tcurr;
    rot = Rcurr;
}

void ICPOdometry::getIncrementalTransformationWithPlane(Eigen::Vector3f & trans, Eigen::Matrix<float, 3, 3, Eigen::RowMajor> & rot,
	Eigen::Vector3f & estimated_trans, Eigen::Matrix<float, 3, 3, Eigen::RowMajor> & estimated_rot, int threads, int blocks)
{
	iterations.push_back(10);
	iterations.push_back(5);
	iterations.push_back(4);

	Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rprev = rot;
	Eigen::Vector3f tprev = trans;

	Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rcurr = estimated_rot;
	Eigen::Vector3f tcurr = estimated_trans;

	Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rprev_inv = Rprev.inverse();
	Mat33 & device_Rprev_inv = device_cast<Mat33>(Rprev_inv);
	float3& device_tprev = device_cast<float3>(tprev);

	cv::Mat resultRt = cv::Mat::eye(4, 4, CV_64FC1);

	for (int i = NUM_PYRS - 1; i >= 0; i--)
	{
		for (int j = 0; j < iterations[i]; j++)
		{
			Eigen::Matrix<float, 6, 6, Eigen::RowMajor> A_icp;
			Eigen::Matrix<float, 6, 1> b_icp;

			Mat33&  device_Rcurr = device_cast<Mat33> (Rcurr);
			float3& device_tcurr = device_cast<float3>(tcurr);

			DeviceArray2D<float>& vmap_curr = vmaps_curr_[i];
			DeviceArray2D<float>& nmap_curr = nmaps_curr_[i];
			DeviceArray2D<bool>& planemap_curr = planemaps_curr_[i];

			DeviceArray2D<float>& vmap_g_prev = vmaps_g_prev_[i];
			DeviceArray2D<float>& nmap_g_prev = nmaps_g_prev_[i];
			DeviceArray2D<bool>& planemap_g_prev = planemaps_g_prev_[i];

			float residual[2];

			icpStep2(device_Rcurr,
				device_tcurr,
				vmap_curr,
				nmap_curr,
				planemap_curr,
				plane_inliers_curr_,
				device_Rprev_inv,
				device_tprev,
				intr(i),
				vmap_g_prev,
				nmap_g_prev,
				planemap_g_prev,
				plane_n_prev_,
				plane_d_prev_,
				planes_lambda_prev_,
				distThres_,
				angleThres_,
				sumData,
				outData,
				A_icp.data(),
				b_icp.data(),
				&residual[0],
				threads,
				blocks);

			lastICPError = sqrt(residual[0]) / residual[1];
			lastICPCount = residual[1];

			Eigen::Matrix<double, 6, 1> result;
			Eigen::Matrix<double, 6, 6, Eigen::RowMajor> dA_icp = A_icp.cast<double>();
			Eigen::Matrix<double, 6, 1> db_icp = b_icp.cast<double>();

			lastA = dA_icp;
			lastb = db_icp;
			result = lastA.ldlt().solve(lastb);

			Eigen::Isometry3f incOdom;

			OdometryProvider::computeProjectiveMatrix(resultRt, result, incOdom);

			Eigen::Isometry3f currentT;
			currentT.setIdentity();
			currentT.rotate(Rprev);
			currentT.translation() = tprev;

			currentT = currentT * incOdom.inverse();

			tcurr = currentT.translation();
			Rcurr = currentT.rotation();
		}
	}

	trans = tcurr;
	rot = Rcurr;
}

Eigen::MatrixXd ICPOdometry::getCovariance()
{
    return lastA.cast<double>().lu().inverse();
}

void ICPOdometry::getVMapCurr(cv::Mat &mat)
{
	vmap_curr.copyTo(mat);
}

void ICPOdometry::getNMapCurr(cv::Mat &mat)
{
	nmap_curr.copyTo(mat);
}

void ICPOdometry::getPMapCurr(cv::Mat &mat)
{
	pmap_curr.copyTo(mat);
}

void ICPOdometry::getPlaneMapCurr(bool *ret)
{
	planemaps_curr_[0].download(ret, width * sizeof(bool));
}

void ICPOdometry::getPlaneMapPrev(bool *ret)
{
	planemaps_g_prev_[0].download(ret, width * sizeof(bool));
}