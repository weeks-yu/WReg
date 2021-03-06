/*
 * ICPOdometry.h
 *
 *  Created on: 17 Sep 2012
 *      Author: thomas
 */

#ifndef ICPODOMETRY_H_
#define ICPODOMETRY_H_

#include "Cuda/internal.h"
#include "OdometryProvider.h"

#include <opencv2/opencv.hpp>
#include <vector>
#include <vector_types.h>
#include <boost/thread/mutex.hpp>
#include <boost/thread/thread.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <boost/thread/condition_variable.hpp>

class ICPOdometry
{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        ICPOdometry(int width,
                    int height,
                    float cx, float cy, float fx, float fy,
					float depthFactor,
                    float distThresh = 0.10f,
                    float angleThresh = sin(20.f * 3.14159254f / 180.f));

        virtual ~ICPOdometry();

        void initICP(unsigned short * depth, const float depthCutoff);

        void initICPModel(unsigned short * depth, const float depthCutoff, const Eigen::Matrix4f & modelPose);

		void initPlanes(std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f>> planes_prev,
			std::vector<float> planes_lambda_prev,
			std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f>> planes_curr,
			std::vector<std::pair<int, int>> plane_corr_id,
			int plane_inliers_count,
			std::vector<std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>> plane_inliers_curr);

        void getIncrementalTransformation(Eigen::Vector3f & trans, Eigen::Matrix<float, 3, 3, Eigen::RowMajor> & rot,
			Eigen::Vector3f & estimated_trans, Eigen::Matrix<float, 3, 3, Eigen::RowMajor> & estimated_rot, int threads, int blocks);

		void getIncrementalTransformationWithPlane(Eigen::Vector3f & trans, Eigen::Matrix<float, 3, 3, Eigen::RowMajor> & rot,
			Eigen::Vector3f & estimated_trans, Eigen::Matrix<float, 3, 3, Eigen::RowMajor> & estimated_rot, int threads, int blocks);

        Eigen::MatrixXd getCovariance();

		void getVMapCurr(cv::Mat &mat);
		void getNMapCurr(cv::Mat &mat);
		void getPMapCurr(cv::Mat &mat);
		void getPlaneMapCurr(bool *ret);
		void getPlaneMapPrev(bool *ret);

        float lastICPError;
        float lastICPCount;

        Eigen::Matrix<double, 6, 6, Eigen::RowMajor> lastA;
        Eigen::Matrix<double, 6, 1> lastb;

    private:
        std::vector<DeviceArray2D<unsigned short> > depth_tmp;

        std::vector<DeviceArray2D<float> > vmaps_g_prev_;
        std::vector<DeviceArray2D<float> > nmaps_g_prev_;
		std::vector<DeviceArray2D<bool> > planemaps_g_prev_;
		DeviceArray<float> plane_n_prev_;
		DeviceArray<float> plane_d_prev_;
		DeviceArray<float> planes_lambda_prev_;

        std::vector<DeviceArray2D<float> > vmaps_curr_;
        std::vector<DeviceArray2D<float> > nmaps_curr_;
		std::vector<DeviceArray2D<float> > pmaps_curr_;
		std::vector<DeviceArray2D<bool> > planemaps_curr_;
		DeviceArray2D<float> plane_inliers_curr_;
		int plane_count_;

		cv::Mat vmap_curr;
		cv::Mat nmap_curr;
		cv::Mat pmap_curr;

        Intr intr;

        DeviceArray<jtjjtr> sumData;
        DeviceArray<jtjjtr> outData;

        static const int NUM_PYRS = 3;

        std::vector<int> iterations;

        float distThres_;
        float angleThres_;

        Eigen::Matrix<double, 6, 6> lastCov;

        const int width;
        const int height;
        const float cx, cy, fx, fy;
		const float depthFactor_;
};

#endif /* ICPODOMETRY_H_ */
