#include "srba.h"
#include "test.h"

#include "PointCloud.h"
#include "pcl/io/pcd_io.h"
#include <pcl/common/common_headers.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>

#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include "Config.h"
#include "ICPOdometry.h"

#include "Frame.h"
#include "Transformation.h"

using namespace std;
using namespace srba;

struct RBA_OPTIONS_GRAPH : public RBA_OPTIONS_DEFAULT
{
	//	typedef ecps::local_areas_fixed_size            edge_creation_policy_t;  //!< One of the most important choices: how to construct the relative coordinates graph problem
	//	typedef options::sensor_pose_on_robot_none      sensor_pose_on_robot_t;  //!< The sensor pose coincides with the robot pose
	typedef options::observation_noise_constant_matrix<observations::RelativePoses_3D>   obs_noise_matrix_t;      // The sensor noise matrix is the same for all observations and equal to some given matrix
	//	typedef options::solver_LM_schur_dense_cholesky solver_t;                //!< Solver algorithm (Default: Lev-Marq, with Schur, with dense Cholesky)
};

typedef RbaEngine <
	kf2kf_poses::SE3,               // Parameterization  of KF-to-KF poses
	landmarks::RelativePoses3D,     // Parameterization of landmark positions
	observations::RelativePoses_3D, // Type of observations
	RBA_OPTIONS_GRAPH
> SrbaGraphT;

void draw_feature_point(cv::Mat &img, const vector<cv::KeyPoint> &kp)
{
	for (int i = 0; i < kp.size(); i++)
	{
		cv::circle(img, kp[i].pt, 3, cv::Scalar(0, 255, 0), 2);
	}
}

void keyframe_test()
{
	vector<cv::Mat> rgb;
	vector<vector<cv::KeyPoint>> fpts;
	vector<cv::Mat> descriptors;

	string directory;
	cin >> directory;

	int count;
	cin >> count;

	cv::SURF surf_detector;
	cv::Mat mask;
	for (int i = 0; i < count; i++)
	{
		stringstream ss;
		ss << directory << "\\keyframe_" << i << "_rgb.png";
		cv::Mat img = cv::imread(ss.str());

		vector<cv::KeyPoint> kp;
		cv::Mat dp;
		surf_detector(img, mask, kp, dp);

		draw_feature_point(img, kp);

		rgb.push_back(img);
		fpts.push_back(kp);
		descriptors.push_back(dp);
	}

	cv::namedWindow("feature");
	if (rgb.size() <= 0)
		return;
	int now = 0;
	cv::imshow("feature", rgb[now]);

	while (true)
	{
		int key = cv::waitKey(33);
		if (key == 'a' || key == 'A')
		{
			if (now - 1 >= 0)
			{
				now--;
				cv::imshow("feature", rgb[now]);
			}
		}
		else if (key == 'd' || key == 'D')
		{
			if (now + 1 < count)
			{
				now++;
				cv::imshow("feature", rgb[now]);
			}
		}
	}
}

void something()
{
	PointCloudT::Ptr pc[4];
	std::string name[4];
	name[0] = "F:/1.pcd";
	name[1] = "F:/2.pcd";
	name[2] = "F:/3.pcd";
	name[3] = "F:/4.pcd";

	PointCloudT::Ptr output;

	output = PointCloudT::Ptr(new PointCloudT);
	for (int i = 0; i < 4; i++)
	{
		pc[i] = PointCloudT::Ptr(new PointCloudT);
		pcl::io::loadPCDFile(name[i], *pc[i]);
		*output += *pc[i];
	}

	pcl::io::savePCDFileASCII("F:/output.pcd", *output);

}

void icp_test()
{
	cv::Mat d1 = cv::imread("G:/d1.png", -1);
	cv::Mat d2 = cv::imread("G:/d3.png", -1);

	int width = Config::instance()->get<int>("image_width");
	int height = Config::instance()->get<int>("image_height");
	double cx = Config::instance()->get<double>("camera_cx");
	double cy = Config::instance()->get<double>("camera_cy");
	double fx = Config::instance()->get<double>("camera_fx");
	double fy = Config::instance()->get<double>("camera_fy");
	double depthFactor = Config::instance()->get<double>("depth_factor");
	ICPOdometry *icpcuda = new ICPOdometry(width, height, cx, cy, fx, fy, depthFactor);

	icpcuda->initICPModel((unsigned short *)d1.data, 20.0f, Eigen::Matrix4f::Identity());
	icpcuda->initICP((unsigned short *)d2.data, 20.0f);
	Eigen::Vector3f tra = Eigen::Matrix4f::Identity().topRightCorner(3, 1);
	Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rot = Eigen::Matrix4f::Identity().topLeftCorner(3, 3);

	int threads = Config::instance()->get<int>("icpcuda_threads");
	int blocks = Config::instance()->get<int>("icpcuda_blocks");
	icpcuda->getIncrementalTransformation(tra, rot, threads, blocks);
	Eigen::Matrix4f tran = Eigen::Matrix4f::Identity();
	tran.topLeftCorner(3, 3) = rot;
	tran.topRightCorner(3, 1) = tra;
	double w = icpcuda->lastICPError > 0 ? sqrt(1.0 / icpcuda->lastICPError) : sqrt(1000000);
}

void Ransac_Test()
{
	const int icount = 3;
	std::string rname[icount], dname[icount];
	rname[0] = "E:/lab/test/r1.png";
	rname[1] = "E:/lab/test/r2.png";
	rname[2] = "E:/lab/test/r3.png";

	dname[0] = "E:/lab/test/d1.png";
	dname[1] = "E:/lab/test/d2.png";
	dname[2] = "E:/lab/test/d3.png";

	cv::Mat r[icount], d[icount];
	PointCloudPtr cloud[icount];

	r[0] = cv::imread(rname[0]);
	d[0] = cv::imread(dname[0], -1);
	cloud[0] = ConvertToPointCloudWithoutMissingData(d[0], r[0], 0, 0);

	Frame *f[icount];
	f[0] = new Frame(r[0], d[0], "SURF", Eigen::Matrix4f::Identity());
	f[0]->f.buildFlannIndex();

	for (int i = 1; i < icount; i++)
	{
		r[i] = cv::imread(rname[i]);
		d[i] = cv::imread(dname[i], -1);
		cloud[i] = ConvertToPointCloudWithoutMissingData(d[i], r[i], i, i);

		f[i] = new Frame(r[i], d[i], "SURF", Eigen::Matrix4f::Identity());
		vector<cv::DMatch> matches;
		f[0]->f.findMatchedPairs(matches, f[i]->f, 128, 2);

		Eigen::Matrix4f tran;
		float rmse;
		vector<cv::DMatch> inliers;
		Feature::getTransformationByRANSAC(tran, rmse, &inliers,
			&f[0]->f, &f[i]->f, matches);

		pcl::transformPointCloud(*cloud[i], *cloud[i], tran);

		//test eularangle
		Eigen::Vector3f translation = TranslationFromMatrix4f(tran);
		Eigen::Vector3f eulerAngle = EulerAngleFromMatrix4f(tran);
		
// 		Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitX());
// 		Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitY());
// 		Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitZ());
		Eigen::Affine3f a(tran);
		Eigen::Vector3f ea2 = a.rotation().eulerAngles(0, 1, 2);
		SrbaGraphT::pose_t pose;
		pose.x() = translation(0);
		pose.y() = translation(1);
		pose.z() = translation(2);
		pose.setYawPitchRoll(eulerAngle(0), eulerAngle(1), eulerAngle(2));
		mrpt::math::CQuaternionDouble q;
		pose.getAsQuaternion(q);

		//Eigen::Quaternionf quaternion = QuaternionFromEulerAngle(pose.yaw(), pose.pitch(), pose.roll());
		Eigen::Quaternionf quaternion(q(0), q(1), q(2), q(3));
		Eigen::Matrix4f rt = transformationFromQuaternionsAndTranslation(quaternion, translation);

		if (rt != tran)
		{
			int a = 1;
		}
	}

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("test"));
	viewer->setBackgroundColor(0, 0, 0);
	viewer->addCoordinateSystem(1.0);
	viewer->initCameraParameters();

	for (int i = 0; i < icount; i++)
	{
		pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud[i]);
		viewer->addPointCloud<pcl::PointXYZRGB>(cloud[i], rgb, rname[i]);
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, rname[i]);
	}
	

	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		//boost::this_thread::sleep (boost::posix_time::microseconds (100000));
	}
	
}

int main()
{
	//keyframe_test();
	//something();
	//icp_test();
	Ransac_Test();
}