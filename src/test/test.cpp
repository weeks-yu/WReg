#include "test.h"

#include "PointCloud.h"
#include "pcl/io/pcd_io.h"

#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include "Config.h"
#include "ICPOdometry.h"

using namespace std;

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

int main()
{
	//keyframe_test();
	//something();
	//icp_test();
}