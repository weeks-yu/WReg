#include "test.h"

#include "PointCloud.h"
#include "pcl/io/pcd_io.h"

#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>

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

int main()
{
	keyframe_test();
	//something();
}