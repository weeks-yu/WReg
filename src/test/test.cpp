#include "SlamEngine.h"
#include "test.h"

#include "PointCloud.h"
#include "pcl/io/pcd_io.h"
#include <pcl/common/common_headers.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/search/kdtree.h>

#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include "Config.h"
#include "ICPOdometry.h"
#include "ICPSlowdometry.h"
//#include "PointCloudCuda.h"

#include "Frame.h"
#include "Transformation.h"

#include "OniReader.h"
#include "KinectReader.h"

#include "IcpcudaRegister.h"
#include "SiftRegister.h"
#include "SurfRegister.h"
#include "OrbRegister.h"
#include "GeneralizedIcpRegister.h"

using namespace std;

typedef boost::shared_ptr<pcl::visualization::PCLVisualizer> ViewerPtr;

map<int, cv::Mat> rgbs, depths;
map<int, PointCloudPtr> clouds;
map<int, double> timestamps;
int frame_count;
map<int, int> keyframe_id;
vector<int> keyframe_indices;
vector<Frame *> graph;
vector<pair<int, int>> gt_loop;
vector<cv::Mat> gt_loop_corr;
vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> gt_trans;

vector<PointCloudPtr> results;

vector<pair<int, int>> pairs;
vector<float> rmses;
vector<pair<int, int>> matches_and_inliers;
vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> trans;
vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> refined_trans;
int pairs_count, now;
vector<bool> is_keyframe_pose_set;
vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> keyframe_poses;
bool show_refined = false;

vector<PointCloudPtr> downsampled_combined_clouds;
vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> combined_trans;

int plane_ids[640][480];
bool bfs_visited[640][480];

int rgb[20][3] = {
	{ 255, 0, 0 },
	{ 0, 255, 0 },
	{ 0, 0, 255 },
	{ 0, 255, 255 },
	{ 255, 0, 255 },
	{ 255, 255, 0 },
	{ 0, 255, 128 },
	{ 255, 128, 0 },
	{ 128, 0, 255 },
	{ 255, 128, 128 },
	{ 128, 128, 255 },
	{ 0, 0, 128 },
	{ 0, 128, 0 },
	{ 128, 0, 0 },
	{ 128, 128, 0 },
	{ 128, 0, 128 },
	{ 0, 128, 128 },
	{ 64, 0, 0 },
	{ 0, 64, 0 },
	{ 0, 0, 64 }
};

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
	const int icount = 2;
	std::string rname[icount], dname[icount];
	rname[0] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_rpy/rgb/1305031230.865523.png";
	rname[1] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_rpy/rgb/1305031230.925860.png";

	dname[0] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_rpy/depth/1305031230.897394.png";
	dname[1] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_rpy/depth/1305031230.937310.png";

	cv::Mat r[icount], d[icount];
	PointCloudPtr cloud[icount];

	for (int i = 0; i < icount; i++)
	{
		r[i] = cv::imread(rname[i]);
		d[i] = cv::imread(dname[i], -1);
		cloud[i] = ConvertToPointCloudWithoutMissingData(d[i], r[i], i, i);
	}

	ICPOdometry *icpcuda = nullptr;
	int threads = Config::instance()->get<int>("icpcuda_threads");
	int blocks = Config::instance()->get<int>("icpcuda_blocks");
	int width = Config::instance()->get<int>("image_width");
	int height = Config::instance()->get<int>("image_height");
	float cx = Config::instance()->get<float>("camera_cx");
	float cy = Config::instance()->get<float>("camera_cy");
	float fx = Config::instance()->get<float>("camera_fx");
	float fy = Config::instance()->get<float>("camera_fy");
	float depthFactor = Config::instance()->get<float>("depth_factor");
	if (icpcuda == nullptr)
		icpcuda = new ICPOdometry(width, height, cx, cy, fx, fy, depthFactor);

	trans.push_back(Eigen::Matrix4f::Identity());

	for (int i = 1; i < icount; i++)
	{
		cout << i << endl;
		icpcuda->initICPModel((unsigned short *)d[i - 1].data, 20.0f, Eigen::Matrix4f::Identity());
		icpcuda->initICP((unsigned short *)d[i].data, 20.0f);

		Eigen::Matrix4f ret_tran = Eigen::Matrix4f::Identity();
		Eigen::Vector3f ret_t = ret_tran.topRightCorner(3, 1);
		Eigen::Matrix<float, 3, 3, Eigen::RowMajor> ret_rot = ret_tran.topLeftCorner(3, 3);

		icpcuda->getIncrementalTransformation(ret_t, ret_rot, threads, blocks);

		ret_tran.topLeftCorner(3, 3) = ret_rot;
		ret_tran.topRightCorner(3, 1) = ret_t;

		trans.push_back(ret_tran);
	}


	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("test"));
	viewer->setBackgroundColor(0, 0, 0);
	viewer->addCoordinateSystem(1.0);
	viewer->initCameraParameters();

	Eigen::Matrix4f tran = Eigen::Matrix4f::Identity();
	for (int i = 0; i < icount; i++)
	{
		tran = tran * trans[i];
		PointCloudPtr tran_cloud(new PointCloudT);
		pcl::transformPointCloud(*cloud[i], *tran_cloud, tran);

		pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(tran_cloud);
		viewer->addPointCloud<pcl::PointXYZRGB>(tran_cloud, rgb, rname[i]);
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, rname[i]);
	}

	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		//boost::this_thread::sleep (boost::posix_time::microseconds (100000));
	}
}

void Ransac_Test()
{
	const int icount = 2;
	std::string rname[icount], dname[icount];
// 	rname[0] = "G:/kinect data/rgbd_dataset_freiburg1_floor/rgb/1305033527.670034.png";
// 	rname[1] = "G:/kinect data/rgbd_dataset_freiburg1_floor/rgb/1305033566.576146.png";
// 	dname[0] = "G:/kinect data/rgbd_dataset_freiburg1_floor/depth/1305033527.699102.png";
// 	dname[1] = "G:/kinect data/rgbd_dataset_freiburg1_floor/depth/1305033566.605653.png";
	rname[0] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_rpy/rgb/1305031230.865523.png";
	rname[1] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_rpy/rgb/1305031230.925860.png";

	dname[0] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_rpy/depth/1305031230.897394.png";
	dname[1] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_rpy/depth/1305031230.937310.png";

	cv::Mat r[icount], d[icount];
	PointCloudPtr cloud[icount];
	string feature_type = "orb";

	r[0] = cv::imread(rname[0]);
	d[0] = cv::imread(dname[0], -1);
	cloud[0] = ConvertToPointCloudWithoutMissingData(d[0], r[0], 0, 0);

	Frame *f[icount];
	f[0] = new Frame(r[0], d[0], feature_type, Eigen::Matrix4f::Identity());
	if (feature_type != "orb")
		f[0]->f->buildFlannIndex();

	int min_matches = Config::instance()->get<int>("min_matches");
	float inlier_percent = Config::instance()->get<float>("min_inlier_p");
	float inlier_dist = Config::instance()->get<float>("max_inlier_dist");

	for (int i = 1; i < icount; i++)
	{
		r[i] = cv::imread(rname[i]);
		d[i] = cv::imread(dname[i], -1);
		cloud[i] = ConvertToPointCloudWithoutMissingData(d[i], r[i], i, i);

		f[i] = new Frame(r[i], d[i], feature_type, Eigen::Matrix4f::Identity());
		vector<cv::DMatch> matches;
		if (feature_type != "orb")
			f[0]->f->findMatchedPairs(matches, f[i]->f);
		else
			f[0]->f->findMatchedPairsBruteForce(matches, f[i]->f);

		Eigen::Matrix4f tran = Eigen::Matrix4f::Identity();
		float rmse;
		vector<cv::DMatch> inliers;
		Feature::getTransformationByRANSAC(tran, rmse, &inliers,
			f[0]->f, f[i]->f, matches, min_matches, inlier_percent, inlier_dist);
		cout << matches.size() << ", " << inliers.size() << endl;

		pcl::transformPointCloud(*cloud[i], *cloud[i], tran);

		cv::Mat result(480, 1280, CV_8UC3);
		for (int u = 0; u < 480; u++)
		{
			for (int v = 0; v < 640; v++)
			{
				result.at<cv::Vec3b>(u, v) = r[0].at<cv::Vec3b>(u, v);
				result.at<cv::Vec3b>(u, v + 640) = r[1].at<cv::Vec3b>(u, v);
			}
		}

		for (int j = 0; j < matches.size(); j++)
		{
			cv::Point a = f[0]->f->feature_pts[matches[j].trainIdx].pt;
			cv::Point b = cv::Point(f[1]->f->feature_pts[matches[j].queryIdx].pt.x + 640,
				f[1]->f->feature_pts[matches[j].queryIdx].pt.y);
			cv::circle(result, a, 3, cv::Scalar(0, 255, 0), 2);
			cv::circle(result, b, 3, cv::Scalar(0, 255, 0), 2);
			cv::line(result, a, b, cv::Scalar(255, 0, 0));
		}
		cv::imshow("result", result);
		cv::waitKey();
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

void KeyboardEventOccurred(const pcl::visualization::KeyboardEvent &event, void* viewer_void)
{
	ViewerPtr viewer = *static_cast<ViewerPtr *> (viewer_void);
	if (event.getKeySym() == "z" && event.keyDown())
	{
		if (now > /*1*/ 0)
		{
			viewer->removeAllPointClouds();
			now--;

// 			cout << now << endl;
// 			pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(results[now]);
// 			viewer->addPointCloud<pcl::PointXYZRGB>(results[now], rgb, "cloud");
// 			viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");

// 			cout << now - 1 << " " << now << endl;
// 			PointCloudPtr cloud_all(new PointCloudT);
// 			PointCloudPtr tran_cloud(new PointCloudT);
// 			*cloud_all += *clouds[now - 1];
// 			pcl::transformPointCloud(*clouds[now], *tran_cloud, trans[now]);
// 			*cloud_all += *tran_cloud;
// 
// 			pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud_all);
// 			viewer->addPointCloud<pcl::PointXYZRGB>(cloud_all, rgb, "cloud");
// 			viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");

			cout << "pair " << now << ": " << pairs[now].first << "\t" << pairs[now].second << "\t"
				<< rmses[now] << "\t" << matches_and_inliers[now].first << "\t" << matches_and_inliers[now].second << endl;
			show_refined = false;
			pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(clouds[pairs[now].first]);
			stringstream ss;
			ss << pairs[now].first;
			viewer->addPointCloud<pcl::PointXYZRGB>(clouds[pairs[now].first], rgb, ss.str());
			viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, ss.str());

			PointCloudPtr tran_cloud(new PointCloudT);
			pcl::transformPointCloud(*clouds[pairs[now].second], *tran_cloud, trans[now]);
			pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb2(tran_cloud);
			ss << pairs[now].second;
			viewer->addPointCloud<pcl::PointXYZRGB>(tran_cloud, rgb2, ss.str());
			viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, ss.str());
		}
	}
	else if (event.getKeySym() == "x" && event.keyDown())
	{
		if (now < pairs_count - 1)
		{
			viewer->removeAllPointClouds();
 			now++;

// 			cout << now << endl;
// 			pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(results[now]);
// 			viewer->addPointCloud<pcl::PointXYZRGB>(results[now], rgb, "cloud");
// 			viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");

// 			cout << now - 1 << " " << now << endl;
// 			PointCloudPtr cloud_all(new PointCloudT);
// 			PointCloudPtr tran_cloud(new PointCloudT);
// 			*cloud_all += *clouds[now - 1];
// 			pcl::transformPointCloud(*clouds[now], *tran_cloud, trans[now]);
// 			*cloud_all += *tran_cloud;
// 
// 			pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud_all);
// 			viewer->addPointCloud<pcl::PointXYZRGB>(cloud_all, rgb, "cloud");
// 			viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");

			cout << "pair " << now << ": " << pairs[now].first << "\t" << pairs[now].second << "\t"
				<< rmses[now] << "\t" << matches_and_inliers[now].first << "\t" << matches_and_inliers[now].second << endl;
			show_refined = false;
			pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(clouds[pairs[now].first]);
			stringstream ss;
			ss << pairs[now].first;
			viewer->addPointCloud<pcl::PointXYZRGB>(clouds[pairs[now].first], rgb, ss.str());
			viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, ss.str());

			PointCloudPtr tran_cloud(new PointCloudT);
			pcl::transformPointCloud(*clouds[pairs[now].second], *tran_cloud, trans[now]);
			pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb2(tran_cloud);
			ss << pairs[now].second;
			viewer->addPointCloud<pcl::PointXYZRGB>(tran_cloud, rgb2, ss.str());
			viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, ss.str());
		}
	}
	else if (event.getKeySym() == "n" && event.keyDown())
	{
		int id_end;
		cout << "input end frame id: ";
		cin >> id_end;
		viewer->removeAllPointClouds();
		PointCloudPtr cloud_all(new PointCloudT);
		if (id_end > pairs_count - 1)
		{
			id_end = pairs_count - 1;
		}
		for (int i = 0; i < id_end / 50; i++)
		{
			*cloud_all += *downsampled_combined_clouds[i];
		}

		Eigen::Matrix4f tran /*= combined_trans[id_end / 50]*/;
		for (int i = int(id_end / 50) * 50; i <= id_end; i++)
		{
			tran = trans[i];
			PointCloudPtr tran_cloud(new PointCloudT);
			pcl::transformPointCloud(*clouds[i], *tran_cloud, tran);
			*cloud_all += *tran_cloud;
		}
		cloud_all = DownSamplingByVoxelGrid(cloud_all, 0.01, 0.01, 0.01);

		pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud_all);
		viewer->addPointCloud<pcl::PointXYZRGB>(cloud_all, rgb, "cloud");
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");

// 		if (show_refined)
// 		{
// 			show_refined = false;
// 			cout << "showing not refined" << endl;
// 			viewer->removeAllPointClouds();
// 			pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(clouds[pairs[now].first]);
// 			stringstream ss;
// 			ss << pairs[now].first;
// 			viewer->addPointCloud<pcl::PointXYZRGB>(clouds[pairs[now].first], rgb, ss.str());
// 			viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, ss.str());
// 
// 			PointCloudPtr tran_cloud(new PointCloudT);
// 			pcl::transformPointCloud(*clouds[pairs[now].second], *tran_cloud, trans[now]);
// 			pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb2(tran_cloud);
// 			ss << pairs[now].second;
// 			viewer->addPointCloud<pcl::PointXYZRGB>(tran_cloud, rgb2, ss.str());
// 			viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, ss.str());
// 		}
// 		else
// 		{
// 			show_refined = true;
// 			cout << "showing refined" << endl;
// 			viewer->removeAllPointClouds();
// 			pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(clouds[pairs[now].first]);
// 			stringstream ss;
// 			ss << pairs[now].first;
// 			viewer->addPointCloud<pcl::PointXYZRGB>(clouds[pairs[now].first], rgb, ss.str());
// 			viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, ss.str());
// 
// 			PointCloudPtr tran_cloud(new PointCloudT);
// 			pcl::transformPointCloud(*clouds[pairs[now].second], *tran_cloud, refined_trans[now]);
// 			pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb2(tran_cloud);
// 			ss << pairs[now].second;
// 			viewer->addPointCloud<pcl::PointXYZRGB>(tran_cloud, rgb2, ss.str());
// 			viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, ss.str());
// 		}
	}
}

void Ransac_Result_Show()
{
	set<int> cloud_needed;
	ifstream result_infile("E:/123.txt");
	pairs_count = 0;
	int nnn;
	result_infile >> nnn;
	for (int i = 0; i < nnn; i++)
	{
		int base_id, target_id;
		float rmse;
		int match_count, inlier_count;
		Eigen::Matrix4f tran;
		float temp;
		result_infile >> target_id >> base_id >> rmse >> match_count >> inlier_count;
		for (int j = 0; j < 4; j++)
		{
			for (int k = 0; k < 4; k++)
			{
				result_infile >> temp;
				tran(j, k) = temp;
			}
		}

		if (match_count > 0)
		{
			cloud_needed.insert(base_id);
			cloud_needed.insert(target_id);
			pairs.push_back(pair<int, int>(base_id, target_id));
			rmses.push_back(rmse);
			matches_and_inliers.push_back(pair<int, int>(match_count, inlier_count));
			trans.push_back(tran);
			pairs_count++;
		}
	}

	string directory = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_xyz";

	ifstream cloud_infile(directory + "/read.txt");
	string line;
	int k = 0;
	while (getline(cloud_infile, line))
	{
		if (cloud_needed.find(k) != cloud_needed.end())
		{
			int pos = line.find(' ');
			cv::Mat rgb = cv::imread(directory + "/" + line.substr(0, pos));
			cv::Mat depth = cv::imread(directory + "/" + line.substr(pos + 1, line.length() - pos - 1), -1);
			rgbs[k] = rgb;
			depths[k] = depth;
			PointCloudPtr cloud = ConvertToPointCloudWithoutMissingData(depth, rgb, k, k);
			clouds[k] = cloud;

			keyframe_indices.push_back(k);
			keyframe_id[k] = keyframe_indices.size() - 1;
		}
		k++;
	}

	now = 0;
	cout << "pair " << now << ": " << pairs[now].first << "\t" << pairs[now].second << "\t"
		<< rmses[now] << "\t" << matches_and_inliers[now].first << "\t" << matches_and_inliers[now].second << endl;

	ViewerPtr viewer(new pcl::visualization::PCLVisualizer("test"));
	viewer->registerKeyboardCallback(KeyboardEventOccurred, (void*)&viewer);
	viewer->setBackgroundColor(0, 0, 0);
	viewer->addCoordinateSystem(1.0);
	viewer->initCameraParameters();

	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(clouds[pairs[now].first]);
	stringstream ss;
	ss << pairs[now].first;
	viewer->addPointCloud<pcl::PointXYZRGB>(clouds[pairs[now].first], rgb, ss.str());
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, ss.str());

	PointCloudPtr tran_cloud(new PointCloudT);
	pcl::transformPointCloud(*clouds[pairs[now].second], *tran_cloud, trans[now]);
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb2(tran_cloud);
	ss << pairs[now].second;
	viewer->addPointCloud<pcl::PointXYZRGB>(tran_cloud, rgb2, ss.str());
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, ss.str());

	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		//boost::this_thread::sleep (boost::posix_time::microseconds (100000));
	}
}

void Registration_Result_Show()
{
	char ch;

	string filename = "";
	cout << "filename: ";
	scanf("%c", &ch);
	while (filename.length() == 0 || ch != '\n')
	{
		if (ch != '\n' && ch != '\"')
		{
			filename += ch;
		}
		scanf("%c", &ch);
	}
	ifstream input(filename);

	string directory = "";
	cout << "directory: ";
	
	scanf("%c", &ch);
	while (directory.length() == 0 || ch != '\n')
	{
		if (ch != '\n' && ch != '\"')
		{
			directory += ch;
		}
		scanf("%c", &ch);
	}

	stringstream ss;
	ss << directory << "/read.txt";
	ifstream fileInput(ss.str());

	int id_start = -1, id_end = -1, id_interval = 1;

	double timestamp;
	Eigen::Vector3f t;
	Eigen::Quaternionf q;
	input >> timestamp >> t(0) >> t(1) >> t(2) >> q.x() >> q.y() >> q.z() >> q.w();

	int id = 0, k = 0;
	string line;
	getline(input, line);
	while (getline(fileInput, line))
	{
		if (id_start > -1 && id < id_start)
		{
			id++;
			continue;
		}

		if (id_end > -1 && id > id_end)
			break;

		if ((id - id_start > -1 ? id_start : 0) % id_interval != 0)
		{
			id++;
			continue;
		}

		int pos = line.find(' ');
		string rgb_name = line.substr(0, pos);
		string depth_name = line.substr(pos + 1, line.length() - pos - 1);
		int pos2 = depth_name.rfind('/');
		int pos3 = depth_name.rfind('.');
		string timestamp_string = depth_name.substr(pos2 + 1, pos3 - pos2 - 1);
		ss.clear();
		ss.str("");
		ss << timestamp_string;
		double timestamp_this;
		ss >> timestamp_this;

		if (!(fabs(timestamp_this - timestamp) < 1e-4))
		{
			if (timestamp_this < timestamp)
			{
				continue;
			}
			while (!(fabs(timestamp_this - timestamp) < 1e-4) &&
				timestamp_this > timestamp && !input.eof())
			{
				input >> timestamp >> t(0) >> t(1) >> t(2) >> q.x() >> q.y() >> q.z() >> q.w();
			}
		}

		Eigen::Matrix4f tran = transformationFromQuaternionsAndTranslation(q, t);

		ss.clear();
		ss.str("");
		ss << directory << "/" << rgb_name;
		cv::Mat rgb = cv::imread(ss.str());

		ss.clear();
		ss.str("");
		ss << directory << "/" << depth_name;
		cv::Mat depth = cv::imread(ss.str(), -1);

		rgbs[k] = rgb;
		depths[k] = depth;
		PointCloudPtr cloud = ConvertToPointCloudWithoutMissingData(depth, rgb, timestamp_this, k);
		cloud = DownSamplingByVoxelGrid(cloud, 0.01, 0.01, 0.01);
		clouds[k] = cloud;
		trans.push_back(tran);

		id++;
		k++;
	}
	fileInput.close();
	input.close();

	PointCloudPtr cloud_temp(new PointCloudT);
	for (int i = 0; i < k; i++)
	{
		PointCloudPtr tran_cloud(new PointCloudT);
		pcl::transformPointCloud(*clouds[i], *tran_cloud, trans[i]);
		*cloud_temp += *tran_cloud;
		if ((i + 1) % 50 == 0)
		{
			cloud_temp = DownSamplingByVoxelGrid(cloud_temp, 0.01, 0.01, 0.01);
			downsampled_combined_clouds.push_back(cloud_temp);
			cloud_temp = PointCloudPtr(new PointCloudT);
		}
	}
	if (k % 50 != 0)
	{
		cloud_temp = DownSamplingByVoxelGrid(cloud_temp, 0.01, 0.01, 0.01);
		downsampled_combined_clouds.push_back(cloud_temp);
	}
	
	ViewerPtr viewer(new pcl::visualization::PCLVisualizer("test"));
	viewer->registerKeyboardCallback(KeyboardEventOccurred, (void*)&viewer);
	viewer->setBackgroundColor(0, 0, 0);
	viewer->addCoordinateSystem(1.0);
	viewer->initCameraParameters();


	PointCloudPtr cloud_all(new PointCloudT);
	for (int i = 0; i < k / 50; i++)
	{
		*cloud_all += *downsampled_combined_clouds[i];
	}
	cloud_all = DownSamplingByVoxelGrid(cloud_all, 0.01, 0.01, 0.01);

	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud_all);
	viewer->addPointCloud<pcl::PointXYZRGB>(cloud_all, rgb, "cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");

	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		//boost::this_thread::sleep (boost::posix_time::microseconds (100000));
	}
}

bool Compare1(const pair<double, string> &v1, const pair<double, string> &v2)
{
	return v1.first < v2.first;
}
void read_txt()
{
	string directory = "G:/kinect data/rgbd_dataset_freiburg1_xyz";
	stringstream ss;
	string line;
	vector<pair<double, string>> rgbtxt, depthtxt;
	vector<bool> rgbused;

	ss.clear();
	ss.str("");
	ss << directory << "/rgb.txt";
	ifstream in_rgb(ss.str());
	while (getline(in_rgb, line))
	{
		if (line[0] == '#')
			continue;
		ss.clear();
		ss.str("");
		ss << line;
		
		double ts;
		string name;
		ss >> ts >> name;
		rgbtxt.push_back(pair<double, string>(ts, name));
		rgbused.push_back(false);
	}
	sort(rgbtxt.begin(), rgbtxt.end(), Compare1);

	ss.clear();
	ss.str("");
	ss << directory << "/depth.txt";
	ifstream in_depth(ss.str());
	while (getline(in_depth, line))
	{
		if (line[0] == '#')
			continue;
		ss.clear();
		ss.str("");
		ss << line;

		double ts;
		string name;
		ss >> ts >> name;
		depthtxt.push_back(pair<double, string>(ts, name));
	}
	sort(depthtxt.begin(), depthtxt.end(), Compare1);

	double offset, max_difference;
	cout << "offset: ";
	cin >> offset;
	cout << "max difference: ";
	cin >> max_difference;

	vector<pair<string, string>> result_pairs;
	for (int i = 0; i < depthtxt.size(); i++)
	{
		double ts = depthtxt[i].first + offset;
		int k = -1;
		double min_diff = 1000000;
		for (int j = 0; j < rgbtxt.size(); j++)
		{
			if (rgbused[j])
				continue;

			double diff = rgbtxt[j].first - ts;
			if (fabs(diff) <= max_difference)
			{
				if (diff < min_diff)
				{
					k = j;
					min_diff = diff;
				}
			}
			else if (diff > max_difference)
			{
				break;
			}
		}
		if (k != -1)
		{
			rgbused[k] = true;
			result_pairs.push_back(pair<string, string>(rgbtxt[k].second, depthtxt[i].second));
		}
	}

	ss.clear();
	ss.str("");
	ss << directory << "/read.txt";
	ofstream outfile(ss.str());
	for (int i = 0; i < result_pairs.size(); i++)
	{
		outfile << result_pairs[i].first << ' ' << result_pairs[i].second << endl;
	}
	outfile.close();
}

void feature_test()
{
	std::string name;
	cin >> name;
	std::string rname = "G:/kinect data/living_room_1/rgb/" + name + ".jpg";
	std::string dname = "G:/kinect data/living_room_1/depth/" + name + ".png";
	cv::Mat r, d;
	r = cv::imread(rname);
	d = cv::imread(dname, -1);
	Frame *f = new Frame(r, d, "surf", Eigen::Matrix4f::Identity());
	
	cv::Mat result;
	cv::drawKeypoints(r, f->f->feature_pts, result);
	cv::imshow("result", result);
	cv::waitKey();
}

void PlaneFittingTest()
{
	std::string name;
	cout << "rgb image name (without .jpg): ";
	cin >> name;
	std::string rname = "G:/kinect data/living_room_1/rgb/" + name + ".jpg";
	std::string dname = "G:/kinect data/living_room_1/depth/" + name + ".png";
	cv::Mat r, d;
	r = cv::imread(rname);
	d = cv::imread(dname, -1);

	ICPOdometry *icpcuda = nullptr;
	int threads = Config::instance()->get<int>("icpcuda_threads");
	int blocks = Config::instance()->get<int>("icpcuda_blocks");
	int width = Config::instance()->get<int>("image_width");
	int height = Config::instance()->get<int>("image_height");
	float cx = Config::instance()->get<float>("camera_cx");
	float cy = Config::instance()->get<float>("camera_cy");
	float fx = Config::instance()->get<float>("camera_fx");
	float fy = Config::instance()->get<float>("camera_fy");
	float depthFactor = Config::instance()->get<float>("depth_factor");
	if (icpcuda == nullptr)
		icpcuda = new ICPOdometry(width, height, cx, cy, fx, fy, depthFactor);

	icpcuda->initICP((unsigned short *)d.data, 20.f);
	cv::Mat normals;
	icpcuda->getNMapCurr(normals);

	memset(bfs_visited, 0, sizeof(bool) * 480 * 640);

	cv::Mat result;
	r.copyTo(result);
	for (int i = 0; i < 640; i++)
	{
		for (int j = 0; j < 480; j++)
		{
			if (plane_ids[i][j] != -1)
			{
				result.at<cv::Vec3b>(j, i)[0] = rgb[plane_ids[i][j]][0];
				result.at<cv::Vec3b>(j, i)[1] = rgb[plane_ids[i][j]][1];
				result.at<cv::Vec3b>(j, i)[2] = rgb[plane_ids[i][j]][2];
			}
		}
	}

	cv::imshow("result", result);
	cv::waitKey();
}

void continuousPlaneExtractingTest()
{

}

void cudaTest()
{
	const int icount = 2;
	std::string rname[icount], dname[icount];
	rname[0] = "E:/lab/pcl/kinect data/living_room_1/rgb/00590.jpg";
	rname[1] = "E:/lab/pcl/kinect data/living_room_1/rgb/00591.jpg";
// 	rname[2] = "E:/lab/pcl/kinect data/living_room_1/rgb/00592.jpg";
// 	rname[3] = "E:/lab/pcl/kinect data/living_room_1/rgb/00593.jpg";
// 	rname[4] = "E:/lab/pcl/kinect data/living_room_1/rgb/00594.jpg";
// 	rname[5] = "E:/lab/pcl/kinect data/living_room_1/rgb/00595.jpg";

	dname[0] = "E:/lab/pcl/kinect data/living_room_1/depth/00590.png";
	dname[1] = "E:/lab/pcl/kinect data/living_room_1/depth/00591.png";
// 	dname[2] = "E:/lab/pcl/kinect data/living_room_1/depth/00592.png";
// 	dname[3] = "E:/lab/pcl/kinect data/living_room_1/depth/00593.png";
// 	dname[4] = "E:/lab/pcl/kinect data/living_room_1/depth/00594.png";
// 	dname[5] = "E:/lab/pcl/kinect data/living_room_1/depth/00595.png";

	cv::Mat r[icount], d[icount];
	PointCloudPtr cloud[icount];

	for (int i = 0; i < icount; i++)
	{
		r[i] = cv::imread(rname[i]);
		d[i] = cv::imread(dname[i], -1);
		cloud[i] = ConvertToPointCloudWithoutMissingData(d[i], r[i], i, i);
	}

	ICPOdometry *icpcuda = nullptr;
	int threads = Config::instance()->get<int>("icpcuda_threads");
	int blocks = Config::instance()->get<int>("icpcuda_blocks");
	int width = Config::instance()->get<int>("image_width");
	int height = Config::instance()->get<int>("image_height");
	float cx = Config::instance()->get<float>("camera_cx");
	float cy = Config::instance()->get<float>("camera_cy");
	float fx = Config::instance()->get<float>("camera_fx");
	float fy = Config::instance()->get<float>("camera_fy");
	float depthFactor = Config::instance()->get<float>("depth_factor");
	if (icpcuda == nullptr)
		icpcuda = new ICPOdometry(width, height, cx, cy, fx, fy, depthFactor);

	trans.push_back(Eigen::Matrix4f::Identity());

	icpcuda->initICPModel((unsigned short *)d[0].data, 20.0f, Eigen::Matrix4f::Identity());
	clock_t start = clock();
	icpcuda->initICP((unsigned short *)d[1].data, 20.0f);
	cv::Mat vmap, nmap;
	icpcuda->getVMapCurr(vmap);
	icpcuda->getNMapCurr(nmap);
	cout << (clock() - start) / 1000.0 << endl;

	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cc(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
	for (int j = 0; j < 480; j++)
	{
		for (int i = 0; i < 640; i++)
		{
			ushort temp = d[1].at<ushort>(j, i);
			if (temp != 0)
			{
				pcl::PointXYZRGBNormal pt;
				pt.z = ((double)temp) / depthFactor;
				pt.x = (i - cx) * pt.z / fx;
				pt.y = (j - cy) * pt.z / fy;
				pt.b = r[1].at<cv::Vec3b>(j, i)[0];
				pt.g = r[1].at<cv::Vec3b>(j, i)[1];
				pt.r = r[1].at<cv::Vec3b>(j, i)[2];
				pt.normal[0] = -nmap.at<cv::Vec3f>(j, i)[0];
				pt.normal[1] = -nmap.at<cv::Vec3f>(j, i)[1];
				pt.normal[2] = -nmap.at<cv::Vec3f>(j, i)[2];
				cc->push_back(pt);
			}
		}
	}

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	viewer->setBackgroundColor(0, 0, 0);
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal> rgb(cc);
	viewer->addPointCloud<pcl::PointXYZRGBNormal>(cc, rgb, "sample cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
	viewer->addPointCloudNormals<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>(cc, cc, 100, 0.05, "normals");
	viewer->addCoordinateSystem(1.0);
	viewer->initCameraParameters();

	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		//boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}

void Statistics()
{
	const int dcount = 4;
	std::string directories[dcount], names[dcount];
// 	directories[0] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_xyz/";
// 	directories[1] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_desk/";
// 	directories[2] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_room/";
// 	directories[3] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_floor/";
	directories[0] = "G:/kinect data/rgbd_dataset_freiburg1_xyz/";
	directories[1] = "G:/kinect data/rgbd_dataset_freiburg1_desk/";
	directories[2] = "G:/kinect data/rgbd_dataset_freiburg1_room/";
	directories[3] = "G:/kinect data/rgbd_dataset_freiburg1_floor/";
	names[0] = "xyz";
	names[1] = "desk";
	names[2] = "room";
	names[3] = "floor";

	const int fcount = 3;
	std::string types[fcount];
	types[0] = "surf";
	types[1] = "sift";
	types[2] = "orb";

	const int sdcount = 6;
	float dists[sdcount];
	dists[0] = 0.01;
	dists[1] = 0.02;
	dists[2] = 0.03;
	dists[3] = 0.05;
	dists[4] = 0.1;
	dists[5] = 0.2;

	vector<double> timestamps;
	stringstream ss;
	vector<int> gt_corresp;

	vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> result_tran[sdcount][fcount];
	vector<double> result_rmse[sdcount][fcount];

	std::vector<int> pointIdxNKNSearch(1);
	std::vector<float> pointNKNSquaredDistance(1);

	float fx = Config::instance()->get<float>("camera_fx");  // focal length x
	float fy = Config::instance()->get<float>("camera_fy");  // focal length y
	float cx = Config::instance()->get<float>("camera_cx");  // optical center x
	float cy = Config::instance()->get<float>("camera_cy");  // optical center y

	float factor = Config::instance()->get<float>("depth_factor");	// for the 16-bit PNG files
	// OR: factor = 1 # for the 32-bit float images in the ROS bag files

	for (int d = 3; d < dcount; d++)
	{
		rgbs.clear();
		depths.clear();
		timestamps.clear();

		cout << "Comparing " << directories[d] << endl;
		ss.clear();
		ss.str("");
		ss << directories[d] << "read.txt";
		ifstream c_in(ss.str());
		string line;
		int k = 0;
		while (getline(c_in, line))
		{
			int pos = line.find(" depth/");
			if (pos != string::npos)
			{
				cv::Mat rgb = cv::imread(directories[d] + "/" + line.substr(0, pos));
				cv::Mat depth = cv::imread(directories[d] + "/" + line.substr(pos + 1, line.length() - pos - 1), -1);

				rgbs[k] = rgb;
				depths[k] = depth;

				string ts_string = line.substr(pos + 7, line.length() - pos - 11);
				ss.clear();
				ss.str("");
				ss << ts_string;
				double ts;
				ss >> ts;
				timestamps.push_back(ts);

				k++;
			}
		}
		c_in.close();

		for (int sd = 0; sd < sdcount; sd++)
		{
			for (int f = 0; f < fcount; f++)
			{

				//cout << "\t" << types[f] << "\t" << dists[sd] << endl;

				ss.clear();
				ss.str("");
				ss << "G:/results/" << names[d] << "_" << types[f] << "_" << dists[sd] << ".txt";

				ifstream r_in(ss.str());
				result_tran[sd][f].clear();
				result_rmse[sd][f].clear();

				int now_index = 0;

				while (getline(r_in, line))
				{
					if (line[0] == '#')
						continue;
					ss.clear();
					ss.str("");
					ss << line;
					double ts;
					Eigen::Vector3f t;
					Eigen::Quaternionf q;
					Eigen::Matrix4f tran = Eigen::Matrix4f::Identity();
					ss >> ts >> t(0) >> t(1) >> t(2) >> q.x() >> q.y() >> q.z() >> q.w();

					while (fabs(ts - timestamps[now_index]) >= 1e-2 && ts > timestamps[now_index])
					{
						result_tran[sd][f].push_back(tran);
						now_index++;
					}

					if (fabs(ts - timestamps[now_index]) < 1e-2)
					{
						// result found
						tran = transformationFromQuaternionsAndTranslation(q, t);
						now_index++;
					}
					else // ts < timestamps[now_index]
					{
						continue;
					}
					result_tran[sd][f].push_back(tran);
				}
				r_in.close();
			}
		}

		ss.clear();
		ss.str("");
		ss << directories[d] << "groundtruth.txt";
		ifstream g_in(ss.str());

		PointCloudPtr last_cloud = nullptr;
		Eigen::Matrix4f last_tran_inv = Eigen::Matrix4f::Identity();
		Eigen::Matrix4f tran = Eigen::Matrix4f::Identity();

		for (int i = 0; i < k; i++)
		{
//			if (i % 100 == 0)
			{
				cout << "\t" << i << endl;
			}

			PointCloudPtr cloud(new PointCloudT());
			for (int v = 0; v < rgbs[i].size().height; v++)
			{
				for (int u = 0; u < rgbs[i].size().width; u++)
				{
					ushort temp = depths[i].at<ushort>(v, u);
					if (temp > 0)
					{
						PointT pt;
						pt.z = ((double)temp) / factor;
						pt.x = (u - cx) * pt.z / fx;
						pt.y = (v - cy) * pt.z / fy;
						pt.b = rgbs[i].at<cv::Vec3b>(v, u)[0];
						pt.g = rgbs[i].at<cv::Vec3b>(v, u)[1];
						pt.r = rgbs[i].at<cv::Vec3b>(v, u)[2];
						cloud->push_back(pt);
					}
				}
			}
			cloud = DownSamplingByVoxelGrid(cloud, 0.01, 0.01, 0.01);

			if (i > 0)
			{
				// ground-truth
				bool gt_found = false;
				while (getline(g_in, line))
				{
					if (line[0] == '#')
						continue;
					ss.clear();
					ss.str("");
					ss << line;
					double ts;
					Eigen::Vector3f t;
					Eigen::Quaternionf q;
					ss >> ts >> t(0) >> t(1) >> t(2) >> q.x() >> q.y() >> q.z() >> q.w();
					if (fabs(ts - timestamps[i]) < 1e-2)
					{
						// ground-truth found
						gt_found = true;
						tran = transformationFromQuaternionsAndTranslation(q, t);
						break;
					}
					else if (ts > timestamps[i])
					{
						// ground-truth not found
						break;
					}
				}

				if (!gt_found)
				{
					continue;
				}
				
				// compare
				for (int sd = 0; sd < sdcount; sd++)
				{
					// ground-truth correspondence
					PointCloudPtr cloud_gt(new PointCloudT);
					pcl::transformPointCloud(*cloud, *cloud_gt, last_tran_inv * tran);
					gt_corresp.clear();

					pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
					tree->setInputCloud(last_cloud);

					for (int j = 0; j < cloud->size(); j++)
					{
						tree->nearestKSearch(cloud->at(j), 1, pointIdxNKNSearch, pointNKNSquaredDistance);
						if (pointIdxNKNSearch.size() > 0 &&
							pointNKNSquaredDistance[0] < dists[sd])
						{
							gt_corresp.push_back(pointIdxNKNSearch[0]);
						}
						else
						{
							gt_corresp.push_back(-1);
						}
					}

					// compare
					for (int f = 0; f < fcount; f++)
					{
						//cout << "\t" << types[f] << "\t" << dists[sd] << endl;
						PointCloudPtr cloud_rt(new PointCloudT);
						pcl::transformPointCloud(*cloud, *cloud_rt, result_tran[sd][f][i]);

						double rmse = 0.0;
						int count = 0;
						for (int j = 0; j < cloud->size(); j++)
						{
							if (gt_corresp[j] != -1)
							{
								Eigen::Vector3f a(cloud_rt->at(j).x, cloud_rt->at(j).y, cloud_rt->at(j).z);
								Eigen::Vector3f b(last_cloud->at(gt_corresp[j]).x, last_cloud->at(gt_corresp[j]).y, last_cloud->at(gt_corresp[j]).z);
								rmse += (a - b).squaredNorm();
								count++;
							}
						}
						rmse /= count;
						result_rmse[sd][f].push_back(rmse);
					}
				}
			}

			last_cloud = cloud;
			last_tran_inv = tran.inverse();
		}

		cout << "results" << endl;
		for (int sd = 0; sd < sdcount; sd++)
		{
			for (int f = 0; f < fcount; f++)
			{
				double rmse_all = 0.0;
				for (int j = 0; j < result_rmse[sd][f].size(); j++)
				{
					rmse_all += result_rmse[sd][f][j];
				}
				cout << "\t" << types[f] << "\t" << dists[sd] << "\t" << rmse_all / result_rmse[sd][f].size() << endl;
			}
		}
	}
	int nnn;
	cin >> nnn;
}

void getClouds()
{
	cout << "Converting images to clouds" << endl;
	clouds.clear();
	for (int i = 0; i < frame_count; i++)
	{
		PointCloudPtr cloud(new PointCloudT);
		cloud = ConvertToPointCloudWithoutMissingData(depths[i], rgbs[i], timestamps[i], i);
		cloud = DownSamplingByVoxelGrid(cloud, 0.01, 0.01, 0.01);
		clouds[i] = cloud;
	}
}

void readData(string directory, int st = -1, int ed = -1, bool convert = true)
{
	cout << "Reading data from " << directory << endl;
	stringstream ss;
	ss.clear();
	ss.str("");
	ss << directory << "/read.txt";
	ifstream c_in(ss.str());
	string line;

	rgbs.clear();
	depths.clear();
	timestamps.clear();
	int k = 0;
	frame_count = 0;
	while (getline(c_in, line))
	{
		if (st != -1 && k < st)
		{
			k++;
			continue;
		}
		if (ed != -1 && k > ed)
		{
			break;
		}
		int pos = line.find(" depth/");
		if (pos != string::npos)
		{
			cv::Mat rgb = cv::imread(directory + "/" + line.substr(0, pos));
			cv::Mat depth = cv::imread(directory + "/" + line.substr(pos + 1, line.length() - pos - 1), -1);

			rgbs[frame_count] = rgb;
			depths[frame_count] = depth;

			string ts_string = line.substr(pos + 7, line.length() - pos - 11);
			ss.clear();
			ss.str("");
			ss << ts_string;
			double ts;
			ss >> ts;
			timestamps[frame_count] = ts;

			frame_count++;
			k++;
		}
	}
	c_in.close();
	cout << "Reading completed.";
	if (convert) getClouds();
}

void ShowPairwiseResults(ofstream *out = nullptr)
{
	bool save = out != nullptr;
	Eigen::Vector3f t;
	Eigen::Quaternionf q;

	PointCloudPtr cloud_all(new PointCloudT);
	*cloud_all += *clouds[0];

	graph[0]->tran = graph[0]->relative_tran;
	if (save)
	{
		t = TranslationFromMatrix4f(graph[0]->tran);
		q = QuaternionFromMatrix4f(graph[0]->tran);

		*out << fixed << setprecision(6) << timestamps[0]
			<< ' ' << t(0) << ' ' << t(1) << ' ' << t(2)
			<< ' ' << q.x() << ' ' << q.y() << ' ' << q.z() << ' ' << q.w() << endl;
	}
	Eigen::Matrix4f last_tran = graph[0]->tran;
	int k = 1;
	for (int i = 1; i < frame_count; i++)
	{
		graph[i]->tran = last_tran * graph[i]->relative_tran;

		if (save)
		{
			t = TranslationFromMatrix4f(graph[i]->tran);
			q = QuaternionFromMatrix4f(graph[i]->tran);

			*out << fixed << setprecision(6) << timestamps[i]
				<< ' ' << t(0) << ' ' << t(1) << ' ' << t(2)
				<< ' ' << q.x() << ' ' << q.y() << ' ' << q.z() << ' ' << q.w() << endl;
		}

		PointCloudPtr cloud_tran(new PointCloudT);
		pcl::transformPointCloud(*clouds[i], *cloud_tran, graph[i]->tran);
		*cloud_all += *cloud_tran;

		if (k < keyframe_indices.size() && keyframe_indices[k] == i)
		{
			last_tran = graph[i]->tran;
			k++;
		}
	}

	if (save)
	{
		out->close();
	}
	cloud_all = DownSamplingByVoxelGrid(cloud_all, 0.01, 0.01, 0.01);

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	viewer->setBackgroundColor(0, 0, 0);
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud_all);
	viewer->addPointCloud<pcl::PointXYZRGB>(cloud_all, rgb, "cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud");
	viewer->addCoordinateSystem(1.0);
	viewer->initCameraParameters();

	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
	}
}

void PairwiseRegistration(string feature_type,
	ofstream *out = nullptr, ofstream *gout = nullptr, bool verbose = true, ofstream *log = nullptr)
{
	vector<int> failed_pairwise;
	bool save = out != nullptr;
	bool save_global = gout != nullptr;
	bool save_log = log != nullptr;
	
	int min_matches = Config::instance()->get<int>("min_matches");
	float inlier_percent = Config::instance()->get<float>("min_inlier_p");
	float inlier_dist = Config::instance()->get<float>("max_inlier_dist");

	int threads = Config::instance()->get<int>("icpcuda_threads");
	int blocks = Config::instance()->get<int>("icpcuda_blocks");
	int width = Config::instance()->get<int>("image_width");
	int height = Config::instance()->get<int>("image_height");
	float cx = Config::instance()->get<float>("camera_cx");
	float cy = Config::instance()->get<float>("camera_cy");
	float fx = Config::instance()->get<float>("camera_fx");
	float fy = Config::instance()->get<float>("camera_fy");
	float depthFactor = Config::instance()->get<float>("depth_factor");
	float distThresh = Config::instance()->get<float>("dist_threshold");
	float angleThresh = Config::instance()->get<float>("angle_threshold");
	ICPOdometry *icpcuda = new ICPOdometry(width, height, cx, cy, fx, fy, depthFactor, distThresh, angleThresh);
	float keyframe_rational = Config::instance()->get<float>("keyframe_rational");

	PairwiseRegister *reg = nullptr;
	if (feature_type == "icpcuda")
	{
		reg = new IcpcudaRegister(icpcuda, threads, blocks, 20.f);
	}
	else if (feature_type == "sift")
	{
		reg = new SiftRegister(min_matches, inlier_percent, inlier_dist);
	}
	else if (feature_type == "orb")
	{
		reg = new OrbRegister(min_matches, inlier_percent, inlier_dist);
	}
	else if (feature_type == "surf")
	{
		reg = new SurfRegister(min_matches, inlier_percent, inlier_dist);
	}

	for (int i = 0; i < graph.size(); i++)
	{
		if (graph[i])
		{
			delete graph[i];
		}
	}
	graph.clear();

	Frame *last_frame;
	Eigen::Matrix4f last_tran;
	cv::Mat last_depth;

	float total_fe = 0;
	float total_ct = 0;
	clock_t start;
	for (int i = 0; i < frame_count; i++)
	{
		if (verbose)
			cout << "Frame " << i;

		if (feature_type == "icpcuda")
		{
			Frame *frame = new Frame();
			float ct = 0;
			if (i == 0)
			{
				frame->relative_tran = Eigen::Matrix4f::Identity();
				last_tran = Eigen::Matrix4f::Identity();
			}
			else
			{
				Eigen::Matrix4f tran = Eigen::Matrix4f::Identity();
				start = clock();
				reg->getTransformation(last_depth.data, depths[i].data, tran);
				ct = (clock() - start) / 1000.0;
				frame->relative_tran = tran;
				last_tran = tran;
			}
			depths[i].copyTo(last_depth);
			last_frame = frame;
			graph.push_back(frame);

			total_ct += ct;
			if (save)
			{
				*out << i << " fe:0 ct:" << ct << endl;
				*out << frame->relative_tran << endl;
			}
		}
		else if (feature_type == "sift" || feature_type == "surf" || feature_type == "orb")
		{
			start = clock();
			Frame *frame = new Frame(rgbs[i], depths[i], feature_type, Eigen::Matrix4f::Identity());
			float fe = (clock() - start) / 1000.0;
			float ct = 0;
			if (i == 0)
			{
				frame->relative_tran = Eigen::Matrix4f::Identity();
				if (feature_type != "orb")
				{
					frame->f->buildFlannIndex();
				}
				last_tran = Eigen::Matrix4f::Identity();
			}
			else
			{
				Eigen::Matrix4f tran = Eigen::Matrix4f::Identity();
				start = clock();
				if (reg->getTransformation(last_frame, frame, tran))
				{
					frame->relative_tran = tran;
				}
				else
				{
					frame->relative_tran = last_tran;
				}
				last_tran = frame->relative_tran;

// 				delete last_frame->f;
// 				last_frame->f = nullptr;
				if (feature_type != "orb")
					frame->f->buildFlannIndex();
				ct = (clock() - start) / 1000.0;
			}

			last_frame = frame;
			graph.push_back(frame);

			total_ct += ct;
			total_fe += fe;
			if (save)
			{
				*out << i << " fe:" << fe << " ct:" << ct << endl;
				*out << frame->relative_tran << endl;
			}
		}
	}

	if (save)
	{
		*out << "-1" << endl;
		*out << "total_fe:" << total_fe << " total_ct:" << total_ct << endl;
		*out << "avg_fe:" << total_fe / graph.size() << " avg_ct:" << total_ct / graph.size() << endl;
		out->close();
	}

	if (save_global)
	{
		Eigen::Matrix4f last_tran = Eigen::Matrix4f::Identity();
		for (int j = 0; j < frame_count; j++)
		{
			graph[j]->tran = last_tran * graph[j]->relative_tran;

			Eigen::Vector3f t = TranslationFromMatrix4f(graph[j]->tran);
			Eigen::Quaternionf q = QuaternionFromMatrix4f(graph[j]->tran);

			*gout << fixed << setprecision(6) << timestamps[j]
				<< ' ' << t(0) << ' ' << t(1) << ' ' << t(2)
				<< ' ' << q.x() << ' ' << q.y() << ' ' << q.z() << ' ' << q.w() << endl;

			last_tran = graph[j]->tran;
		}
		gout->close();
	}

	delete icpcuda;
	delete reg;
}

void readPairwiseResult(string filename, string feature_type = "surf")
{
	cout << "Reading pairwise result from file: " << filename << endl;
	ifstream in(filename);
	string kf, rf;
	float tmp;

	keyframe_indices.clear();
	keyframe_id.clear();
	graph.clear();
	for (int i = 0; i < frame_count; i++)
	{
		cout << "Frame " << i;
		int id;
		bool keyframe = false;
		Eigen::Matrix4f tran;
		in >> id >> kf >> rf;
		if (kf == "true")
		{
			keyframe = true;
		}

		for (int u = 0; u < 4; u++)
		{
			for (int v = 0; v < 4; v++)
			{
				in >> tmp;
				tran(u, v) = tmp;
			}
		}
		//cout << tran << endl;
		Frame *frame;
		if (keyframe)
		{
			cout << ", Keyframe";
			frame = new Frame();
			keyframe_indices.push_back(i);
			keyframe_id[i] = keyframe_indices.size() - 1;
		}
		else
		{
			frame = new Frame();
		}
		frame->relative_tran = tran;
		if (rf == "true")
			frame->ransac_failed = true;
		else
			frame->ransac_failed = false;
		graph.push_back(frame);
		cout << endl;
	}
	in.close();
}

void ShowPairwiseResultsEachKeyframe()
{
	//test
// 	pairs_count = 0;
// 	for (int i = 76; i < 80; i++)
// 	{
// 		PointCloudPtr cloud(new PointCloudT);
// 		*cloud += *clouds[75];
// 		PointCloudPtr cloud_tran(new PointCloudT);
// 		pcl::transformPointCloud(*clouds[i], *cloud_tran, graph[i]->relative_tran);
// 		*cloud += *cloud_tran;
// 		results.push_back(cloud);
// 		pairs_count++;
// 	}

	PointCloudPtr cloud(new PointCloudT);
	int k = 0;
	pairs_count = 0;
	Eigen::Matrix4f tran = Eigen::Matrix4f::Identity();

	for (int i = 1; i <= keyframe_indices.size(); i++)
	{
		while (k < frame_count && (i == keyframe_indices.size() || k < keyframe_indices[i]))
		{
			PointCloudPtr cloud_tran(new PointCloudT);
			pcl::transformPointCloud(*clouds[k], *cloud_tran, tran * graph[k]->relative_tran);
			*cloud += *cloud_tran;
			k++;
		}

		if (k >= frame_count || (i != keyframe_indices.size() && k >= keyframe_indices[i]))
		{
			results.push_back(cloud);
			pairs_count++;
			cloud = PointCloudPtr(new PointCloudT);
			if (k < frame_count)
			{
				tran = tran * graph[k]->relative_tran;
				PointCloudPtr cloud_tran(new PointCloudT);
				pcl::transformPointCloud(*clouds[k], *cloud_tran, tran);
				k++;
				*cloud += *cloud_tran;
			}
		}
	}
	
	now = 0;
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	viewer->setBackgroundColor(0, 0, 0);
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(results[now]);
	viewer->registerKeyboardCallback(KeyboardEventOccurred, (void*)&viewer);
	viewer->addPointCloud<pcl::PointXYZRGB>(results[now], rgb, "cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");
	viewer->addCoordinateSystem(1.0);
	viewer->initCameraParameters();

	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
	}
}

void GlobalRegistration(string graph_ftype = "surf",
	ofstream *result_out = nullptr,
	ofstream *log_out = nullptr)
{
	bool save_result = result_out != nullptr;
	bool save_log = log_out != nullptr;

	// find keyframes
	keyframe_indices.clear();
	keyframe_id.clear();

	int min_matches = Config::instance()->get<int>("min_matches");
	float inlier_percentage = Config::instance()->get<float>("min_inlier_p");
	float inlier_dist = Config::instance()->get<float>("max_inlier_dist");

	PairwiseRegister *reg = new SurfRegister(min_matches, inlier_percentage, inlier_dist);

	int accumulated_frame_count = 0, last_keyframe_id;
	Eigen::Matrix4f accumulated_transformation = Eigen::Matrix4f::Identity();
	bool is_last_frame_keyframe = false;
	float last_rational = 1;

	for (int i = 0; i < frame_count; i++)
	{
		bool isKeyframe = false;
		if (i == 0)
		{
			isKeyframe = true;
		}
		else
		{
			accumulated_frame_count++;
			accumulated_transformation = accumulated_transformation * trans[i];

			float rrr = reg->getCorrespondencePercent(graph[last_keyframe_id], graph[i], accumulated_transformation);
			if (is_last_frame_keyframe)
			{
				last_rational = rrr;
			}
			rrr /= last_rational;
			if (rrr < Config::instance()->get<float>("keyframe_rational"))
			{
				isKeyframe = true;
			}
		}

		if (accumulated_frame_count >= Config::instance()->get<int>("max_keyframe_interval"))
		{
			isKeyframe = true;
		}

		graph[i]->relative_tran = accumulated_transformation;
		if (isKeyframe)
		{
			accumulated_frame_count = 0;
			accumulated_transformation = Eigen::Matrix4f::Identity();
			last_keyframe_id = i;
			keyframe_indices.push_back(i);
			keyframe_id.insert(pair<int, int>(i, keyframe_indices.size() - 1));
			is_last_frame_keyframe = true;
		}
		else
		{
			is_last_frame_keyframe = false;
		}
	}

	// graph optimization
	double total_time = 0.0, time;
	clock_t start;

	cout << "Global registration" << endl;
 	SlamEngine engine;
	engine.setGraphRegister("surf");
	engine.setGraphManager("robust");

	min_matches = Config::instance()->get<int>("graph_min_matches");
	inlier_percentage = Config::instance()->get<float>("graph_min_inlier_p");
	inlier_dist = Config::instance()->get<float>("graph_max_inlier_dist");
	engine.setGraphParametersFeature(min_matches, inlier_percentage, inlier_dist);

	for (int i = 0; i < frame_count; i++)
	{
		bool isKeyframe = false;
		if (keyframe_id.find(i) != keyframe_id.end())
		{
			isKeyframe = true;
		}
		start = clock();
		engine.AddGraph(graph[i], clouds[i], isKeyframe, timestamps[i]);
		time = clock() - start;
		total_time += time;
	}

	if (save_result)
	{
		vector<pair<double, Eigen::Matrix4f>> transformations = engine.GetTransformations();
		for (int i = 0; i < transformations.size(); i++)
		{
			Eigen::Vector3f t = TranslationFromMatrix4f(transformations[i].second);
			Eigen::Quaternionf q = QuaternionFromMatrix4f(transformations[i].second);

			*result_out << fixed << setprecision(6) << transformations[i].first
				<< ' ' << t(0) << ' ' << t(1) << ' ' << t(2)
				<< ' ' << q.x() << ' ' << q.y() << ' ' << q.z() << ' ' << q.w() << endl;
		}
		result_out->close();
	}

	if (save_log)
	{
		RobustManager *manager = static_cast<RobustManager *>(engine.graph_manager);

		*log_out << "Keyframe number " << keyframe_indices.size() << endl;
		*log_out << "LC candidate " << manager->keyframe_for_lc.size() << endl;
		*log_out << "total time " << total_time << endl;
		*log_out << "total time per frame " << total_time / frame_count << endl;
		*log_out << "kdtree build " << manager->total_kdtree_build << endl;
		*log_out << "kdtree build per frame " << manager->total_kdtree_build / frame_count << endl;
		*log_out << "kdtree match " << manager->total_kdtree_match << endl;
		*log_out << "kdtree match per frame " << manager->total_kdtree_match / frame_count << endl;
		*log_out << "loop ransac " << manager->total_loop_ransac << endl;
		*log_out << "loop ransac per frame " << manager->total_loop_ransac / frame_count << endl;
		*log_out << "graph " << manager->total_graph_opt_time << endl;
		*log_out << "graph per frame " << manager->total_graph_opt_time / frame_count << endl;
		log_out->close();
	}
}
// 
// void LoopAnalysis()
// {
// 	if (keyframe_indices.size() <= 0)
// 		return;
// 
// 	Eigen::Matrix4f first_inv = gt_trans[0].inverse();
// 	vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> r_gt_trans;
// 	r_gt_trans.push_back(Eigen::Matrix4f::Identity());
// 
// 	for (int i = 1; i < keyframe_indices.size(); i++)
// 	{
// 		r_gt_trans.push_back(first_inv * gt_trans[i]);
// 	}
// 
// 	vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> keyframe_poses;
// 	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
// 	stringstream ss;
// 	PointCloudPtr cloud(new PointCloudT);
// 	for (int i = 0; i < keyframe_indices.size(); i++)
// 	{
// 		pcl::PointXYZRGB pt;
// 		Eigen::Vector3f translation = TranslationFromMatrix4f(r_gt_trans[i]);
// 		keyframe_poses.push_back(translation);
// 		pt.x = translation(0) + 0.5;
// 		pt.y = translation(1) + 0.5;
// 		pt.z = translation(2) + 0.55;
// 		pt.r = 0;
// 		pt.g = 255;
// 		pt.b = 0;
// 		pt.a = 255;
// 		ss.clear();
// 		ss.str("");
// 		ss << i << " " << keyframe_indices[i];
// 		viewer->addText3D(ss.str(), pt, 0.01);
// 		pt.z = translation(2) + 0.5;
// 		cloud->push_back(pt);
// 	}
// 
// 	cout << "dists: " << endl;
// 	for (int i = 0; i < gt_loop.size(); i++)
// 	{
// 		Eigen::Vector3f a = keyframe_poses[gt_loop[i].first];
// 		Eigen::Vector3f b = keyframe_poses[gt_loop[i].second];
// 		cout << gt_loop[i].first << " " << gt_loop[i].second << " " << (a - b).norm() << endl;
// 	}
// 
// 	viewer->setBackgroundColor(0, 0, 0);
// 	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
// 	//viewer->registerKeyboardCallback(KeyboardEventOccurred, (void*)&viewer);
// 	viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "cloud");
// 	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "cloud");
// 	viewer->addCoordinateSystem(1.0);
// 	viewer->initCameraParameters();
// 
// 	while (!viewer->wasStopped())
// 	{
// 		viewer->spinOnce(100);
// 	}
// }
// 
// void GlobalWithAll(ofstream *result_out = nullptr, ofstream *log_out = nullptr)
// {
// 	bool save_result = result_out != nullptr;
// 	bool save_log = log_out != nullptr;
// 
// 	cout << "Global registration with gt loop(while transformation estimated by ransac)" << endl;
// 	SlamEngine engine;
// 	engine.setUsingRobustOptimizer(true);
// 
// 	for (int i = 0; i < keyframe_indices.size(); i++)
// 	{
// 		graph[keyframe_indices[i]]->f->buildFlannIndex();
// 	}
// 
// 	for (int i = 0; i < frame_count; i++)
// 	{
// 		bool keyframe = false;
// 		if (keyframe_id.find(i) != keyframe_id.end())
// 		{
// 			keyframe = true;
// 		}
// 		vector<int> loop;
// 		if (keyframe)
// 		{
// 			for (int j = 0; j < gt_loop.size(); j++)
// 			{
// 				if (gt_loop[j].second == keyframe_id[i])
// 				{
// 					loop.push_back(keyframe_indices[gt_loop[j].first]);
// 				}
// 			}
// 		}
// 		engine.AddGraph(graph[i], clouds[i], keyframe, true, loop, timestamps[i]);
// 	}
// 
// 	if (save_result)
// 	{
// 		vector<pair<double, Eigen::Matrix4f>> transformations = engine.GetTransformations();
// 		for (int i = 0; i < transformations.size(); i++)
// 		{
// 			Eigen::Vector3f t = TranslationFromMatrix4f(transformations[i].second);
// 			Eigen::Quaternionf q = QuaternionFromMatrix4f(transformations[i].second);
// 
// 			*result_out << fixed << setprecision(6) << transformations[i].first
// 				<< ' ' << t(0) << ' ' << t(1) << ' ' << t(2)
// 				<< ' ' << q.x() << ' ' << q.y() << ' ' << q.z() << ' ' << q.w() << endl;
// 		}
// 		result_out->close();
// 	}
// 
// 	cout << "Getting scene" << endl;
// 	PointCloudPtr cloud = engine.GetScene();
// 	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
// 	viewer->setBackgroundColor(0, 0, 0);
// 	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
// 	//viewer->registerKeyboardCallback(KeyboardEventOccurred, (void*)&viewer);
// 	viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "cloud");
// 	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");
// 	viewer->addCoordinateSystem(1.0);
// 	viewer->initCameraParameters();
// 
// 	while (!viewer->wasStopped())
// 	{
// 		viewer->spinOnce(100);
// 	}
// }

void pairwise_results()
{
	const int dcount = 9;
	std::string directories[dcount], names[dcount];
	directories[0] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_xyz/";
	directories[1] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_desk/";
	directories[2] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_desk2/";
	directories[3] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_360/";
	directories[4] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_room/";
	directories[5] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_floor/";
	directories[6] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_rpy";
	directories[7] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_teddy/";
	directories[8] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_plant/";

	names[0] = "xyz";
	names[1] = "desk";
	names[2] = "desk2";
	names[3] = "360";
	names[4] = "room";
	names[5] = "floor";
	names[6] = "rpy";
	names[7] = "teddy";
	names[8] = "plant";

	const int fcount = 4;
	std::string ftypes[fcount];
	ftypes[0] = "surf";
	ftypes[1] = "sift";
	ftypes[2] = "orb";
	ftypes[3] = "icpcuda";

	const int sdcount = 8;
	float dists[sdcount];
	dists[0] = 0.01;
	dists[1] = 0.02;
	dists[2] = 0.03;
	dists[3] = 0.05;
	dists[4] = 0.08;
	dists[5] = 0.1;
	dists[6] = 0.2;
	dists[7] = 0.3;

	int st = -1, ed = -1;
	stringstream ss;

	for (int dd = 0; dd < dcount; dd++)
	{
		readData(directories[dd], st, ed);

		for (int sd = 0; sd < sdcount; sd++)
		{
			cout << "\t dists: " << dists[sd];
			Config::instance()->set<float>("max_inlier_dist", dists[sd]);
			Config::instance()->set<float>("dist_threshold", dists[sd]);

			for (int f = 0; f < fcount; f++)
			{
				cout << "\t" << ftypes[f];
				ss.clear();
				ss.str("");
				ss << "E:/tempresults2/" << names[dd] << "/"
					<< names[dd] << "_ftof_" << ftypes[f] << "_" << dists[sd] << ".txt";
				ofstream out1(ss.str());

				ss.clear();
				ss.str("");
				ss << "E:/tempresults2/" << names[dd] << "/"
					<< names[dd] << "_" << ftypes[f] << "_" << dists[sd] << ".txt";
				ofstream out2(ss.str());

				PairwiseRegistration(ftypes[f], &out1, &out2, false);
			}
			cout << endl;
		}
	}
}

void repeat_pairwise_results()
{
	const int dcount = 9;
	std::string directories[dcount], names[dcount];
	directories[0] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_xyz/";
	directories[1] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_desk/";
	directories[2] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_desk2/";
	directories[3] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_360/";
	directories[4] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_room/";
	directories[5] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_floor/";
	directories[6] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_rpy";
 	directories[7] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_teddy/";
 	directories[8] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_plant/";

	names[0] = "xyz";
	names[1] = "desk";
	names[2] = "desk2";
	names[3] = "360";
	names[4] = "room";
	names[5] = "floor";
 	names[6] = "rpy";
 	names[7] = "teddy";
 	names[8] = "plant";

	const int fcount = 4;
	std::string ftypes[fcount];
 	ftypes[0] = "surf";
	ftypes[1] = "sift";
	ftypes[2] = "orb";
	ftypes[3] = "icpcuda";

	float dists[dcount] = {0};
	dists[0] = 0.05;	// xyz
	dists[1] = 0.02;	// desk
	dists[2] = 0.02;	// desk2
	dists[3] = 0.08;	// 360
	dists[4] = 0.02;	// room
	dists[5] = 0.08;	// floor
	dists[6] = 0.01;	// rpy
	dists[7] = 0.2;		// teddy
	dists[8] = 0.2;		// plants

	int repeat_time = 5;
	int st = -1, ed = -1;
	stringstream ss;

	for (int dd = 0; dd < dcount; dd++)
	{
		readData(directories[dd], st, ed, false);

		Config::instance()->set<float>("max_inlier_dist", dists[dd]);
		Config::instance()->set<float>("dist_threshold", dists[dd]);

		for (int fd = 0; fd < fcount; fd++)
		{
			cout << "\tfeature: " << ftypes[fd] << " dists: " << dists[dd];

			for (int i = 0; i < repeat_time; i++)
			{
				ss.clear();
				ss.str("");
				ss << "E:/tempresults2/" << names[dd] << "/"
					<< names[dd] << "_repeat_" << ftypes[fd] << "_ftof_" << i << ".txt";
				ofstream out1(ss.str());

				ss.clear();
				ss.str("");
				ss << "E:/tempresults2/" << names[dd] << "/"
					<< names[dd] << "_repeat_" << ftypes[fd] << "_" << i << ".txt";
				ofstream out2(ss.str());

				PairwiseRegistration(ftypes[fd], &out1, &out2, false);
			}
		}
		cout << endl;
	}
}

void repeat_global_results()
{
	const int dcount = 9;
	std::string directories[dcount], names[dcount], tnames[dcount];
	directories[0] = "E:/school/data/rgbd_dataset_freiburg1_xyz/";
	directories[1] = "E:/school/data/rgbd_dataset_freiburg1_desk/";
	directories[2] = "E:/school/data/rgbd_dataset_freiburg1_desk2/";
	directories[3] = "E:/school/data/rgbd_dataset_freiburg1_360/";
	directories[4] = "E:/school/data/rgbd_dataset_freiburg1_room/";
	directories[5] = "E:/school/data/rgbd_dataset_freiburg1_floor/";
	directories[6] = "E:/school/data/rgbd_dataset_freiburg1_rpy";
	directories[7] = "E:/school/data/rgbd_dataset_freiburg1_teddy/";
	directories[8] = "E:/school/data/rgbd_dataset_freiburg1_plant/";

	names[0] = "xyz";
	names[1] = "desk";
	names[2] = "desk2";
	names[3] = "360";
	names[4] = "room";
	names[5] = "floor";
	names[6] = "rpy";
	names[7] = "teddy";
	names[8] = "plant";

	for (int i = 0; i < dcount; i++)
	{
		tnames[i] = "E:/school/bestresults2/" + names[i] + "/" + names[i] + "_ftof_surf.txt";
	}

	float dists_pair[dcount] = { 0 };
	dists_pair[0] = 0.05;	// xyz
	dists_pair[1] = 0.02;	// desk
	dists_pair[2] = 0.02;	// desk2
	dists_pair[3] = 0.08;	// 360
	dists_pair[4] = 0.02;	// room
	dists_pair[5] = 0.08;	// floor
	dists_pair[6] = 0.01;	// rpy
	dists_pair[7] = 0.2;	// teddy
	dists_pair[8] = 0.2;	// plants

//	const int sdcount = 8;
	float dists_graph[dcount] = { 0/*0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.2, 0.3 */};
	dists_graph[0] = 0.01;	// xyz
	dists_graph[1] = 0.02;	// desk
	dists_graph[2] = 0.02;	// desk2
	dists_graph[3] = 0.02;	// 360
	dists_graph[4] = 0.01;	// room
	dists_graph[5] = 0.02;	// floor
	dists_graph[6] = 0.02;	// rpy
	dists_graph[7] = 0.05;	// teddy
	dists_graph[8] = 0.02;	// plants

	const int ncount = 2;
	int ninterval[ncount] = { 15, 30 };

	const int pcount = 3;
	float percent[pcount] = { 0.5, 0.6, 0.7 };

	int st = -1, ed = -1;
	stringstream ss;

	for (int dd = 0; dd < dcount; dd++)
	{
		readData(directories[dd], st, ed, false);

		int id;
		string fe, ct;
		trans.clear();
		ifstream input(tnames[dd]);
		while (!input.eof())
		{
			input >> id;
			if (id != -1)
			{
				input >> fe >> ct;
				Eigen::Matrix4f tran;
				for (int i = 0; i < 4; i++)
				{
					for (int j = 0; j < 4; j++)
					{
						input >> tran(i, j);
					}
				}
				trans.push_back(tran);
			}
			else
			{
				break;
			}
		}

		cout << "\tpairwise dists: " << dists_pair[dd] << endl;
		Config::instance()->set<float>("max_inlier_dist", dists_pair[dd]);
		Config::instance()->set<float>("graph_max_inlier_dist", dists_graph[dd]);

		for (int i = 0; i < graph.size(); i++)
		{
			if (graph[i])
			{
				delete graph[i];
			}
		}
		graph.clear();

		for (int i = 0; i < frame_count; i++)
		{
			Frame *frame = new Frame(rgbs[i], depths[i], "surf");
			frame->f->buildFlannIndex();
			graph.push_back(frame);
		}

		for (int rd = 0; rd < 10; rd++)
		{
// 			ss.clear();
// 			ss.str("");
// 			ss << "E:/tempresults2/" << names[dd] << "/"
// 				<< names[dd] << "_ftof_surf_" << rd  << "_2.txt";
// 			ofstream outp(ss.str());
// 
// 			ss.clear();
// 			ss.str("");
// 			ss << "E:/tempresults2/" << names[dd] << "/"
// 				<< names[dd] << "_surf_" << rd << "_2.txt";
// 			ofstream outg(ss.str());

// 			PairwiseRegistration("surf", &outp, &outg, false);
// 
// 			trans.clear();
// 			for (int i = 0; i < frame_count; i++)
// 			{
// 				trans.push_back(graph[i]->relative_tran);
// 			}
// 
// 			Config::instance()->set<float>("graph_max_inlier_dist", dists_graph[rd]);

			for (int nd = 0; nd < ncount; nd++)
			{
				Config::instance()->set<int>("max_keyframe_interval", ninterval[nd]);
				for (int pd = 0; pd < pcount; pd++)
				{
					cout << "\t\tInterval: " << ninterval[nd] << ", Percent" << percent[pd] << endl;
					Config::instance()->set<float>("keyframe_rational", percent[pd]);

					ss.clear();
					ss.str("");
					ss << "E:/school/tempresults2/" /*<< names[dd] << "/"*/
						<< names[dd] << "_global_surf_" << rd << "_"
						<< ninterval[nd] << "_" << percent[pd] << ".txt";
					ofstream out1(ss.str());

					ss.clear();
					ss.str("");
					ss << "E:/school/tempresults2/" /*<< names[dd] << "/"*/
						<< names[dd] << "_global_surf_" << rd << "_"
						<< ninterval[nd] << "_" << percent[pd] << ".log";
					ofstream out2(ss.str());

					GlobalRegistration("surf", &out1, &out2);
				}
			}
		}
	}	
}

void showResultWithTrajectory(ifstream *input)
{
	if (!input)
		return;

	set<int> needed;
	trans.clear();
	string line;
	stringstream ss;
	int k = 0;
	while (getline(*input, line))
	{
		ss.clear();
		ss.str("");
		ss << line;

		double timestamp;
		Eigen::Vector3f t;
		Eigen::Quaternionf q;
		ss >> timestamp >> t(0) >> t(1) >> t(2) >> q.x() >> q.y() >> q.z() >> q.w();
		Eigen::Matrix4f tran = transformationFromQuaternionsAndTranslation(q, t);

		while (k < frame_count &&
			fabs(timestamp - timestamps[k]) >= 1e-4 && timestamp > timestamps[k])
		{
			trans.push_back(Eigen::Matrix4f::Identity());
			k++;
		}

		if (k < frame_count && fabs(timestamp - timestamps[k]) < 1e-4)
		{
			trans.push_back(tran);
			needed.insert(k);
			k++;
		}
	}

	PointCloudPtr cloud(new PointCloudT);
	for (int i = 0; i < frame_count; i++)
	{
		if (needed.find(i) == needed.end())
			continue;
		PointCloudPtr tc(new PointCloudT);
		pcl::transformPointCloud(*clouds[i], *tc, trans[i]);

		*cloud += *tc;
		if ((i + 1) % 30 == 0)
		{
			cloud = DownSamplingByVoxelGrid(cloud, 0.01, 0.01, 0.01);
		}
	}
	cloud = DownSamplingByVoxelGrid(cloud, 0.01, 0.01, 0.01);
	
	cout << "Getting scene" << endl;
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	viewer->setBackgroundColor(0, 0, 0);
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
//	viewer->registerKeyboardCallback(KeyboardEventOccurred, (void*)&viewer);
	viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");
//	viewer->addCoordinateSystem(1.0);
	viewer->initCameraParameters();

	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
	}
}

// void time_test()
// {
// 	const int dcount = 9;
// 	std::string directories[dcount], names[dcount], prs[dcount];
// 	directories[0] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_xyz/";
// 	directories[1] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_desk/";
// 	directories[2] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_desk2/";
// 	directories[3] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_360/";
// 	directories[4] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_room/";
// 	directories[5] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_floor/";
// 	directories[6] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_rpy";
// 	directories[7] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_teddy/";
// 	directories[8] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_plant/";
// 
// 	names[0] = "xyz";
// 	names[1] = "desk";
// 	names[2] = "desk2";
// 	names[3] = "360";
// 	names[4] = "room";
// 	names[5] = "floor";
// 	names[6] = "rpy";
// 	names[7] = "teddy";
// 	names[8] = "plant";
// 
// 	prs[0] = "E:/bestresults/xyz/xyz_ftof_SURF_0.05.txt";
// 	prs[1] = "E:/bestresults/desk/desk_ftof_SURF_0.08.txt";
// 	prs[2] = "E:/bestresults/desk2/desk2_ftof_SURF_0.1.txt";
// 	prs[3] = "E:/bestresults/360/360_repeat_SURF_1.txt";
// 	prs[4] = "E:/bestresults/room/room_repeat_SURF_1.txt";
// 	prs[5] = "E:/bestresults/floor/floor_repeat_SURF_2.txt";
// 	prs[6] = "E:/bestresults/rpy/rpy_repeat_SURF_2.txt";
// 	prs[7] = "E:/bestresults/teddy/teddy_repeat_SURF_0.txt";
// 	prs[8] = "E:/bestresults/plant/plant_repeat_SURF_1.txt";
// 
// 	float dists[dcount] = { 0.01, 0.02, 0.05, 0.01, 0.03, 0.03, 0.03, 0.03, 0.01 };
// 
// 	int st = -1, ed = -1;
// 	stringstream ss;
// 
// 	for (int dd = 0; dd < dcount; dd++)
// 	{
// 		readData(directories[dd], st, ed, false);
// 		Config::instance()->set<float>("max_inlier_dist", dists[dd]);
// 
// 		ss.clear();
// 		ss.str("");
// 		ss << "E:/" << names[dd] << "_log.txt";
// 		ofstream log_out(ss.str());
// 		PairwiseRegistration("surf", false, true, nullptr, nullptr, true, &log_out);
// 
// 		GlobalRegistration("surf", nullptr, nullptr, &log_out);
// 	}
// }

// 特征点比值判定
void e1()
{
	cv::Mat r1 = cv::imread("E:/1.png");
	cv::Mat r2 = cv::imread("E:/2.png");
	
	cv::SURF surf_detector;
	cv::Mat mask, descriptors1, descriptors2;
	vector<cv::KeyPoint> fpts1, fpts2;

	surf_detector(r1, mask, fpts1, descriptors1);
	surf_detector(r2, mask, fpts2, descriptors2);

	int trees = Config::instance()->get<int>("kdtree_trees");
	int max_leaf = Config::instance()->get<int>("kdtree_max_leaf");
	cv::FlannBasedMatcher *flann_matcher = new cv::FlannBasedMatcher(new cv::flann::KDTreeIndexParams(trees),
		new cv::flann::SearchParams(max_leaf));
	vector<cv::Mat> ds;
	ds.push_back(descriptors1);
	flann_matcher->add(ds);

	vector<vector<cv::DMatch>> matches_;
	flann_matcher->knnMatch(descriptors2, matches_, 2);
	float ratio = Config::instance()->get<float>("matches_criterion");

	vector<int> matches_success, matches_failed;
	for (int i = 0; i < matches_.size(); i++)
	{
		if (matches_[i][0].distance < ratio * matches_[i][1].distance)
		{
			matches_success.push_back(i);
		}
		else
		{
			matches_failed.push_back(i);
		}
	}

	cv::Mat s1, s2;
	int p = 0;

	while (true)
	{
		r1.copyTo(s1);
		r2.copyTo(s2);

		cv::DMatch m1 = matches_[matches_failed[p]][0];
		cv::DMatch m2 = matches_[matches_failed[p]][1];

		cv::circle(s2, cv::Point(fpts2[m1.queryIdx].pt), 5, cv::Scalar(0, 255, 0), 3);
		cv::circle(s1, cv::Point(fpts1[m1.trainIdx].pt), 5, cv::Scalar(0, 0, 255), 3);
		cv::circle(s1, cv::Point(fpts1[m2.trainIdx].pt), 5, cv::Scalar(255, 0, 0), 3);

		cv::imshow("1", s1);
		cv::imshow("2", s2);
		int key = cv::waitKey();

		if (key == 's')
		{
			cv::imwrite("E:/save1.png", s1);
			cv::imwrite("E:/save2.png", s2);
			cout << m1.distance << "\t" << m2.distance << endl;
		}
		else
			p++;
	}
	
}

// ICPCUDA， pairwise
void e2()
{
	const int dcount = 9;
	std::string directories[dcount], names[dcount];
	directories[0] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_xyz/";
	directories[1] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_desk/";
	directories[2] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_desk2/";
	directories[3] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_360/";
	directories[4] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_room/";
	directories[5] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_floor/";
	directories[6] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_rpy";
	directories[7] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_teddy/";
	directories[8] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_plant/";

	names[0] = "xyz";
	names[1] = "desk";
	names[2] = "desk2";
	names[3] = "360";
	names[4] = "room";
	names[5] = "floor";
	names[6] = "rpy";
	names[7] = "teddy";
	names[8] = "plant";

	const int fcount = 4;
	std::string ftype[fcount];
	ftype[0] = "sift";
	ftype[1] = "surf";
	ftype[2] = "orb";
	ftype[3] = "ICPCUDA";

	const int sdcount = 8;
	float dists[sdcount];
	dists[0] = 0.01;
	dists[1] = 0.02;
	dists[2] = 0.03;
	dists[3] = 0.05;
	dists[4] = 0.08;
	dists[5] = 0.1;
	dists[6] = 0.2;
	dists[7] = 0.3;

	int st = -1, ed = -1;
	stringstream ss;

	for (int dd = 0; dd < dcount; dd++)
	{
		readData(directories[dd], st, ed);

		for (int fd = 0; fd < fcount; fd++)
		{
			cout << "type: " << ftype[fd];

			for (int sd = 0; sd < sdcount; sd++)
			{
				cout << "\t dists: " << dists[sd];

				ss.clear();
				ss.str("");
				ss << "E:/tempresults2/" << names[dd] << "/"
					<< names[dd] << "_ICPCUDA_" << dists[sd] << ".txt";
				ofstream out2(ss.str());

				int threads = Config::instance()->get<int>("icpcuda_threads");
				int blocks = Config::instance()->get<int>("icpcuda_blocks");
				int width = Config::instance()->get<int>("image_width");
				int height = Config::instance()->get<int>("image_height");
				float cx = Config::instance()->get<float>("camera_cx");
				float cy = Config::instance()->get<float>("camera_cy");
				float fx = Config::instance()->get<float>("camera_fx");
				float fy = Config::instance()->get<float>("camera_fy");
				float depthFactor = Config::instance()->get<float>("depth_factor");
				float distThresh = dists[sd];
				float angleThresh = Config::instance()->get<float>("angle_threshold");
				ICPOdometry *icpcuda = new ICPOdometry(width, height, cx, cy, fx, fy, depthFactor, distThresh, angleThresh);

				clock_t start, total_start = clock();
				double time;
				Eigen::Matrix4f last_tran;
				cv::Mat last_depth;

				for (int i = 0; i < frame_count; i++)
				{
					if (i == 0)
					{
						last_tran = Eigen::Matrix4f::Identity();

						Eigen::Vector3f t = TranslationFromMatrix4f(last_tran);
						Eigen::Quaternionf q = QuaternionFromMatrix4f(last_tran);
						out2 << fixed << setprecision(6) << timestamps[i]
							<< ' ' << t(0) << ' ' << t(1) << ' ' << t(2)
							<< ' ' << q.x() << ' ' << q.y() << ' ' << q.z() << ' ' << q.w() << endl;
					}
					else
					{
						Eigen::Matrix4f tran;
						start = clock();
						icpcuda->initICPModel((unsigned short *)depths[i - 1].data, 20.0f, Eigen::Matrix4f::Identity());
						icpcuda->initICP((unsigned short *)depths[i].data, 20.0f);

						tran = Eigen::Matrix4f::Identity();
						Eigen::Vector3f t = tran.topRightCorner(3, 1);
						Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rot = tran.topLeftCorner(3, 3);

						icpcuda->getIncrementalTransformation(t, rot, threads, blocks);

						tran.topLeftCorner(3, 3) = rot;
						tran.topRightCorner(3, 1) = t;

						last_tran = last_tran * tran;
						t = TranslationFromMatrix4f(last_tran);
						Eigen::Quaternionf q = QuaternionFromMatrix4f(last_tran);
						out2 << fixed << setprecision(6) << timestamps[i]
							<< ' ' << t(0) << ' ' << t(1) << ' ' << t(2)
							<< ' ' << q.x() << ' ' << q.y() << ' ' << q.z() << ' ' << q.w() << endl;
					}
				}

				time = clock() - total_start;
				out2.close();

				delete icpcuda;
				cout << endl;
			}
		}
	}
}

void keyframe_e()
{
	char ch;

	string filename = "";
	cout << "filename: ";
	scanf("%c", &ch);
	while (filename.length() == 0 || ch != '\n')
	{
		if (ch != '\n' && ch != '\"')
		{
			filename += ch;
		}
		scanf("%c", &ch);
	}
	ifstream input(filename);

	string directory = "";
	cout << "directory: ";

	scanf("%c", &ch);
	while (directory.length() == 0 || ch != '\n')
	{
		if (ch != '\n' && ch != '\"')
		{
			directory += ch;
		}
		scanf("%c", &ch);
	}
	
	for (int i = 0; i < graph.size(); i++)
	{
		if (graph[i])
		{
			delete graph[i];
		}
	}
	graph.clear();

	int id;
	string fe, ct;

	cout << "Reading trajectories" << endl;
	while (!input.eof())
	{
		input >> id;
		if (id != -1)
		{
			input >> fe >> ct;
			Eigen::Matrix4f tran;
			for (int i = 0; i < 4; i++)
			{
				for (int j = 0; j < 4; j++)
				{
					input >> tran(i, j);
				}
			}
			trans.push_back(tran);
		}
		else
		{
			break;
		}
	}
	input.close();

	readData(directory, -1, -1);
	
	cout << "Extracting Features" << endl;
	for (int i = 0; i < frame_count; i++)
	{
		Frame *frame = new Frame(rgbs[i], depths[i], "surf");
		frame->f->buildFlannIndex();
		graph.push_back(frame);
	}

	int min_matches = Config::instance()->get<int>("min_matches");
	float inlier_percentage = Config::instance()->get<float>("min_inlier_p");
	float inlier_dist;
	cout << "dist: ";
	cin >> inlier_dist;

	PairwiseRegister *reg = new SurfRegister(min_matches, inlier_percentage, inlier_dist);

	int accumulated_frame_count, last_keyframe_id;
	Eigen::Matrix4f accumulated_transformation;
	bool is_last_frame_keyframe = false;
	float last_rational = 1;
	vector<int> keyframe_1, keyframe_2;

	cout << "Keyframe method 1" << endl;
	for (int i = 0; i < frame_count; i++)
	{
		cout << "Frame " << i << ": ";
		bool isKeyframe = false;
		if (i == 0)
		{
			isKeyframe = true;
			cout << "1" << endl;
		}
		else
		{
			accumulated_frame_count++;
			accumulated_transformation = accumulated_transformation * trans[i];

			float rrr = reg->getCorrespondencePercent(graph[last_keyframe_id], graph[i], accumulated_transformation);
			if (is_last_frame_keyframe)
			{
				last_rational = rrr;
			}
			rrr /= last_rational;
			cout << rrr << endl;
			if (rrr < Config::instance()->get<float>("keyframe_rational"))
			{
				isKeyframe = true;
			}
		}

		if (accumulated_frame_count >= Config::instance()->get<int>("max_keyframe_interval"))
		{
			isKeyframe = true;
		}

		if (isKeyframe)
		{
			accumulated_frame_count = 0;
			accumulated_transformation = Eigen::Matrix4f::Identity();
			last_keyframe_id = i;
			keyframe_1.push_back(i);
			is_last_frame_keyframe = true;
		}
		else
		{
			is_last_frame_keyframe = false;
		}
	}

	cout << "keyframe method 2" << endl;
	for (int i = 0; i < frame_count; i++)
	{
		cout << "Frame " << i << ": ";
		bool isKeyframe = false;
		if (i == 0)
		{
			isKeyframe = true;
			cout << "1" << endl;
		}
		else
		{
			accumulated_frame_count++;
			accumulated_transformation = accumulated_transformation * trans[i];

			float rrr = reg->getCorrespondencePercent(graph[last_keyframe_id], graph[i], accumulated_transformation);
			cout << rrr << endl;
			if (rrr < Config::instance()->get<float>("keyframe_rational"))
			{
				isKeyframe = true;
			}
		}

		if (accumulated_frame_count >= Config::instance()->get<int>("max_keyframe_interval"))
		{
			isKeyframe = true;
		}

		if (isKeyframe)
		{
			accumulated_frame_count = 0;
			accumulated_transformation = Eigen::Matrix4f::Identity();
			last_keyframe_id = i;
			keyframe_2.push_back(i);
			is_last_frame_keyframe = true;
		}
		else
		{
			is_last_frame_keyframe = false;
		}
	}

	cout << keyframe_1.size() << endl;
	cout << keyframe_2.size() << endl;
}

void keyframe_e2()
{
	const int dcount = 9;
	std::string directories[dcount], names[dcount], tnames[dcount];
	directories[0] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_xyz/";
	directories[1] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_desk/";
	directories[2] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_desk2/";
	directories[3] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_360/";
	directories[4] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_room/";
	directories[5] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_floor/";
	directories[6] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_rpy";
	directories[7] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_teddy/";
	directories[8] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_plant/";

	names[0] = "xyz";
	names[1] = "desk";
	names[2] = "desk2";
	names[3] = "360";
	names[4] = "room";
	names[5] = "floor";
	names[6] = "rpy";
	names[7] = "teddy";
	names[8] = "plant";

	for (int i = 0; i < 9; i++)
	{
		tnames[i] = "E:/bestresults2/" + names[i] + "/" + names[i] + "_ftof_surf.txt";
	}

	float dists[dcount] = { 0 };
	dists[0] = 0.05;	// xyz
	dists[1] = 0.02;	// desk
	dists[2] = 0.02;	// desk2
	dists[3] = 0.08;	// 360
	dists[4] = 0.02;	// room
	dists[5] = 0.08;	// floor
	dists[6] = 0.01;	// rpy
	dists[7] = 0.2;		// teddy
	dists[8] = 0.2;		// plants

	int st = -1, ed = -1;
	stringstream ss;
	int id;
	string fe, ct;
	Eigen::Matrix4f tran;

	for (int dd = 0; dd < dcount; dd++)
	{
		cout << "Reading trajectories" << endl;
		trans.clear();
		ifstream input(tnames[dd]);
		while (!input.eof())
		{
			input >> id;
			if (id != -1)
			{
				input >> fe >> ct;
				Eigen::Matrix4f tran;
				for (int i = 0; i < 4; i++)
				{
					for (int j = 0; j < 4; j++)
					{
						input >> tran(i, j);
					}
				}
				trans.push_back(tran);
			}
			else
			{
				break;
			}
		}
		readData(directories[dd], st, ed, false);

		cout << "\tfeature: surf\tdists: " << dists[dd];

		ss.clear();
		ss.str("");
		ss << "E:/tempresults2/" << names[dd] << "/"
			<< names[dd] << "_keyframe_50_2.txt";
		ofstream out1(ss.str());

		for (int i = 0; i < graph.size(); i++)
		{
			if (graph[i])
			{
				delete graph[i];
			}
		}
		graph.clear();

		for (int i = 0; i < frame_count; i++)
		{
			Frame *frame = new Frame(rgbs[i], depths[i], "surf");
			frame->f->buildFlannIndex();
			graph.push_back(frame);
		}

		int min_matches = Config::instance()->get<int>("min_matches");
		float inlier_percentage = Config::instance()->get<float>("min_inlier_p");
		float inlier_dist = dists[dd];

		PairwiseRegister *reg = new SurfRegister(min_matches, inlier_percentage, inlier_dist);

		int accumulated_frame_count = 0, last_keyframe_id;
		Eigen::Matrix4f accumulated_transformation = Eigen::Matrix4f::Identity();
		bool is_last_frame_keyframe = false;
		float last_rational = 1;
		vector<int> keyframe;

		for (int i = 0; i < frame_count; i++)
		{
			bool isKeyframe = false;
			if (i == 0)
			{
				isKeyframe = true;
			}
			else
			{
				accumulated_frame_count++;
				accumulated_transformation = accumulated_transformation * trans[i];

				float rrr = reg->getCorrespondencePercent(graph[last_keyframe_id], graph[i], accumulated_transformation);
// 				if (is_last_frame_keyframe)
// 				{
// 					last_rational = rrr;
// 				}
// 				rrr /= last_rational;
				if (rrr < Config::instance()->get<float>("keyframe_rational"))
				{
					isKeyframe = true;
				}
			}

			if (accumulated_frame_count >= Config::instance()->get<int>("max_keyframe_interval"))
			{
				isKeyframe = true;
			}

			if (isKeyframe)
			{
				accumulated_frame_count = 0;
				accumulated_transformation = Eigen::Matrix4f::Identity();
				last_keyframe_id = i;
				keyframe.push_back(i);
				is_last_frame_keyframe = true;
				out1 << i << endl;
			}
			else
			{
				is_last_frame_keyframe = false;
			}
		}
		out1.close();
		cout << endl;
	}
}

void time_analyze()
{
	const int dcount = 9;
	string names[dcount];

	names[0] = "xyz";
	names[1] = "desk";
	names[2] = "desk2";
	names[3] = "360";
	names[4] = "room";
	names[5] = "floor";
	names[6] = "rpy";
	names[7] = "teddy";
	names[8] = "plant";

	const int ncount = 2;
	int ninterval[ncount] = { 15, 30 };

	const int pcount = 3;
	float percent[pcount] = { 0.5, 0.6, 0.7 };

	stringstream ss;

	ss.clear();
	ss.str("");
	ss << "E:/data/result_time_20161115.csv";
	ofstream outfile(ss.str());
	outfile << "Time,,,,,,,,,,," << endl;
	outfile << "Scene,,NInterval,Percent,Frame,Keyframe,Candidata,total,kdtree build,kdtree match,ransac,graph";
	
	for (int dd = 0; dd < dcount; dd++)
	{
		outfile << names[dd];
		for (int fd = 0; fd < 10; fd++)
		{
			for (int nd = 0; nd < ncount; nd++)
			{
				for (int pd = 0; pd < pcount; pd++)
				{
					outfile << "," << fd << "," << ninterval[nd] << "," << percent[pd] << ",,";

					ss.clear();
					ss.str("");
					ss << "E:/school/tempresults2/" << names[dd] << "/"
						<< names[dd] << "_global_surf_" << fd << "_"
						<< ninterval[nd] << "_" << percent[pd] << ".log";

					ifstream infile(ss.str());
					string line;

					while (getline(infile, line))
					{
						int pos = line.find("Keyframe number ");
						if (pos != -1)
						{
							outfile << line.substr(16, line.length() - 16) << ",";
						}

						pos = line.find("LC candidate ");
						if (pos != -1)
						{
							outfile << line.substr(13, line.length() - 13) << ",";
						}

						pos = line.find("total time per frame ");
						if (pos != -1)
						{
							outfile << line.substr(21, line.length() - 21) << ",";
						}

						pos = line.find("kdtree build per frame");
						if (pos != -1)
						{
							outfile << line.substr(22, line.length() - 22) << ",";
						}

						pos = line.find("kdtree match per frame");
						if (pos != -1)
						{
							outfile << line.substr(22, line.length() - 22) << ",";
						}

						pos = line.find("loop ransac per frame");
						if (pos != -1)
						{
							outfile << line.substr(21, line.length() - 21) << ",";
						}

						pos = line.find("graph per frame");
						if (pos != -1)
						{
							outfile << line.substr(15, line.length() - 15);
						}
					}
					infile.close();
					outfile << endl;
				}
			}
		}
	}
	outfile.close();
}

// ransac + icp or ransac failed icp
void icp_test_1()
{
	const int dcount = 2;
	std::string directories[dcount], names[dcount];
	directories[0] = "E:/school/data/rgbd_dataset_freiburg1_desk/";
	directories[1] = "E:/school/data/rgbd_dataset_freiburg1_floor/";

	names[0] = "desk";
	names[1] = "floor";

// 	const int fcount = 4;
// 	std::string ftypes[fcount];
// 	ftypes[0] = "surf_icpcuda";
// 	ftypes[1] = "surf_icpcuda_only_failed";

	float dists[dcount];
	dists[0] = 0.02;	// desk
	dists[1] = 0.08;	// floor

	int st = -1, ed = -1;
	stringstream ss;

	for (int dd = 0; dd < dcount; dd++)
	{
		readData(directories[dd], st, ed, false);

		cout << "\t dists: " << dists[dd];
		Config::instance()->set<float>("max_inlier_dist", dists[dd]);
		Config::instance()->set<float>("dist_threshold", dists[dd]);

		for (int i = 0; i < graph.size(); i++)
		{
			if (graph[i])
			{
				delete graph[i];
			}
		}
		graph.clear();

		for (int i = 0; i < frame_count; i++)
		{
			Frame *frame = new Frame(rgbs[i], depths[i], "surf", Eigen::Matrix4f::Identity());
			frame->f->buildFlannIndex();
			graph.push_back(frame);
		}

		int min_matches = Config::instance()->get<int>("min_matches");
		float inlier_percent = Config::instance()->get<float>("min_inlier_p");
		float inlier_dist = Config::instance()->get<float>("max_inlier_dist");

		int threads = Config::instance()->get<int>("icpcuda_threads");
		int blocks = Config::instance()->get<int>("icpcuda_blocks");
		int width = Config::instance()->get<int>("image_width");
		int height = Config::instance()->get<int>("image_height");
		float cx = Config::instance()->get<float>("camera_cx");
		float cy = Config::instance()->get<float>("camera_cy");
		float fx = Config::instance()->get<float>("camera_fx");
		float fy = Config::instance()->get<float>("camera_fy");
		float depthFactor = Config::instance()->get<float>("depth_factor");
		float distThresh = Config::instance()->get<float>("dist_threshold");
		float angleThresh = Config::instance()->get<float>("angle_threshold");
		ICPOdometry *icpcuda = new ICPOdometry(width, height, cx, cy, fx, fy, depthFactor, distThresh, angleThresh);
		float keyframe_rational = Config::instance()->get<float>("keyframe_rational");

		PairwiseRegister *reg_surf = new SurfRegister(min_matches, inlier_percent, inlier_dist);
		PairwiseRegister *reg_icpcuda = new IcpcudaRegister(icpcuda, threads, blocks, 20.0f);

		for (int rd = 0; rd < 5; rd++)
		{
			// surf + icpcuda if surf failed
			for (int i = 0; i < frame_count; i++)
			{
				if (i == 0)
				{
					graph[i]->relative_tran = Eigen::Matrix4f::Identity();
				}
				else
				{
					Eigen::Matrix4f tran = Eigen::Matrix4f::Identity();
					if (reg_surf->getTransformation(graph[i - 1], graph[i], tran))
					{
						graph[i]->relative_tran = tran;
					}
					else
					{
						// surf failed, run icpcuda
						reg_icpcuda->getTransformation(depths[i - 1].data, depths[i].data, tran);
						graph[i]->relative_tran = tran;
					}
				}
			}

			ss.clear();
			ss.str("");
			ss << "E:/school/tempresults2/" << names[dd] << "/"
				<< names[dd] << "_surf_failed_icpcuda_" << rd << ".txt";
			ofstream gout(ss.str());

			Eigen::Matrix4f last_tran = Eigen::Matrix4f::Identity();
			for (int j = 0; j < frame_count; j++)
			{
				graph[j]->tran = last_tran * graph[j]->relative_tran;

				Eigen::Vector3f t = TranslationFromMatrix4f(graph[j]->tran);
				Eigen::Quaternionf q = QuaternionFromMatrix4f(graph[j]->tran);

				gout << fixed << setprecision(6) << timestamps[j]
					<< ' ' << t(0) << ' ' << t(1) << ' ' << t(2)
					<< ' ' << q.x() << ' ' << q.y() << ' ' << q.z() << ' ' << q.w() << endl;

				last_tran = graph[j]->tran;
			}
			gout.close();

			// surf + icpcuda

			for (int i = 0; i < frame_count; i++)
			{
				if (i == 0)
				{
					graph[i]->relative_tran = Eigen::Matrix4f::Identity();
				}
				else
				{
					Eigen::Matrix4f tran = Eigen::Matrix4f::Identity();
					reg_surf->getTransformation(graph[i - 1], graph[i], tran);
					Eigen::Matrix4f inv_tran = tran.inverse();
					reg_icpcuda->getTransformation(depths[i - 1].data, depths[i].data, inv_tran);
					graph[i]->relative_tran = tran * inv_tran;
				}
			}

// 			boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("test"));
// 			viewer->setBackgroundColor(0, 0, 0);
// 			viewer->addCoordinateSystem(1.0);
// 			viewer->initCameraParameters();
// 
// 			Eigen::Matrix4f tran = Eigen::Matrix4f::Identity();
// 			PointCloudPtr cloud_all(new PointCloudT), tran_cloud(new PointCloudT);
// 			for (int i = 0; i < frame_count; i++)
// 			{
// 				tran = tran * graph[i]->relative_tran;
// 				PointCloudPtr tran_cloud(new PointCloudT);
// 				pcl::transformPointCloud(*clouds[i], *tran_cloud, tran);
// 				*cloud_all += *tran_cloud;
// 			}
// 
// 			cloud_all = DownSamplingByVoxelGrid(cloud_all, 0.01, 0.01, 0.01);
// 			pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud_all);
// 			viewer->addPointCloud<pcl::PointXYZRGB>(cloud_all, rgb, "result");
// 			viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "result");
// 
// 			while (!viewer->wasStopped())
// 			{
// 				viewer->spinOnce(100);
// 				//boost::this_thread::sleep (boost::posix_time::microseconds (100000));
// 			}

			ss.clear();
			ss.str("");
			ss << "E:/school/tempresults2/" << names[dd] << "/"
				<< names[dd] << "_surf_icpcuda_" << rd << ".txt";
			ofstream gout1(ss.str());

			last_tran = Eigen::Matrix4f::Identity();
			for (int j = 0; j < frame_count; j++)
			{
				graph[j]->tran = last_tran * graph[j]->relative_tran;

				Eigen::Vector3f t = TranslationFromMatrix4f(graph[j]->tran);
				Eigen::Quaternionf q = QuaternionFromMatrix4f(graph[j]->tran);

				gout1 << fixed << setprecision(6) << timestamps[j]
					<< ' ' << t(0) << ' ' << t(1) << ' ' << t(2)
					<< ' ' << q.x() << ' ' << q.y() << ' ' << q.z() << ' ' << q.w() << endl;

				last_tran = graph[j]->tran;
			}
			gout1.close();
			
		}
		cout << endl;
	}
}

// GICP vs ICPCUDA
void icp_test_2()
{
	const int dcount = 2;
	std::string directories[dcount], names[dcount];
	directories[0] = "E:/school/data/rgbd_dataset_freiburg1_desk/";
	directories[1] = "E:/school/data/rgbd_dataset_freiburg1_floor/";

	names[0] = "desk";
	names[1] = "floor";

	float dists[dcount];
	dists[0] = 0.02;	// desk
	dists[1] = 0.08;	// floor

	int st = -1, ed = -1;
	stringstream ss;

	for (int dd = 0; dd < dcount; dd++)
	{
		readData(directories[dd], st, ed, true);

		cout << "\t dists: " << dists[dd];
		Config::instance()->set<float>("max_inlier_dist", dists[dd]);
		Config::instance()->set<float>("dist_threshold", dists[dd]);

		for (int i = 0; i < graph.size(); i++)
		{
			if (graph[i])
			{
				delete graph[i];
			}
		}
		graph.clear();

		for (int i = 0; i < frame_count; i++)
		{
			Frame *frame = new Frame();
			graph.push_back(frame);
		}

		int threads = Config::instance()->get<int>("icpcuda_threads");
		int blocks = Config::instance()->get<int>("icpcuda_blocks");
		int width = Config::instance()->get<int>("image_width");
		int height = Config::instance()->get<int>("image_height");
		float cx = Config::instance()->get<float>("camera_cx");
		float cy = Config::instance()->get<float>("camera_cy");
		float fx = Config::instance()->get<float>("camera_fx");
		float fy = Config::instance()->get<float>("camera_fy");
		float depthFactor = Config::instance()->get<float>("depth_factor");
		float distThresh = Config::instance()->get<float>("dist_threshold");
		float angleThresh = Config::instance()->get<float>("angle_threshold");
		ICPOdometry *icpcuda = new ICPOdometry(width, height, cx, cy, fx, fy, depthFactor, distThresh, angleThresh);

		PairwiseRegister *reg_gicp = new GeneralizedIcpRegister(distThresh);
		PairwiseRegister *reg_icpcuda = new IcpcudaRegister(icpcuda, threads, blocks, 20.0f);

		for (int rd = 0; rd < 5; rd++)
		{
			// gicp test
			for (int i = 0; i < frame_count; i++)
			{
				if (i == 0)
				{
					graph[i]->relative_tran = Eigen::Matrix4f::Identity();
				}
				else
				{
					Eigen::Matrix4f tran = Eigen::Matrix4f::Identity();
					if (reg_gicp->getTransformation(&clouds[i - 1], &clouds[i], tran))
					{
						graph[i]->relative_tran = tran;
					}
					else
					{
						graph[i]->relative_tran = graph[i - 1]->relative_tran;
					}
				}
			}

			ss.clear();
			ss.str("");
			ss << "E:/school/tempresults2/" << names[dd] << "/"
				<< names[dd] << "_gicp_" << rd << ".txt";
			ofstream gout(ss.str());

			Eigen::Matrix4f last_tran = Eigen::Matrix4f::Identity();
			for (int j = 0; j < frame_count; j++)
			{
				graph[j]->tran = last_tran * graph[j]->relative_tran;

				Eigen::Vector3f t = TranslationFromMatrix4f(graph[j]->tran);
				Eigen::Quaternionf q = QuaternionFromMatrix4f(graph[j]->tran);

				gout << fixed << setprecision(6) << timestamps[j]
					<< ' ' << t(0) << ' ' << t(1) << ' ' << t(2)
					<< ' ' << q.x() << ' ' << q.y() << ' ' << q.z() << ' ' << q.w() << endl;

				last_tran = graph[j]->tran;
			}
			gout.close();

			// surf + icpcuda

			for (int i = 0; i < frame_count; i++)
			{
				if (i == 0)
				{
					graph[i]->relative_tran = Eigen::Matrix4f::Identity();
				}
				else
				{
					Eigen::Matrix4f tran = Eigen::Matrix4f::Identity();
					reg_icpcuda->getTransformation(depths[i - 1].data, depths[i].data, tran);
					graph[i]->relative_tran = tran;
				}
			}

// 			boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("test"));
// 			viewer->setBackgroundColor(0, 0, 0);
// 			viewer->addCoordinateSystem(1.0);
// 			viewer->initCameraParameters();
// 
// 			Eigen::Matrix4f tran = Eigen::Matrix4f::Identity();
// 			PointCloudPtr cloud_all(new PointCloudT), tran_cloud(new PointCloudT);
// 			for (int i = 0; i < frame_count; i++)
// 			{
// 				tran = tran * graph[i]->relative_tran;
// 				PointCloudPtr tran_cloud(new PointCloudT);
// 				pcl::transformPointCloud(*clouds[i], *tran_cloud, tran);
// 				*cloud_all += *tran_cloud;
// 			}
// 
// 			cloud_all = DownSamplingByVoxelGrid(cloud_all, 0.01, 0.01, 0.01);
// 			pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud_all);
// 			viewer->addPointCloud<pcl::PointXYZRGB>(cloud_all, rgb, "result");
// 			viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "result");
// 
// 			while (!viewer->wasStopped())
// 			{
// 				viewer->spinOnce(100);
// 				//boost::this_thread::sleep (boost::posix_time::microseconds (100000));
// 			}

			ss.clear();
			ss.str("");
			ss << "E:/school/tempresults2/" << names[dd] << "/"
				<< names[dd] << "_icpcuda_" << rd << ".txt";
			ofstream gout1(ss.str());

			last_tran = Eigen::Matrix4f::Identity();
			for (int j = 0; j < frame_count; j++)
			{
				graph[j]->tran = last_tran * graph[j]->relative_tran;

				Eigen::Vector3f t = TranslationFromMatrix4f(graph[j]->tran);
				Eigen::Quaternionf q = QuaternionFromMatrix4f(graph[j]->tran);

				gout1 << fixed << setprecision(6) << timestamps[j]
					<< ' ' << t(0) << ' ' << t(1) << ' ' << t(2)
					<< ' ' << q.x() << ' ' << q.y() << ' ' << q.z() << ' ' << q.w() << endl;

				last_tran = graph[j]->tran;
			}
			gout1.close();

		}
		cout << endl;
	}

}

// 1 kdtree vs multiple kdtrees
void lc_test()
{
	const int dcount = 2;
	std::string directories[dcount], names[dcount];
	directories[0] = "E:/school/data/rgbd_dataset_freiburg1_desk/";
	directories[1] = "E:/school/data/rgbd_dataset_freiburg1_floor/";

	names[0] = "desk";
	names[1] = "floor";

	float dists[dcount];
	dists[0] = 0.02;	// desk
	dists[1] = 0.08;	// floor

	int st = -1, ed = -1;
	stringstream ss;

	for (int dd = 0; dd < dcount; dd++)
	{
		readData(directories[dd], st, ed, false);

		cout << "\t dists: " << dists[dd];
		Config::instance()->set<float>("max_inlier_dist", dists[dd]);
		Config::instance()->set<float>("dist_threshold", dists[dd]);

		for (int i = 0; i < graph.size(); i++)
		{
			if (graph[i])
			{
				delete graph[i];
			}
		}
		graph.clear();

		for (int i = 0; i < frame_count; i++)
		{
			Frame *frame = new Frame(rgbs[i], depths[i], "surf", Eigen::Matrix4f::Identity());
			frame->f->buildFlannIndex();
			graph.push_back(frame);
		}

		int min_matches = Config::instance()->get<int>("min_matches");
		float inlier_percent = Config::instance()->get<float>("min_inlier_p");
		float inlier_dist = Config::instance()->get<float>("max_inlier_dist");

		int threads = Config::instance()->get<int>("icpcuda_threads");
		int blocks = Config::instance()->get<int>("icpcuda_blocks");
		int width = Config::instance()->get<int>("image_width");
		int height = Config::instance()->get<int>("image_height");
		float cx = Config::instance()->get<float>("camera_cx");
		float cy = Config::instance()->get<float>("camera_cy");
		float fx = Config::instance()->get<float>("camera_fx");
		float fy = Config::instance()->get<float>("camera_fy");
		float depthFactor = Config::instance()->get<float>("depth_factor");
		float distThresh = Config::instance()->get<float>("dist_threshold");
		float angleThresh = Config::instance()->get<float>("angle_threshold");
		ICPOdometry *icpcuda = new ICPOdometry(width, height, cx, cy, fx, fy, depthFactor, distThresh, angleThresh);

		PairwiseRegister *reg_surf = new SurfRegister(min_matches, inlier_percent, inlier_dist);
		PairwiseRegister *reg_icpcuda = new IcpcudaRegister(icpcuda, threads, blocks, 20.0f);

		for (int i = 0; i < frame_count; i++)
		{
			if (i == 0)
			{
				graph[i]->relative_tran = Eigen::Matrix4f::Identity();
			}
			else
			{
				Eigen::Matrix4f tran = Eigen::Matrix4f::Identity();
				if (reg_surf->getTransformation(graph[i - 1], graph[i], tran))
				{
					graph[i]->relative_tran = tran;
				}
				else
				{
					// surf failed, run icpcuda
					reg_icpcuda->getTransformation(depths[i - 1].data, depths[i].data, tran);
					graph[i]->relative_tran = tran;
				}
			}
		}

		int accumulated_frame_count = 0, last_keyframe_id;
		Eigen::Matrix4f accumulated_transformation = Eigen::Matrix4f::Identity();
		bool is_last_frame_keyframe = false;
		float last_rational = 1;

		for (int i = 0; i < frame_count; i++)
		{
			bool isKeyframe = false;
			if (i == 0)
			{
				isKeyframe = true;
			}
			else
			{
				accumulated_frame_count++;
				accumulated_transformation = accumulated_transformation * trans[i];

				float rrr = reg_surf->getCorrespondencePercent(graph[last_keyframe_id], graph[i], accumulated_transformation);
				if (is_last_frame_keyframe)
				{
					last_rational = rrr;
				}
				rrr /= last_rational;
				if (rrr < Config::instance()->get<float>("keyframe_rational"))
				{
					isKeyframe = true;
				}
			}

			if (accumulated_frame_count >= Config::instance()->get<int>("max_keyframe_interval"))
			{
				isKeyframe = true;
			}

			graph[i]->relative_tran = accumulated_transformation;
			if (isKeyframe)
			{
				accumulated_frame_count = 0;
				accumulated_transformation = Eigen::Matrix4f::Identity();
				last_keyframe_id = i;
				keyframe_indices.push_back(i);
				keyframe_id.insert(pair<int, int>(i, keyframe_indices.size() - 1));
				is_last_frame_keyframe = true;
			}
			else
			{
				is_last_frame_keyframe = false;
			}
		}

		// lc test
		

		cout << endl;
	}
}

// F in candidate selection
void F_test()
{

}

int main()
{
//	keyframe_e2();
//	Registration_Result_Show();
	repeat_global_results();
//	time_analyze();
//	icp_test_1();
//	icp_test_2();
	return 0;
//	pairwise_results();
// 	repeat_pairwise_results();
// 	return 0;
// 	std::string filename;
// 	cout << "Oni file: ";
// 	cin >> filename;

// 	Status result = STATUS_OK;
// 
// 	result = OpenNI::initialize();
// 	if (result != STATUS_OK) return false;
// 
// 	Device oniDevice;
// 	VideoStream depthStream, colorStream;
// 
// 	result = oniDevice.open(filename.c_str());
// 	if (result != STATUS_OK) return false;
// 
// 	// depthstream and color stream
// 	result = depthStream.create(oniDevice, SENSOR_DEPTH);
// 	if (result != STATUS_OK) return false;
// 	unsigned short max_depth = (unsigned short)depthStream.getMaxPixelValue();
// 
// 	result = oniDevice.setImageRegistrationMode(IMAGE_REGISTRATION_DEPTH_TO_COLOR);
// 	if (result != STATUS_OK) return false;
// 
// 	result = colorStream.create(oniDevice, SENSOR_COLOR);
// 	if (result != STATUS_OK) return false;
// 
// 
// 	// playback control
// 	PlaybackControl *control = oniDevice.getPlaybackControl();
// 	control->setSpeed(-1);
// 
// 	result = depthStream.start();
// 	if (result != STATUS_OK) return false;
// 
// 	result = colorStream.start();
// 	if (result != STATUS_OK) return false;
// 
// 	VideoFrameRef oniDepth, oniColor;
// 
// 	for (int i = 0; i < control->getNumberOfFrames(colorStream); i++)
// 	{
// 		if (colorStream.readFrame(&oniColor) == STATUS_OK)
// 		{
// 			cv::Mat r(oniColor.getHeight(), oniColor.getWidth(), CV_8UC3, (void *)oniColor.getData());
// 			cv::cvtColor(r, r, CV_RGB2BGR);
// 			cv::imshow("rgb", r);
// 		}
// 
// 		if (depthStream.readFrame(&oniDepth) == STATUS_OK)
// 		{
// 			cv::Mat d(oniDepth.getHeight(), oniDepth.getWidth(), CV_16UC1, (void *)oniDepth.getData());
// 			cv::imshow("depth", d);
// 		}
// 		cv::waitKey(30);
// 	}
// 
// 	return 0;

	RGBDReader *reader = new KinectReader();
	reader->create(NULL);
	reader->start();
	cv::Mat r, d, t;

	cout << reader->intrColor.fx << "\t" << reader->intrColor.fy << "\t" << reader->intrColor.cx << "\t" << reader->intrColor.cy << endl;
	cout << reader->intrDepth.fx << "\t" << reader->intrDepth.fy << "\t" << reader->intrDepth.cx << "\t" << reader->intrDepth.cy << endl;
	int key = 0;
	while (key != 27)
	{
		if (reader->getNextFrame(r, d))
		{
			d.convertTo(t, CV_8U, 255.0 / 1000.0);
			cv::imshow("rgb", r);
			cv::imshow("depth", t);
		}
		key = cv::waitKey(15);
	}

	reader->stop();
	KinectReader::shutdown();

	return 0;
	//keyframe_test();
	//something();
	//icp_test();
	//return 0;
	//Ransac_Test();
	//return 0;
	//Ransac_Result_Show();
	//return 0;
	//Registration_Result_Show();
	//read_txt();
	//feature_test();
	//PlaneFittingTest();
	//continuousPlaneExtractingTest();
	//cudaTest();
	//plane_icp_test();
	//corr_test();
	//FeatureTest();
	//return 0;
	//Statistics();

// 	pairwise_results();
//	repeat_pairwise_results();
//	repeat_global_results();
//	time_test();

// 	cv::Mat img = cv::imread("E:/1305033538.912812.png", -1);
// 	cv::Mat img2(img.rows, img.cols, CV_8UC1);
// 	for (int i = 0; i < img.rows; i++)
// 	{
// 		for (int j = 0; j < img.cols; j++)
// 		{
// 			img2.at<unsigned char>(i, j) = img.at<ushort>(i, j) / 30000 * 255;
// 		}
// 	}
// 
// 	//cv::equalizeHist(img2, img2);
// 	cv::imshow("result", img2);
// 	cv::waitKey();
// 	return 0;

	std::string directory = "E:/school/data/";

	const int dcount = 9;
	std::string directories[dcount], names[dcount];
	directories[0] = directory + "rgbd_dataset_freiburg1_xyz/";
	directories[1] = directory + "rgbd_dataset_freiburg1_desk/";
	directories[2] = directory + "rgbd_dataset_freiburg1_desk2/";
	directories[3] = directory + "rgbd_dataset_freiburg1_floor/";
	directories[4] = directory + "rgbd_dataset_freiburg1_rpy/";
	directories[5] = directory + "rgbd_dataset_freiburg1_360/";
	directories[6] = directory + "rgbd_dataset_freiburg1_room/";
	directories[7] = directory + "rgbd_dataset_freiburg1_teddy/";
	directories[8] = directory + "rgbd_dataset_freiburg1_plant/";
// 	directories[0] = "G:/kinect data/rgbd_dataset_freiburg1_xyz/";
// 	directories[1] = "G:/kinect data/rgbd_dataset_freiburg1_desk/";
// 	directories[2] = "G:/kinect data/rgbd_dataset_freiburg1_room/";
// 	directories[3] = "G:/kinect data/rgbd_dataset_freiburg1_floor/";

	int dd, st, ed;
	float dist;
	string fname;

	// test
// 	dd = 1;
// 	st = -1;
// 	ed = -1;
// 	dist = 0.2;
// 	string test_type = "orb";
// 	int repeat_time = 10;
// 
// 	readData(directories[dd], st, ed);
// 	Config::instance()->set<float>("max_inlier_dist", dist);
// 	Config::instance()->set<int>("ransac_max_iteration", 1000);
// 
// 	stringstream ss;
// 
// 	for (int i = 0; i < repeat_time; i++)
// 	{
// 		ss.clear();
// 		ss.str("");
// 		ss << "E:/desk_ftof_ORB_" << i << ".txt";
// 		ofstream out1(ss.str());
// 		PairwiseRegistration(test_type, false, false, &out1);
// 
// 		ss.clear();
// 		ss.str("");
// 		ss << "E:/desk_ORB_" << i << ".txt";
// 		ofstream out2(ss.str());
// 		Eigen::Matrix4f last_tran = Eigen::Matrix4f::Identity();
// 		int k = 0;
// 		for (int j = 0; j < frame_count; j++)
// 		{
// 			graph[j]->tran = last_tran * graph[j]->relative_tran;
// 
// 			Eigen::Vector3f t = TranslationFromMatrix4f(graph[j]->tran);
// 			Eigen::Quaternionf q = QuaternionFromMatrix4f(graph[j]->tran);
// 
// 			out2 << fixed << setprecision(6) << timestamps[j]
// 				<< ' ' << t(0) << ' ' << t(1) << ' ' << t(2)
// 				<< ' ' << q.x() << ' ' << q.y() << ' ' << q.z() << ' ' << q.w() << endl;
// 
// 			if (k < keyframe_indices.size() && keyframe_indices[k] == j)
// 			{
// 				last_tran = graph[j]->tran;
// 				k++;
// 			}
// 		}
// 		out2.close();
// 	}
// 
// 	return 0;
	// test end
	
	// test2
// 	dd = 1;
// 	st = -1;
// 	ed = -1;
// 	dist = 0.03;
// 	string test_type = sift;
// 	int repeat_time = 20;
// 	stringstream ss;
// 	string test_name = "E:/desk_ftof_ORB_0.2.txt";
// 
// 	readData(directories[dd], st, ed);
// 	Config::instance()->set<float>("max_inlier_dist", dist);
// 	// 	Config::instance()->set<int>("ransac_max_iteration", 2000);
// 
// 	for (int i = 0; i < repeat_time; i++)
// 	{
// 		readPairwiseResult(test_name, "orb");
// 		ss.clear();
// 		ss.str("");
// 		ss << "E:/123/desk_SIFT_" << i << ".txt";
// 		ofstream out1(ss.str());
// 		GlobalRegistration(sift, nullptr, &out1);
// 	}
// 	return 0;
	// test2 end

	cout << "0: " << directories[0] << endl;
	cout << "1: " << directories[1] << endl;
	cout << "2: " << directories[2] << endl;
	cout << "3: " << directories[3] << endl;
	cout << "4: " << directories[4] << endl;
	cout << "5: " << directories[5] << endl;
	cout << "6: " << directories[6] << endl;
	cout << "7: " << directories[7] << endl;
	cout << "8: " << directories[8] << endl;

	cout << "directory: ";
	cin >> dd;

	
	cout << "st ed: ";
	cin >> st >> ed;
	readData(directories[dd], st, ed);

	int method;
	cout << "choose method:" << endl;
	cout << "0: pairwise registration & show result" << endl;
	cout << "1: read pairwise result & show result" << endl;
	cout << "2: pairwise registration & global registration & show result" << endl;
	cout << "3: read pairwise result & global registration & show result" << endl;
	cout << "4: read pairwise result & loop closure test" << endl;
//	cout << "5: read pairwise result & global registration gt & show result" << endl;
	cout << "6: show result by trajectory" << endl;
	cin >> method;

	if (method == 0 || method == 2)
	{
		string ftype;
		cout << "feature type(SURF, SIFT, ORB): ";
		cin >> ftype;

		cout << "max dist for inliers: ";
		cin >> dist;
		Config::instance()->set<float>("max_inlier_dist", dist);

		string fname;
		cout << "pairwise filename: ";
		cin >> fname;
		ofstream out(fname);

		string tname;
		cout << "trajectory filename: ";
		cin >> tname;
		ofstream out2(tname);
		PairwiseRegistration(ftype, &out, &out2);
	}
	else if (method == 1 || method == 3 || method == 4 || method == 5)
	{
		string ftype;
		cout << "feature type(SURF, SIFT, ORB): ";
		cin >> ftype;

		string fname;
		cout << "pairwise filename: ";
		cin >> fname;
		readPairwiseResult(fname, ftype);
	}
	
	if (method == 0 || method == 1)
	{
// 		string pname;
// 		cout << "pairwise result filename: ";
// 		cin >> pname;
// 		ofstream out(pname);
		ShowPairwiseResults(/*&out*/);
//		ShowPairwiseResultsEachKeyframe();
	}

	if (method == 2 || method == 3)
	{
		string gftype;
		cout << "graph feature type(SURF, SIFT, ORB): ";
		cin >> gftype;

		cout << "max dist for inliers: ";
		cin >> dist;
		Config::instance()->set<float>("max_inlier_dist", dist);

		string gname;
		cout << "global filename: ";
		cin >> gname;
		ofstream global_result(gname);

// 		string gtname;
// 		cout << "gtloop filenmae: ";
// 		cin >> gtname;
// 		ifstream gtloop_in(gtname);
		//GlobalRegistration(gftype, nullptr, &global_result);
	}

	if (method == 4)
	{
// 		string gftype;
// 		cout << "graph feature type(SURF, SIFT, ORB): ";
// 		cin >> gftype;
// 
// 		string gtname;
// 		cout << "ground-truth loop closure filename: ";
// 		cin >> gtname;
// 		ofstream gt_loop_result(gtname);
// 		FindGroundTruthLoop(directories[dd], gftype, &gt_loop_result);
// 		LoopAnalysis();
	}

	if (method == 5)
	{
// 		string gname;
// 		cout << "global gt filename: ";
// 		cin >> gname;
// 		ofstream global_result(gname);
// 		FindGroundTruthLoop(directories[dd]);
// 		GlobalWithAll(&global_result);
	}

	if (method == 6)
	{
		string trname;
		cout << "trajectory filename: ";
		cin >> trname;
		ifstream tr_result(trname);
		showResultWithTrajectory(&tr_result);
	}
}  
