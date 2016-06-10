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
#include "PointCloudCuda.h"

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

struct bfs_visitor_struct
{
	bool visit_filter_feat(const TLandmarkID lm_ID, const topo_dist_t cur_dist)
	{
		return false;
	}

	void visit_feat(const TLandmarkID lm_ID, const topo_dist_t cur_dist) { }

	bool visit_filter_kf(const TKeyFrameID kf_ID, const topo_dist_t cur_dist)
	{
		return true;
	}

	void visit_kf(const TKeyFrameID kf_ID, const topo_dist_t cur_dist) { }

	bool visit_filter_k2k(
		const TKeyFrameID current_kf,
		const TKeyFrameID next_kf,
		const SrbaGraphT::k2k_edge_t* edge,
		const topo_dist_t cur_dist)
	{
		return true;
	}

	void visit_k2k(
		const TKeyFrameID current_kf,
		const TKeyFrameID next_kf,
		const SrbaGraphT::k2k_edge_t* edge,
		const topo_dist_t cur_dist)
	{
		if (!is_keyframe_pose_set[next_kf])
		{
			SrbaGraphT::pose_t pose = edge->inv_pose;
			Eigen::Vector3f translation(pose.x(), pose.y(), pose.z());
			mrpt::math::CQuaternionDouble q;
			pose.getAsQuaternion(q);
			Eigen::Quaternionf quaternion(q.r(), q.x(), q.y(), q.z());
			Eigen::Matrix4f rt = transformationFromQuaternionsAndTranslation(quaternion, translation);

			keyframe_poses[next_kf] = keyframe_poses[current_kf] * rt.inverse();
			is_keyframe_pose_set[next_kf] = true;
		}
	}

	bool visit_filter_k2f(
		const TKeyFrameID current_kf,
		const SrbaGraphT::k2f_edge_t* edge,
		const topo_dist_t cur_dist)
	{
		return false;
	}

	void visit_k2f(
		const TKeyFrameID current_kf,
		const SrbaGraphT::k2f_edge_t *edge,
		const topo_dist_t cur_dist) { }
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

		Eigen::Matrix4f estimated_tran = Eigen::Matrix4f::Identity();
		Eigen::Vector3f t = estimated_tran.topRightCorner(3, 1);
		Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rot = estimated_tran.topLeftCorner(3, 3);

		icpcuda->getIncrementalTransformation(ret_t, ret_rot, t, rot, threads, blocks);

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
	string feature_type = "ORB";

	r[0] = cv::imread(rname[0]);
	d[0] = cv::imread(dname[0], -1);
	cloud[0] = ConvertToPointCloudWithoutMissingData(d[0], r[0], 0, 0);

	Frame *f[icount];
	f[0] = new Frame(r[0], d[0], feature_type, Eigen::Matrix4f::Identity());
	if (feature_type != "ORB")
		f[0]->f->buildFlannIndex();

	for (int i = 1; i < icount; i++)
	{
		r[i] = cv::imread(rname[i]);
		d[i] = cv::imread(dname[i], -1);
		cloud[i] = ConvertToPointCloudWithoutMissingData(d[i], r[i], i, i);

		f[i] = new Frame(r[i], d[i], feature_type, Eigen::Matrix4f::Identity());
		vector<cv::DMatch> matches;
		if (feature_type != "ORB")
			f[0]->f->findMatchedPairs(matches, f[i]->f);
		else
			f[0]->f->findMatchedPairsBruteForce(matches, f[i]->f);

		Eigen::Matrix4f tran = Eigen::Matrix4f::Identity();
		float rmse;
		int pc, pcorrc;
		vector<cv::DMatch> inliers;
		Eigen::Matrix<double, 6, 6> information;
		Feature::getTransformationByRANSAC(tran, information, pc, pcorrc, rmse, &inliers,
			f[0]->f, f[i]->f, nullptr, matches);
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
	string directory;
	int id_start, id_end, id_interval;

	ifstream infile("E:/test2.txt");
	getline(infile, directory);
	infile >> id_interval >> id_start >> id_end;
	
	stringstream ss;
	ss << directory << "/read.txt";
	ifstream cloud_infile(ss.str());
	string line;
	int k = 0, id = 0;
	while (getline(cloud_infile, line))
	{
		if (id < id_start)
		{
			id++;
			continue;
		}

		if (id <= id_end)
		{
			cout << k << endl;;
			int pos = line.find(' ');
			cv::Mat rgb = cv::imread(directory + "/" + line.substr(0, pos));
			cv::Mat depth = cv::imread(directory + "/" + line.substr(pos + 1, line.length() - pos - 1), -1);
			rgbs[k] = rgb;
			depths[k] = depth;
			PointCloudPtr cloud = ConvertToPointCloudWithoutMissingData(depth, rgb, k, k);
			cloud = DownSamplingByVoxelGrid(cloud, 0.01, 0.01, 0.01);
			clouds[k] = cloud;
		}
		else
		{
			break;
		}
		id++;
		k++;
	}
	cloud_infile.close();

	while (!infile.eof())
	{
		double timestamp;
		Eigen::Vector3f t;
		Eigen::Quaternionf q;
		infile >> timestamp >> t(0) >> t(1) >> t(2) >> q.x() >> q.y() >> q.z() >> q.w();
		Eigen::Matrix4f tran = transformationFromQuaternionsAndTranslation(q, t);
		trans.push_back(tran);
	}

	PointCloudPtr cloud_temp(new PointCloudT);
	Eigen::Matrix4f tran = Eigen::Matrix4f::Identity();
	combined_trans.push_back(tran);
	for (int i = 0; i < k; i++)
	{
		tran = trans[i];
		PointCloudPtr tran_cloud(new PointCloudT);
		pcl::transformPointCloud(*clouds[i], *tran_cloud, tran);
		*cloud_temp += *tran_cloud;
		if ((i + 1) % 50 == 0)
		{
			combined_trans.push_back(tran);
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

	now = 1;
	pairs_count = id_end + 1;
	PointCloudPtr cloud_all(new PointCloudT);
	PointCloudPtr tran_cloud(new PointCloudT);
	*cloud_all += *clouds[now - 1];
	pcl::transformPointCloud(*clouds[now], *tran_cloud, trans[now]);
	*cloud_all += *tran_cloud;

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
	Frame *f = new Frame(r, d, "SURF", Eigen::Matrix4f::Identity());
	
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
	cv::Mat vmap, nmap, pmap;
	icpcuda->getVMapCurr(vmap);
	icpcuda->getNMapCurr(nmap);
	icpcuda->getPMapCurr(pmap);
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

void plane_icp_test()
{
	const int icount = 2;
	std::string rname[icount], dname[icount];
 	rname[0] = "G:/kinect data/living_room_1/rgb/00594.jpg";
 	rname[1] = "G:/kinect data/living_room_1/rgb/00595.jpg";
//	rname[0] = "E:/lab/pcl/kinect data/living_room_1/rgb/00594.jpg";
//	rname[1] = "E:/lab/pcl/kinect data/living_room_1/rgb/00595.jpg";
	// 	rname[2] = "E:/lab/pcl/kinect data/living_room_1/rgb/00592.jpg";
	// 	rname[3] = "E:/lab/pcl/kinect data/living_room_1/rgb/00593.jpg";
	// 	rname[4] = "E:/lab/pcl/kinect data/living_room_1/rgb/00594.jpg";
	// 	rname[5] = "E:/lab/pcl/kinect data/living_room_1/rgb/00595.jpg";

 	dname[0] = "G:/kinect data/living_room_1/depth/00594.png";
 	dname[1] = "G:/kinect data/living_room_1/depth/00595.png";
//	dname[0] = "E:/lab/pcl/kinect data/living_room_1/depth/00594.png";
//	dname[1] = "E:/lab/pcl/kinect data/living_room_1/depth/00595.png";
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

	srand((unsigned)time(NULL));
	cv::Mat p[icount];
	vector<pair<int, int>> initials;
	int new_id = 0;
	int move[4][2] = { {1, 0}, {0, 1}, {-1, 0}, {0, -1} };
	vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f>> planes[icount];

	for (int i = 0; i < icount; i++)
	{
		cout << "frame " << i << " ";
		p[i] = cv::Mat(480, 640, CV_8SC1);
		for (int x = 0; x < 480; x++)
		{
			for (int y = 0; y < 640; y++)
			{
				p[i].at<char>(x, y) = -1;
			}
		}

		for (int j = 0; j < 20; j++)
		{
			cout << ", " << j;
			int r, c, count = 0;
			do 
			{
				r = rand() % 480;
				c = rand() % 640;
			} while ((p[i].at<char>(r, c) != -1 || d[i].at<ushort>(r, c) == 0) && count++ < 10);
			
			if (count >= 10)
			{
				continue;
			}

			int u_st = (r - 50) > 0 ? r - 50 : 0;
			int u_ed = (r + 50) < 480 ? r + 50 : 479;
			int v_st = (c - 50) > 0 ? c - 50 : 0;
			int v_ed = (c + 50) < 640 ? c + 50 : 639;
			initials.clear();
			memset(bfs_visited, 0, 480 * 640 * sizeof(bool));
			for (int x = u_st; x <= u_ed; x++)
			{
				for (int y = v_st; y <= v_ed; y++)
				{
					if (d[i].at<ushort>(x, y) != 0 && p[i].at<char>(x, y) == -1)
					{
						initials.push_back(pair<int, int>(x, y));
					}
					bfs_visited[x][y] = true;
				}
			}

			if (initials.size() > 10000)
			{
				Eigen::Vector4f plane;
				vector<pair<int, int>> inliers;
				bool success = Feature::getPlanesByRANSAC(plane, &inliers, d[i], initials);

				if (success && inliers.size() > 10000)
				{
					std::queue<pair<int, int>> q;
					for (int k = 0; k < inliers.size(); k++)
					{
						if (inliers[k].first == u_st || inliers[k].first == u_ed ||
							inliers[k].second == v_st || inliers[k].second == v_ed)
						{
							q.push(inliers[k]);
						}
						p[i].at<char>(inliers[k].first, inliers[k].second) = new_id;
					}

					while (!q.empty())
					{
						pair<int, int> now = q.front();
						q.pop();
						for (int k = 0; k < 4; k++)
						{
							int x = now.first + move[k][0];
							int y = now.second + move[k][1];
							if (x < 0 || y < 0 || x >= 480 || y >= 640 ||
								d[i].at<ushort>(x, y) == 0 || p[i].at<char>(x, y) != -1 || bfs_visited[x][y])
							{
								continue;
							}
							bfs_visited[x][y] = true;
							Eigen::Vector3f point;
							point(2) = ((double)d[i].at<ushort>(x, y)) / depthFactor;
							point(0) = (y - cx) * point(2) / fx;
							point(1) = (x - cy) * point(2) / fy;

							float dist = fabs(plane(0) * point(0) + plane(1) * point(1) + plane(2) * point(2) + plane(3));
							if (dist < 0.02)
							{
								p[i].at<char>(x, y) = new_id;
								q.push(pair<int, int>(x, y));
							}
						}
					}
					planes[i].push_back(plane);
					new_id++;
				}
			}
		}
		cout << endl;
	}

	icpcuda->initICPModel((unsigned short *)d[0].data, 20.0f, Eigen::Matrix4f::Identity());
	icpcuda->initICP((unsigned short *)d[1].data, 20.0f);

	Eigen::Matrix4f ret_tran = Eigen::Matrix4f::Identity();
	Eigen::Vector3f ret_t = ret_tran.topRightCorner(3, 1);
	Eigen::Matrix<float, 3, 3, Eigen::RowMajor> ret_rot = ret_tran.topLeftCorner(3, 3);

	Eigen::Matrix4f estimated_tran = Eigen::Matrix4f::Identity();
	Eigen::Vector3f t = estimated_tran.topRightCorner(3, 1);
	Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rot = estimated_tran.topLeftCorner(3, 3);

	icpcuda->getIncrementalTransformation(ret_t, ret_rot, t, rot, threads, blocks);

	ret_tran.topLeftCorner(3, 3) = ret_rot;
	ret_tran.topRightCorner(3, 1) = ret_t;

	std::vector<std::pair<int, int>> plane_corr;
	std::vector<int> plane_id_curr;
	std::vector<float> planes_lambda_prev;
	for (int i = 0; i < planes[0].size(); i++)
	{
		planes_lambda_prev.push_back(1);
		for (int j = 0; j < planes[1].size(); j++)
		{
			float nm = (ret_tran.transpose() * planes[1][j] - planes[0][i]).norm();
			if (nm < 0.1)
			{
				plane_corr.push_back(pair<int, int>(i, j));
				for (int x = 0; x < 480; x++)
				{
					for (int y = 0; y < 640; y++)
					{
						if (p[1].at<char>(x, y) == planes[0].size() + j)
						{
							p[1].at<char>(x, y) = i;
						}
					}
				}
			}
		}
	}

	std::vector<std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>> plane_inliers_curr;
	int plane_inlier_count = 10;
	for (int i = 0; i < plane_corr.size(); i++)
	{
		std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> inliers;
		for (int j = 0; j < plane_inlier_count; j++)
		{
			int r, c;
			do 
			{
				r = rand() % 480;
				c = rand() % 640;
			} while (p[1].at<char>(r, c) != plane_corr[i].first);
			Eigen::Vector3f point;
			point(2) = ((double)d[1].at<ushort>(r, c)) / depthFactor;
			point(0) = (c - cx) * point(2) / fx;
			point(1) = (r - cy) * point(2) / fy;
			inliers.push_back(point);
		}
		plane_inliers_curr.push_back(inliers);
		planes_lambda_prev.push_back(1);
	}

	icpcuda->initICPModel((unsigned short *)d[0].data, 20.0f, Eigen::Matrix4f::Identity());
	icpcuda->initICP((unsigned short *)d[1].data, 20.0f);
	icpcuda->initPlanes(planes[0], planes_lambda_prev, planes[1], plane_corr, plane_inlier_count, plane_inliers_curr);

	bool *ptmp = new bool[640 * 480];
	icpcuda->getPlaneMapCurr(ptmp);
	cv::Mat planemap(480, 640, CV_8UC3);
	for (int i = 0; i < 480; i++)
	{
		for (int j = 0; j < 640; j++)
		{
			planemap.at<cv::Vec3b>(i, j)[0] = ptmp[i * 640 + j] ? 255 : 0;
			planemap.at<cv::Vec3b>(i, j)[1] = 0;
			planemap.at<cv::Vec3b>(i, j)[2] = 0;
		}
	}
	delete ptmp;

	Eigen::Matrix4f ret_tran2 = Eigen::Matrix4f::Identity();
	ret_t = ret_tran2.topRightCorner(3, 1);
	ret_rot = ret_tran2.topLeftCorner(3, 3);

	icpcuda->getIncrementalTransformationWithPlane(ret_t, ret_rot, t, rot, threads, blocks);

	ret_tran2.topLeftCorner(3, 3) = ret_rot;
	ret_tran2.topRightCorner(3, 1) = ret_t;

	cv::Mat result[icount];
	for (int i = 0; i < icount; i++)
	{
		result[i] = cv::Mat(480, 640, CV_8UC3);
		for (int x = 0; x < 480; x++)
		{
			for (int y = 0; y < 640; y++)
			{
				if (p[i].at<char>(x, y) != -1)
				{
					result[i].at<cv::Vec3b>(x, y)[0] = rgb[p[i].at<char>(x, y)][0];
					result[i].at<cv::Vec3b>(x, y)[1] = rgb[p[i].at<char>(x, y)][1];
					result[i].at<cv::Vec3b>(x, y)[2] = rgb[p[i].at<char>(x, y)][2];
				}
				else
				{
					result[i].at<cv::Vec3b>(x, y)[0] = r[i].at<cv::Vec3b>(x, y)[0];
					result[i].at<cv::Vec3b>(x, y)[1] = r[i].at<cv::Vec3b>(x, y)[1];
					result[i].at<cv::Vec3b>(x, y)[2] = r[i].at<cv::Vec3b>(x, y)[2];
				}
			}
		}
	}

	cv::imshow("0", result[0]);
	cv::imshow("1", result[1]);
	cv::imshow("2", planemap);

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("test"));
	viewer->setBackgroundColor(0, 0, 0);
	viewer->addCoordinateSystem(1.0);
	viewer->initCameraParameters();

	PointCloudPtr tran_cloud(new PointCloudT);
	pcl::transformPointCloud(*cloud[1], *tran_cloud, ret_tran);
	PointCloudPtr cloud_all(new PointCloudT);
	*cloud_all = *cloud[0] + *tran_cloud;

	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud_all);
	viewer->addPointCloud<pcl::PointXYZRGB>(cloud_all, rgb, "result");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "result");

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer2(new pcl::visualization::PCLVisualizer("test2"));
	viewer2->setBackgroundColor(0, 0, 0);
	viewer2->addCoordinateSystem(1.0);
	viewer2->initCameraParameters();

	PointCloudPtr tran_cloud2(new PointCloudT);
	pcl::transformPointCloud(*cloud[1], *tran_cloud2, ret_tran2);
	PointCloudPtr cloud_all2(new PointCloudT);
	*cloud_all2 = *cloud[0] + *tran_cloud2;

	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb2(cloud_all2);
	viewer2->addPointCloud<pcl::PointXYZRGB>(cloud_all2, rgb2, "result2");
	viewer2->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "result2");

	while (!viewer->wasStopped() && !viewer2->wasStopped())
	{
		viewer->spinOnce(100);
		//boost::this_thread::sleep (boost::posix_time::microseconds (100000));
	}
	
// 	for (int i = 0; i < icount; i++)
// 	{
// 		cout << i << ": " << endl;
// 		for (int j = 0; j < planes[i].size(); j++)
// 		{
// 			cout << "\t" << planes[i][j] << endl;
// 		}
// 	}
}

void corr_test()
{
	const int icount = 2;
	std::string rname[icount], dname[icount];
	rname[0] = "G:/kinect data/living_room_1/rgb/00594.jpg";
	rname[1] = "G:/kinect data/living_room_1/rgb/00595.jpg";
	//rname[0] = "E:/lab/pcl/kinect data/living_room_1/rgb/00594.jpg";
	//rname[1] = "E:/lab/pcl/kinect data/living_room_1/rgb/00595.jpg";
	// 	rname[2] = "E:/lab/pcl/kinect data/living_room_1/rgb/00592.jpg";
	// 	rname[3] = "E:/lab/pcl/kinect data/living_room_1/rgb/00593.jpg";
	// 	rname[4] = "E:/lab/pcl/kinect data/living_room_1/rgb/00594.jpg";
	// 	rname[5] = "E:/lab/pcl/kinect data/living_room_1/rgb/00595.jpg";

	dname[0] = "G:/kinect data/living_room_1/depth/00594.png";
	dname[1] = "G:/kinect data/living_room_1/depth/00595.png";
	//dname[0] = "E:/lab/pcl/kinect data/living_room_1/depth/00594.png";
	//dname[1] = "E:/lab/pcl/kinect data/living_room_1/depth/00595.png";
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
	icpcuda->initICP((unsigned short *)d[1].data, 20.0f);

	Eigen::Matrix4f ret_tran = Eigen::Matrix4f::Identity();
	Eigen::Vector3f ret_t = ret_tran.topRightCorner(3, 1);
	Eigen::Matrix<float, 3, 3, Eigen::RowMajor> ret_rot = ret_tran.topLeftCorner(3, 3);

	Eigen::Matrix4f estimated_tran = Eigen::Matrix4f::Identity();
	Eigen::Vector3f t = estimated_tran.topRightCorner(3, 1);
	Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rot = estimated_tran.topLeftCorner(3, 3);

	icpcuda->getIncrementalTransformation(ret_t, ret_rot, t, rot, threads, blocks);

	ret_tran.topLeftCorner(3, 3) = ret_rot;
	ret_tran.topRightCorner(3, 1) = ret_t;

	PointCloudCuda *pcc = nullptr;
	if (pcc == nullptr)
		pcc = new PointCloudCuda(width, height, cx, cy, fx, fy, depthFactor);
	pcc->initPrev((unsigned short *)d[0].data, 20.0f);
	pcc->initCurr((unsigned short *)d[1].data, 20.0f);
	int point_count, point_corr_count;
	Eigen::Matrix<double, 6, 6> information;
	pcc->getCoresp(ret_t, ret_rot, information, point_count, point_corr_count, threads, blocks);

	cout << point_count << endl << point_corr_count << endl;

	cout << cloud[1]->size() << endl;
	int tehrjwlkt;
	cin >> tehrjwlkt;
}

void FeatureTest()
{

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
	types[0] = "SURF";
	types[1] = "SIFT";
	types[2] = "ORB";

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

void PairwiseRegistration(string feature_type = "SURF", bool FtoKF = true, bool useICP = false,
	ofstream *out = nullptr, ofstream *gout = nullptr, bool verbose = true)
{
	vector<int> failed_pairwise;
	bool save = out != nullptr;
	bool save_global = gout != nullptr;
	
	Frame *last_keyframe, *last_frame;
	bool last_frame_is_keyframe = false;
	float rational_reference = 1.0;
	vector<cv::DMatch> matches, inliers;
	float rmse;
	int pc, pcorrc;
	Eigen::Matrix4f relative_tran, last_tran, ac_tran, last_keyframe_transformation;
	Eigen::Matrix<double, 6, 6> information;
	int ac_count = 0;
	int mki = Config::instance()->get<int>("max_keyframe_interval");

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
	//ICPOdometry *icpcuda = nullptr;

	const float max_dist_m = Config::instance()->get<float>("max_dist_for_inliers");
	const float squared_max_dist_m = max_dist_m * max_dist_m;
	const float min_inlier_percent = Config::instance()->get<float>("min_inliers_percent");

	keyframe_indices.clear();
	keyframe_id.clear();
	for (int i = 0; i < graph.size(); i++)
	{
		if (graph[i])
		{
			delete graph[i];
		}
	}
	graph.clear();

	clock_t start, total_start = clock();

	for (int i = 0; i < frame_count; i++)
	{
		if (verbose)
			cout << "Frame " << i;
		start = clock();
		Frame *frame = new Frame(rgbs[i], depths[i], feature_type, Eigen::Matrix4f::Identity());
		if (verbose)
			cout << ", F: " << clock() - start << "ms";

		if (i == 0)
		{
			frame->relative_tran = Eigen::Matrix4f::Identity();
			if (feature_type != "ORB")
			{
				if (FtoKF)
					last_keyframe->f->buildFlannIndex();
				else
					frame->f->buildFlannIndex();
			}
				

			last_frame = frame;
			last_keyframe = frame;
			last_frame_is_keyframe = true;

			ac_count = 0;
			ac_tran = Eigen::Matrix4f::Identity();
			last_tran = Eigen::Matrix4f::Identity();
			last_keyframe_transformation = Eigen::Matrix4f::Identity();

			graph.push_back(frame);
			keyframe_indices.push_back(i);
			keyframe_id[i] = 0;
			
			if (verbose)
				cout << ", KF";
			if (save)
			{
				*out << i << " true false" << endl;
				*out << frame->relative_tran << endl;
			}
		}
		else
		{
			matches.clear();
			inliers.clear();

			bool isKeyframe = false;
			Eigen::Matrix4f tran;

			// f to f ransac
			start = clock();
			if (FtoKF)
			{
				if (feature_type == "ORB")
					last_keyframe->f->findMatchedPairsBruteForce(matches, frame->f);
				else
					last_keyframe->f->findMatchedPairs(matches, frame->f);
			}
			else
			{
				if (feature_type == "ORB")
					last_frame->f->findMatchedPairsBruteForce(matches, frame->f);
				else
					last_frame->f->findMatchedPairs(matches, frame->f);
			}
			
			if (verbose)
				cout << ", M: " << clock() - start << "ms";

			start = clock();
			bool ransac;
			if (FtoKF)
			{
				ransac = Feature::getTransformationByRANSAC(tran, information,
					pc, pcorrc, rmse, &inliers, last_keyframe->f, frame->f, nullptr, matches);
			}
			else
			{
				ransac = Feature::getTransformationByRANSAC(tran, information,
					pc, pcorrc, rmse, &inliers, last_frame->f, frame->f, nullptr, matches);
			}
			
			if (verbose)
				cout << ", R: " << clock() - start << "ms";

			// is new keyframe?
			start = clock();
			if (ransac)
			{
				relative_tran = tran;
				if (verbose)
					cout << ", " << matches.size() << ", " << inliers.size();

				if (!FtoKF)
				{
					matches.clear();
					inliers.clear();
					if (feature_type == "ORB")
						last_keyframe->f->findMatchedPairsBruteForce(matches, frame->f);
					else
						last_keyframe->f->findMatchedPairs(matches, frame->f);

// 					Feature::getTransformationByRANSAC(tran, information, pc, pcorrc, rmse,
// 						&inliers, last_keyframe->f, frame->f, nullptr, matches);
					Feature::computeInliersAndError(inliers, rmse, nullptr, matches,
						ac_tran * relative_tran,
						last_keyframe->f, frame->f);
				}

				float rrr;
				if (matches.size() > 0)
					rrr = (float)inliers.size() / matches.size();
				else
					rrr = 0;
				if (last_frame_is_keyframe)
				{
					if (rrr > 0)
						rational_reference = rrr;
					else
						rational_reference = 10000;
				}
				rrr /= rational_reference;
				if (verbose)
					cout << ", " << rrr;
				if (rrr < Config::instance()->get<float>("keyframe_rational"))
				{
					isKeyframe = true;
				}
			}
			else
			{
				Eigen::Matrix4f tran2 = Eigen::Matrix4f::Identity();
				bool last_frame_success = false;
				if (FtoKF && !last_frame_is_keyframe)
				{
					if (feature_type != "ORB")
						last_frame->f->buildFlannIndex();
					matches.clear();
					inliers.clear();

					if (feature_type == "ORB")
						last_frame->f->findMatchedPairsBruteForce(matches, frame->f);
					else
						last_frame->f->findMatchedPairs(matches, frame->f);

					last_frame_success = Feature::getTransformationByRANSAC(tran2, information,
						pc, pcorrc, rmse, &inliers, last_frame->f, frame->f, nullptr, matches);
				}

				
				if (FtoKF && last_frame_success)
				{
					isKeyframe = true;
					relative_tran = ac_tran * tran2;
				}
				else
				{
					if (useICP)
					{
						icpcuda->initICPModel((unsigned short *)depths[i - 1].data, 20.0f, Eigen::Matrix4f::Identity());
						icpcuda->initICP((unsigned short *)depths[i].data, 20.0f);

						tran2 = Eigen::Matrix4f::Identity();
						Eigen::Vector3f t = tran2.topRightCorner(3, 1);
						Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rot = tran2.topLeftCorner(3, 3);

						Eigen::Matrix4f estimated_tran = Eigen::Matrix4f::Identity();
						Eigen::Vector3f estimated_t = estimated_tran.topRightCorner(3, 1);
						Eigen::Matrix<float, 3, 3, Eigen::RowMajor> estimated_rot = estimated_tran.topLeftCorner(3, 3);

						icpcuda->getIncrementalTransformation(t, rot, estimated_t, estimated_rot, threads, blocks);

						tran2.topLeftCorner(3, 3) = rot;
						tran2.topRightCorner(3, 1) = t;

						if (FtoKF)
						{
							isKeyframe = true;
							relative_tran = ac_tran * tran2;
						}
						else
						{
							relative_tran = tran2;

// 							matches.clear();
// 							inliers.clear();
// 							if (feature_type == "ORB")
// 								last_keyframe->f->findMatchedPairsBruteForce(matches, frame->f);
// 							else
// 								last_keyframe->f->findMatchedPairs(matches, frame->f);
// 
// // 							Feature::getTransformationByRANSAC(tran, information, pc, pcorrc, rmse,
// // 								&inliers, last_keyframe->f, frame->f, nullptr, matches);
// 							Feature::computeInliersAndError(inliers, rmse, nullptr, matches,
// 								ac_tran * relative_tran,
// 								last_keyframe->f, frame->f);
// 
// 							float rrr;
// 							if (matches.size() > 0)
// 								rrr = (float)inliers.size() / matches.size();
// 							else
// 								rrr = 0;
// 							if (last_frame_is_keyframe)
// 							{
// 								if (rrr > 0)
// 									rational_reference = rrr;
// 								else
// 									rational_reference = 10000;
// 							}
// 							rrr /= rational_reference;
// 							if (verbose)
// 								cout << ", " << rrr;
// 							if (rrr < Config::instance()->get<float>("keyframe_rational"))
// 							{
								isKeyframe = true;
//							}
						}
					}
					else
					{
						relative_tran = last_tran;
						if (verbose)
							cout << ", failed";
						failed_pairwise.push_back(i);
						frame->ransac_failed = true;
						isKeyframe = true;
					}
				}
			}

			last_tran = relative_tran;
			ac_count++;
			if (FtoKF)
				ac_tran = relative_tran;
			else
			{
				ac_tran = ac_tran * relative_tran;
				relative_tran = ac_tran;
			}

			if (ac_count > mki)
			{
				isKeyframe = true;
			}
			if (verbose)
				cout << ", KF: " << clock() - start << "ms";

			if (isKeyframe)
			{
				ac_count = 0;
				ac_tran = Eigen::Matrix4f::Identity();
			}

			frame->relative_tran = relative_tran;

			if (!FtoKF)
			{
				if (!last_frame_is_keyframe)
				{
					delete last_frame->f;
					last_frame->f = nullptr;
				}
				if (feature_type != "ORB")
					frame->f->buildFlannIndex();
			}
			last_frame = frame;

			if (isKeyframe)
			{
				if (verbose)
					cout << ", is Keyframe";
				delete last_keyframe->f;
				last_keyframe->f = nullptr;
				if (FtoKF && feature_type != "ORB")
					frame->f->buildFlannIndex();
				last_keyframe = frame;
				last_frame_is_keyframe = true;

				keyframe_indices.push_back(i);
				keyframe_id[i] = keyframe_indices.size() - 1;

				if (save)
				{
					*out << i << " true " << (frame->ransac_failed ? "true" : "false") << endl;
					*out << frame->relative_tran << endl;
				}
			}
			else
			{
				last_frame_is_keyframe = false;
				if (save)
				{
					*out << i << " false false" << endl;
					*out << frame->relative_tran << endl;
				}
			}
			graph.push_back(frame);
		}
		if (verbose)
			cout << endl;
	}

	if (last_frame_is_keyframe)
	{
		delete last_keyframe->f;
		last_keyframe->f = nullptr;
	}

	if (save)
	{
		out->close();
	}

	if (save_global)
	{
		Eigen::Matrix4f last_tran = Eigen::Matrix4f::Identity();
		int k = 0;
		for (int j = 0; j < frame_count; j++)
		{
			graph[j]->tran = last_tran * graph[j]->relative_tran;

			Eigen::Vector3f t = TranslationFromMatrix4f(graph[j]->tran);
			Eigen::Quaternionf q = QuaternionFromMatrix4f(graph[j]->tran);

			*gout << fixed << setprecision(6) << timestamps[j]
				<< ' ' << t(0) << ' ' << t(1) << ' ' << t(2)
				<< ' ' << q.x() << ' ' << q.y() << ' ' << q.z() << ' ' << q.w() << endl;

			if (k < keyframe_indices.size() && keyframe_indices[k] == j)
			{
				last_tran = graph[j]->tran;
				k++;
			}
		}
		gout->close();
	}

	if (verbose)
	{
		cout << failed_pairwise.size() << endl;
		for (int i = 0; i < failed_pairwise.size(); i++)
		{
			cout << failed_pairwise[i] << endl;
		}
	}
	delete icpcuda;
}

void readPairwiseResult(string filename, string feature_type = "SURF")
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

void GlobalRegistration(string graph_ftype = "SIFT",
	ifstream *gtloop_in = nullptr,
	ofstream *result_out = nullptr,
	ofstream *log_out = nullptr)
{
	bool save_result = result_out != nullptr;
	bool save_log = log_out != nullptr;
	bool read_gtloop = gtloop_in != nullptr;

	cout << "Global registration" << endl;
 	SlamEngine engine;
	engine.setUsingHogmanOptimizer(false);
	engine.setUsingSrbaOptimzier(false);
	engine.setUsingRobustOptimzier(true);
	for (int i = 0; i < frame_count; i++)
	{
		bool keyframe = false;
		if (keyframe_id.find(i) != keyframe_id.end())
		{
			keyframe = true;
			Eigen::Matrix4f tran = graph[i]->relative_tran;
			bool rf = graph[i]->ransac_failed;
			delete graph[i];
			graph[i] = new Frame(rgbs[i], depths[i], graph_ftype, Eigen::Matrix4f::Identity());
			graph[i]->relative_tran = tran;
			graph[i]->ransac_failed = rf;
			if (graph_ftype != "ORB")
			{
				graph[i]->f->buildFlannIndex();
			}
		}
		engine.AddGraph(graph[i], clouds[i], keyframe, timestamps[i]);
	}

	if (read_gtloop)
	{
		// read gtloop
		int gtloop_count;
		*gtloop_in >> gtloop_count;
		vector<pair<int, int>> gtloops;
		for (int i = 0; i < gtloop_count; i++)
		{
			int a, b;
			*gtloop_in >> a >> b;
			gtloops.push_back(pair<int, int>(a, b));
		}

		// analyze
		cout << "keyframe to detect loop closure: " << endl;
		for (int i = 0; i < engine.id_detect_lc.size(); i++)
		{
			cout << keyframe_id[engine.id_detect_lc[i]] << " " << engine.id_detect_lc[i] << endl;
		}

		for (int i = 1; i < keyframe_indices.size(); i++)
		{
			int gtc = 0;
			vector<int> others;
			for (int j = 0; j < gtloop_count; j++)
			{
				if (gtloops[j].second == keyframe_indices[i])
				{
					gtc++;
					others.push_back(gtloops[j].first);
				}
			}

			cout << i << " " << keyframe_indices[i];
			Eigen::Vector3f current_pose =  engine.current_positions[i];
			vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> position_lists = engine.positions[i];
			for (int j = 0; j < gtc; j++)
			{
				Eigen::Vector3f pose =  position_lists[keyframe_id[others[j]]];
				cout << ", " << others[j] << " " << (current_pose - pose).norm();
				bool found = false;
				for (int k = 0; k < engine.id_detect_lc.size(); k++)
				{
					if (others[j] == engine.id_detect_lc[k])
					{
						found = true;
						break;
					}
				}
				cout << " " << (found ? "T" : "F");
			}
			cout << endl;
		}
	}
	
// 	for (int i = 1; i < keyframe_indices.size(); i++)
// 	{
// 		for (int j = 0; j < i; j++)
// 		{
// 			if ()
// 			{
// 			}
// 		}
// 	}

	// end
	
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

// 	vector<pair<int, int>> loops = engine.GetLoop();
// 	cout << "Loop: " << loops.size() << endl;
// 	for (int i = 0; i < loops.size(); i++)
// 	{
// 		cout << loops[i].first << " " << loops[i].second << endl;
// 	}
// 
	cout << "Getting scene" << endl;
	PointCloudPtr cloud = engine.GetScene();
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	viewer->setBackgroundColor(0, 0, 0);
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
	//viewer->registerKeyboardCallback(KeyboardEventOccurred, (void*)&viewer);
	viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");
	viewer->addCoordinateSystem(1.0);
	viewer->initCameraParameters();

	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
	}
}

void FindGroundTruthLoop(string directory, string feature_type = "SURF", ofstream *gt_out = nullptr)
{
	// ground_truth
	stringstream ss;
	ss.clear();
	ss.str("");
	ss << directory << "/groundtruth.txt";
	ifstream g_in(ss.str());

	string line;
	Eigen::Matrix4f tran;

	for (int i = 0; i < keyframe_indices.size(); i++)
	{
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
			if (fabs(ts - timestamps[keyframe_indices[i]]) < 1e-2)
			{
				// ground-truth found
				gt_found = true;
				tran = transformationFromQuaternionsAndTranslation(q, t);
				break;
			}
			else if (ts > timestamps[keyframe_indices[i]])
			{
				// ground-truth not found
				tran = Eigen::Matrix4f::Identity();
				break;
			}
		}

		if (!gt_found)
		{
			continue;
		}

		gt_trans.push_back(tran);
	}
	g_in.close();

	int threads = Config::instance()->get<int>("icpcuda_threads");
	int blocks = Config::instance()->get<int>("icpcuda_blocks");
	int width = Config::instance()->get<int>("image_width");
	int height = Config::instance()->get<int>("image_height");
	float cx = Config::instance()->get<float>("camera_cx");
	float cy = Config::instance()->get<float>("camera_cy");
	float fx = Config::instance()->get<float>("camera_fx");
	float fy = Config::instance()->get<float>("camera_fy");
	float depthFactor = Config::instance()->get<float>("depth_factor");
	float distThrehold = Config::instance()->get<float>("dist_threshold");
	float angleThrehold = Config::instance()->get<float>("angle_threshold");
	PointCloudCuda *pcc = new PointCloudCuda(width, height, cx, cy, fx, fy, depthFactor, distThrehold, angleThrehold);

	int point_count, point_corr_count;
	Eigen::Matrix<double, 6, 6> information;
	// find ground-truth loop closure, pair-wise
	for (int i = 0; i < keyframe_indices.size() - 1; i++)
	{
		pcc->initPrev((unsigned short *)depths[keyframe_indices[i]].data, 20.0f);
		for (int j = i + 1; j < keyframe_indices.size(); j++)
		{
			pcc->initCurr((unsigned short *)depths[keyframe_indices[j]].data, 20.0f);
			Eigen::Matrix4f relative_tran = gt_trans[i].inverse() * gt_trans[j];
			Eigen::Vector3f t = relative_tran.topRightCorner(3, 1);
			Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rot = relative_tran.topLeftCorner(3, 3);
			cv::Mat pairs;
			pcc->getCorespPairs(t, rot, information, pairs, point_count, point_corr_count, threads, blocks);

			if ((float)point_corr_count / point_count >= 0.3)
			{
				gt_loop.push_back(pair<int, int>(i, j));
				gt_loop_corr.push_back(pairs);
			}
		}
	}

	if (gt_out != nullptr)
	{
		*gt_out << gt_loop.size() << endl;
		for (int i = 0; i < gt_loop.size(); i++)
		{
			*gt_out << keyframe_indices[gt_loop[i].first] << " " << keyframe_indices[gt_loop[i].second] << endl;
		}
		gt_out->close();
	}
	cout << gt_loop.size() << endl;

// 	for (int i = 0; i < gt_loop.size(); i++)
// 	{
// 		cv::Mat result(480, 1280, CV_8UC3);
// 		for (int u = 0; u < 480; u++)
// 		{
// 			for (int v = 0; v < 640; v++)
// 			{
// 				result.at<cv::Vec3b>(u, v) = rgbs[keyframe_indices[gt_loop[i].first]].at<cv::Vec3b>(u, v);
// 				result.at<cv::Vec3b>(u, v + 640) = rgbs[keyframe_indices[gt_loop[i].second]].at<cv::Vec3b>(u, v);
// 			}
// 		}
// 
// 		int show_count = 0;
// 		for (int u = 0; u < 480; u++)
// 		{
// 			for (int v = 0; v < 640; v++)
// 			{
// 				int cu, cv;
// 				cu = gt_loop_corr[i].at<cv::Vec2i>(u, v)[0];
// 				cv = gt_loop_corr[i].at<cv::Vec2i>(u, v)[1];
// 				if (cu != -1 && cv != -1)
// 				{
// 					if (show_count % 1000 == 0)
// 					{
// 						//cout << u << " " << v << ", " << cu << " " << cv << endl;
// 						cv::circle(result, cv::Point(cv, cu), 3, cv::Scalar(0, 255, 0), 2);
// 						cv::circle(result, cv::Point(v + 640, u), 3, cv::Scalar(0, 255, 0), 2);
// 						cv::line(result, cv::Point(cv, cu), cv::Point(v + 640, u), cv::Scalar(0, 255, 0));
// 					}
// 					show_count++;
// 				}
// 			}
// 		}
// 		cv::imshow("result", result);
// 		cv::waitKey();
// 	}

	vector<PointCloudPtr> full_key_clouds;
	for (int i = 0; i < keyframe_indices.size(); i++)
	{
		graph[keyframe_indices[i]]->f = new Feature();
		graph[keyframe_indices[i]]->f->extract(rgbs[keyframe_indices[i]],
			depths[keyframe_indices[i]], feature_type);

		if (feature_type != "ORB")
			graph[keyframe_indices[i]]->f->buildFlannIndex();
		PointCloudPtr cloud = ConvertToPointCloud(depths[keyframe_indices[i]],
			rgbs[keyframe_indices[i]], timestamps[keyframe_indices[i]], keyframe_indices[i]);
		full_key_clouds.push_back(cloud);
	}

	double pp = 0, pq = 0;
	vector<cv::DMatch> matches;
	float rmse;
	int pc, pcorrc;
	vector<cv::DMatch> inliers;
	int mleaf = Config::instance()->get<int>("kdtree_max_leaf");
	for (int i = 0; i < gt_loop.size(); i++)
	{
		cout << keyframe_indices[gt_loop[i].first] << " " << keyframe_indices[gt_loop[i].second];

		matches.clear();
		inliers.clear();
		if (feature_type == "ORB")
			graph[keyframe_indices[gt_loop[i].first]]->f->findMatchedPairsBruteForce(matches,
				graph[keyframe_indices[gt_loop[i].second]]->f);
		else
			graph[keyframe_indices[gt_loop[i].first]]->f->findMatchedPairs(matches,
				graph[keyframe_indices[gt_loop[i].second]]->f);

		cout << " " << matches.size();
		pcc->initPrev((unsigned short *)depths[keyframe_indices[gt_loop[i].first]].data, 20.0f);
		pcc->initCurr((unsigned short *)depths[keyframe_indices[gt_loop[i].second]].data, 20.0f);
		if (Feature::getTransformationByRANSAC(tran, information, pc, pcorrc, rmse, &inliers,
			graph[keyframe_indices[gt_loop[i].first]]->f,
			graph[keyframe_indices[gt_loop[i].second]]->f,
			pcc, matches))
		{
			PointCloudPtr cloud_tran(new PointCloudT);
			pcl::transformPointCloud(*full_key_clouds[gt_loop[i].second], *cloud_tran, tran);
			double rmse = 0.0;
			int cov_count = 0;
			for (int j = 0; j < height; j++)
			{
				for (int k = 0; k < width; k++)
				{
					int cx, cy;
					cx = gt_loop_corr[i].at<cv::Vec2i>(j, k)[0];
					cy = gt_loop_corr[i].at<cv::Vec2i>(j, k)[1];
					if (cx != -1 && cy != -1)
					{
						Eigen::Vector3f a;
						Eigen::Vector3f b;
						a(0) = full_key_clouds[gt_loop[i].first]->points[cx * width + cy].x;
						a(1) = full_key_clouds[gt_loop[i].first]->points[cx * width + cy].y;
						a(2) = full_key_clouds[gt_loop[i].first]->points[cx * width + cy].z;
						b(0) = cloud_tran->points[j * width + k].x;
						b(1) = cloud_tran->points[j * width + k].y;
						b(2) = cloud_tran->points[j * width + k].z;
						rmse += (a - b).squaredNorm();
						cov_count++;
					}
				}
			}
			rmse /= cov_count;
			cout << " found";
			if (rmse < distThrehold * distThrehold)
			{
				cout << " true";
				pp = pp + 1;
			}
			pq = pq + 1;

			
//			*cloud_tran += *full_key_clouds[gt_loop[i].first];		
// 			boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
// 			viewer->setBackgroundColor(0, 0, 0);
// 			//viewer->registerKeyboardCallback(KeyboardEventOccurred, (void*)&viewer);
// 			viewer->addCoordinateSystem(1.0);
// 			viewer->initCameraParameters();
// 			pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud_tran);
// 			viewer->addPointCloud<pcl::PointXYZRGB>(cloud_tran, rgb, "cloud");
// 			viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");
// 			while (!viewer->wasStopped())
// 			{
// 				viewer->spinOnce(100);
// 			}
		}
		cout << endl;
	}

	cout << "precision: " << pp / pq << "recall: " << pp / gt_loop.size() << endl;
}

void LoopAnalysis()
{
	if (keyframe_indices.size() <= 0)
		return;

	Eigen::Matrix4f first_inv = gt_trans[0].inverse();
	vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> r_gt_trans;
	r_gt_trans.push_back(Eigen::Matrix4f::Identity());

	for (int i = 1; i < keyframe_indices.size(); i++)
	{
		r_gt_trans.push_back(first_inv * gt_trans[i]);
	}

	vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> keyframe_poses;
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	stringstream ss;
	PointCloudPtr cloud(new PointCloudT);
	for (int i = 0; i < keyframe_indices.size(); i++)
	{
		pcl::PointXYZRGB pt;
		Eigen::Vector3f translation = TranslationFromMatrix4f(r_gt_trans[i]);
		keyframe_poses.push_back(translation);
		pt.x = translation(0) + 0.5;
		pt.y = translation(1) + 0.5;
		pt.z = translation(2) + 0.55;
		pt.r = 0;
		pt.g = 255;
		pt.b = 0;
		pt.a = 255;
		ss.clear();
		ss.str("");
		ss << i << " " << keyframe_indices[i];
		viewer->addText3D(ss.str(), pt, 0.01);
		pt.z = translation(2) + 0.5;
		cloud->push_back(pt);
	}

	cout << "dists: " << endl;
	for (int i = 0; i < gt_loop.size(); i++)
	{
		Eigen::Vector3f a = keyframe_poses[gt_loop[i].first];
		Eigen::Vector3f b = keyframe_poses[gt_loop[i].second];
		cout << gt_loop[i].first << " " << gt_loop[i].second << " " << (a - b).norm() << endl;
	}

	viewer->setBackgroundColor(0, 0, 0);
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
	//viewer->registerKeyboardCallback(KeyboardEventOccurred, (void*)&viewer);
	viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "cloud");
	viewer->addCoordinateSystem(1.0);
	viewer->initCameraParameters();

	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
	}
}

void GlobalWithAll(ofstream *result_out = nullptr, ofstream *log_out = nullptr)
{
	bool save_result = result_out != nullptr;
	bool save_log = log_out != nullptr;

	cout << "Global registration with gt loop(while transformation estimated by ransac)" << endl;
	SlamEngine engine;
	engine.setUsingHogmanOptimizer(false);
	engine.setUsingSrbaOptimzier(false);
	engine.setUsingRobustOptimzier(true);

	for (int i = 0; i < keyframe_indices.size(); i++)
	{
		graph[keyframe_indices[i]]->f->buildFlannIndex();
	}

	for (int i = 0; i < frame_count; i++)
	{
		bool keyframe = false;
		if (keyframe_id.find(i) != keyframe_id.end())
		{
			keyframe = true;
		}
		vector<int> loop;
		if (keyframe)
		{
			for (int j = 0; j < gt_loop.size(); j++)
			{
				if (gt_loop[j].second == keyframe_id[i])
				{
					loop.push_back(keyframe_indices[gt_loop[j].first]);
				}
			}
		}
		engine.AddGraph(graph[i], clouds[i], keyframe, true, loop, timestamps[i]);
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

	cout << "Getting scene" << endl;
	PointCloudPtr cloud = engine.GetScene();
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	viewer->setBackgroundColor(0, 0, 0);
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
	//viewer->registerKeyboardCallback(KeyboardEventOccurred, (void*)&viewer);
	viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");
	viewer->addCoordinateSystem(1.0);
	viewer->initCameraParameters();

	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
	}
}

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

// 	directories[0] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_rpy";
// 	directories[1] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_teddy/";
// 	directories[2] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_plant/";

	names[0] = "xyz";
	names[1] = "desk";
	names[2] = "desk2";
	names[3] = "360";
	names[4] = "room";
	names[5] = "floor";
	names[6] = "rpy";
	names[7] = "teddy";
	names[8] = "plant";

// 	names[0] = "rpy";
// 	names[1] = "teddy";
// 	names[2] = "plant";

	const int fcount = 3;
	std::string ftypes[fcount];
	ftypes[0] = "ORB";
	ftypes[1] = "SIFT";
	ftypes[2] = "SURF";

	const int sdcount = 10;
	float dists[sdcount];
	dists[0] = 0.01;
	dists[1] = 0.02;
	dists[2] = 0.03;
	dists[3] = 0.05;
	dists[4] = 0.08;
	dists[5] = 0.1;
	dists[6] = 0.15;
	dists[7] = 0.2;
	dists[8] = 0.25;
	dists[9] = 0.30;

	float siftdists[dcount] = { 
		0.1,
		0.03,
		0.02,
		0.3,
		0.1,
		0.25,
		0.1,
		0.25,
		0.3 };
	float surfdists[dcount] = {
		0.03,
		0.03,
		0.1,
		0.25,
		0.03,
		0.3,
		0.3,
		0.25,
		0.2	};

	int st = -1, ed = -1;
	stringstream ss;

	for (int dd = 0; dd < dcount; dd++)
	{
		readData(directories[dd], st, ed);

		for (int sd = 0; sd < sdcount; sd++)
		{
			cout << "\t dists: " << dists[sd];
			Config::instance()->set<float>("max_dist_for_inliers", dists[sd]);

			for (int f = 0; f < fcount; f++)
			{
				if (ftypes[f] == "SIFT" && dists[sd] != siftdists[dd] ||
					ftypes[f] == "SURF" && dists[sd] != surfdists[dd])
				{
					continue;
				}
				cout << "\t" << ftypes[f];
				ss.clear();
				ss.str("");
				ss << "E:/tempresults/" << names[dd] << "/"
					<< names[dd] << "_ftof_" << ftypes[f] << "_" << dists[sd] << ".txt";
				ofstream out1(ss.str());

				ss.clear();
				ss.str("");
				ss << "E:/tempresults/" << names[dd] << "/"
					<< names[dd] << "_" << ftypes[f] << "_" << dists[sd] << ".txt";
				ofstream out2(ss.str());

				PairwiseRegistration(ftypes[f], false, true, &out1, &out1, false);

// 				Eigen::Matrix4f last_tran = Eigen::Matrix4f::Identity();
// 				int k = 0;
// 				for (int j = 0; j < frame_count; j++)
// 				{
// 					graph[j]->tran = last_tran * graph[j]->relative_tran;
// 
// 					Eigen::Vector3f t = TranslationFromMatrix4f(graph[j]->tran);
// 					Eigen::Quaternionf q = QuaternionFromMatrix4f(graph[j]->tran);
// 
// 					out2 << fixed << setprecision(6) << timestamps[j]
// 						<< ' ' << t(0) << ' ' << t(1) << ' ' << t(2)
// 						<< ' ' << q.x() << ' ' << q.y() << ' ' << q.z() << ' ' << q.w() << endl;
// 
// 					if (k < keyframe_indices.size() && keyframe_indices[k] == j)
// 					{
// 						last_tran = graph[j]->tran;
// 						k++;
// 					}
// 				}
// 				out2.close();
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

// 	directories[0] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_rpy";
// 	directories[1] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_teddy/";
// 	directories[2] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_plant/";

	names[0] = "xyz";
	names[1] = "desk";
	names[2] = "desk2";
	names[3] = "360";
	names[4] = "room";
	names[5] = "floor";
 	names[6] = "rpy";
 	names[7] = "teddy";
 	names[8] = "plant";

// 	names[0] = "rpy";
// 	names[1] = "teddy";
// 	names[2] = "plant";

	const int fcount = 1;
	std::string ftypes[fcount];
//	ftypes[0] = "ORB";
// 	ftypes[1] = "SIFT";
 	ftypes[0] = "SURF";

	const int sdcount = 1;
	float dists[dcount][sdcount] = {0};
	dists[0][0] = 0.03;	// xyz
	dists[1][0] = 0.03;	// desk
	dists[2][0] = 0.1;  // desk2
	dists[3][0] = 0.25; // 360
	dists[4][0] = 0.3; // room
	dists[5][0] = 0.3;  // floor
	dists[6][0] = 0.3;  // rpy
	dists[7][0] = 0.25;  // teddy
	dists[8][0] = 0.2;  // plants

	int repeat_time = 3;
	int st = -1, ed = -1;
	stringstream ss;

	for (int dd = 0; dd < dcount; dd++)
	{
		readData(directories[dd], st, ed, false);

		for (int sd = 0; sd < sdcount; sd++)
		{
			cout << "\t dists: " << dists[dd][sd];
			Config::instance()->set<float>("max_dist_for_inliers", dists[dd][sd]);

			for (int f = 0; f < fcount; f++)
			{
				cout << "\t" << ftypes[f];

				for (int i = 0; i < repeat_time; i++)
				{
					ss.clear();
					ss.str("");
					ss << "E:/tempresults/" << names[dd] << "/repeat/"
						<< names[dd] << "_repeat_" << ftypes[f] << "_" << repeat_time * sd + i << ".txt";
					ofstream out1(ss.str());

					ss.clear();
					ss.str("");
					ss << "E:/tempresults/" << names[dd] << "/repeat/"
						<< names[dd] << "_" << ftypes[f] << "_" << repeat_time * sd + i << ".txt";
					ofstream out2(ss.str());

					PairwiseRegistration(ftypes[f], false, true, &out1, &out2, false);

// 					Eigen::Matrix4f last_tran = Eigen::Matrix4f::Identity();
// 					int k = 0;
// 					for (int j = 0; j < frame_count; j++)
// 					{
// 						graph[j]->tran = last_tran * graph[j]->relative_tran;
// 
// 						Eigen::Vector3f t = TranslationFromMatrix4f(graph[j]->tran);
// 						Eigen::Quaternionf q = QuaternionFromMatrix4f(graph[j]->tran);
// 
// 						out2 << fixed << setprecision(6) << timestamps[j]
// 							<< ' ' << t(0) << ' ' << t(1) << ' ' << t(2)
// 							<< ' ' << q.x() << ' ' << q.y() << ' ' << q.z() << ' ' << q.w() << endl;
// 
// 						if (k < keyframe_indices.size() && keyframe_indices[k] == j)
// 						{
// 							last_tran = graph[j]->tran;
// 							k++;
// 						}
// 					}
// 					out2.close();
				}
			}
			cout << endl;
		}
	}
}

void repeat_global_results()
{
	const int dcount = 9;
	std::string directories[dcount], names[dcount], prs[dcount];
	directories[0] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_xyz/";
	directories[1] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_desk/";
	directories[2] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_desk2/";
	directories[3] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_360/";
	directories[4] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_room/";
	directories[5] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_floor/";
	directories[6] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_rpy";
	directories[7] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_teddy/";
	directories[8] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_plant/";

// 	directories[0] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_rpy";
// 	directories[1] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_teddy/";
// 	directories[2] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_plant/";

	names[0] = "xyz";
	names[1] = "desk";
	names[2] = "desk2";
	names[3] = "360";
	names[4] = "room";
	names[5] = "floor";
	names[6] = "rpy";
	names[7] = "teddy";
	names[8] = "plant";

	prs[0] = "E:/bestresults/xyz/xyz_repeat_ORB_15.txt";
	prs[1] = "E:/bestresults/desk/desk_repeat_ORB_6.txt";
	prs[2] = "E:/bestresults/desk2/desk2_repeat_ORB_18.txt";
	prs[3] = "E:/bestresults/360/360_repeat_ORB_2.txt";
	prs[4] = "E:/bestresults/room/room_repeat_ORB_19.txt";
	prs[5] = "E:/bestresults/floor/floor_repeat_ORB_19.txt";
	prs[6] = "E:/bestresults/rpy/rpy_repeat_ORB_12.txt";
	prs[7] = "E:/bestresults/teddy/teddy_repeat_ORB_12.txt";
	prs[8] = "E:/bestresults/plant/plant_repeat_ORB_0.txt";

// 	names[0] = "rpy";
// 	names[1] = "teddy";
// 	names[2] = "plant";

	const int gfcount = 2;
	std::string gftypes[gfcount];
	gftypes[0] = "SIFT";
	gftypes[1] = "SURF";

	const int sdcount = 1;
	float dists[dcount][gfcount][sdcount] = {0};
	dists[0][0][0] = 0.1; dists[0][1][0] = 0.03;
	dists[1][0][0] = 0.03; dists[1][1][0] = 0.03;
	dists[2][0][0] = 0.02; dists[2][1][0] = 0.1;
	dists[3][0][0] = 0.3; dists[3][1][0] = 0.25;
	dists[4][0][0] = 0.1; dists[4][1][0] = 0.03;
	dists[5][0][0] = 0.25; dists[5][1][0] = 0.3;
	dists[6][0][0] = 0.1; dists[6][1][0] = 0.3;
	dists[7][0][0] = 0.25; dists[7][1][0] = 0.25;
	dists[8][0][0] = 0.3; dists[8][1][0] = 0.2;

	int st = -1, ed = -1;
	stringstream ss;

	int repeat_time = 4;

	for (int dd = 6; dd < dcount; dd++)
	{
		readData(directories[dd], st, ed, false);

		for (int f = gfcount - 1; f >= 0; f--)
		{
			cout << "\t" << gftypes[f];

			for (int sd = 0; sd < sdcount; sd++)
			{
				cout << "\tdists: " << dists[dd][f][sd] << endl;
				Config::instance()->set<float>("max_dist_for_inliers", dists[dd][f][sd]);

				for (int i = 0; i < repeat_time; i++)
				{
					readPairwiseResult(prs[dd], "ORB");

					ss.clear();
					ss.str("");
					ss << "E:/tempresults/" << names[dd] << "/global/"
						<< names[dd] << "_" << gftypes[f] << "_" << i << ".txt";
					ofstream out1(ss.str());
					GlobalRegistration(gftypes[f], nullptr, &out1);
				}
			}
		}
	}	
}

int main()
{
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
	repeat_pairwise_results();
//	repeat_global_results();
 	return 0;

	const int dcount = 6;
	std::string directories[dcount], names[dcount];
	directories[0] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_xyz/";
	directories[1] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_desk/";
	directories[2] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_desk2/";
	directories[3] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_floor/";
	directories[4] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_rpy/";
	directories[5] = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_360/";
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
// 	string test_type = "ORB";
// 	int repeat_time = 10;
// 
// 	readData(directories[dd], st, ed);
// 	Config::instance()->set<float>("max_dist_for_inliers", dist);
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
// 	string test_type = "SIFT";
// 	int repeat_time = 20;
// 	stringstream ss;
// 	string test_name = "E:/desk_ftof_ORB_0.2.txt";
// 
// 	readData(directories[dd], st, ed);
// 	Config::instance()->set<float>("max_dist_for_inliers", dist);
// 	// 	Config::instance()->set<int>("ransac_max_iteration", 2000);
// 
// 	for (int i = 0; i < repeat_time; i++)
// 	{
// 		readPairwiseResult(test_name, "ORB");
// 		ss.clear();
// 		ss.str("");
// 		ss << "E:/123/desk_SIFT_" << i << ".txt";
// 		ofstream out1(ss.str());
// 		GlobalRegistration("SIFT", nullptr, &out1);
// 	}
// 	return 0;
	// test2 end

	cout << "0: " << directories[0] << endl;
	cout << "1: " << directories[1] << endl;
	cout << "2: " << directories[2] << endl;
	cout << "3: " << directories[3] << endl;
	cout << "4: " << directories[4] << endl;
	cout << "5: " << directories[5] << endl;

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
	cout << "5: read pairwise result & global registration gt & show result" << endl;
	cin >> method;

	if (method == 0 || method == 2)
	{
		string ftype;
		cout << "feature type(SURF, SIFT, ORB): ";
		cin >> ftype;

		cout << "max dist for inliers: ";
		cin >> dist;
		Config::instance()->set<float>("max_dist_for_inliers", dist);

		string fname;
		cout << "pairwise filename: ";
		cin >> fname;
		ofstream out(fname);

		string tname;
		cout << "trajectory filename: ";
		cin >> tname;
		ofstream out2(tname);
		PairwiseRegistration(ftype, false, true, &out, &out2);
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
		Config::instance()->set<float>("max_dist_for_inliers", dist);

		string gname;
		cout << "global filename: ";
		cin >> gname;
		ofstream global_result(gname);

// 		string gtname;
// 		cout << "gtloop filenmae: ";
// 		cin >> gtname;
// 		ifstream gtloop_in(gtname);
		GlobalRegistration(gftype, nullptr, &global_result);
	}

	if (method == 4)
	{
		string gftype;
		cout << "graph feature type(SURF, SIFT, ORB): ";
		cin >> gftype;

		string gtname;
		cout << "ground-truth loop closure filename: ";
		cin >> gtname;
		ofstream gt_loop_result(gtname);
		FindGroundTruthLoop(directories[dd], gftype, &gt_loop_result);
		LoopAnalysis();
	}

	if (method == 5)
	{
		string gname;
		cout << "global gt filename: ";
		cin >> gname;
		ofstream global_result(gname);
		FindGroundTruthLoop(directories[dd]);
		GlobalWithAll(&global_result);
	}
}  

bool getPlanesByRANSACCuda(
	vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f>> &result_planes,
	vector<vector<pair<int, int>>> *matches,
	const cv::Mat &rgb, const cv::Mat &depth)
{
// 	pcl::cuda::PointCloudAOS<pcl::cuda::Device>::Ptr cloud;
// 	cloud.reset(new pcl::cuda::PointCloudAOS<pcl::cuda::Device>);
// 	cloud->width = depth.size().width;
// 	cloud->height = depth.size().height;
// 	cloud->points.resize(cloud->width * cloud->height);
// 
// 	float fx = Config::instance()->get<float>("camera_fx");  // focal length x
// 	float fy = Config::instance()->get<float>("camera_fy");  // focal length y
// 	float cx = Config::instance()->get<float>("camera_cx");  // optical center x
// 	float cy = Config::instance()->get<float>("camera_cy");  // optical center y
// 
// 	float factor = Config::instance()->get<float>("depth_factor");	// for the 16-bit PNG files
// 
// 	for (int j = 0; j < depth.size().height; j++)
// 	{
// 		for (int i = 0; i < depth.size().width; i++)
// 		{
// 			ushort temp = depth.at<ushort>(j, i);
// 			if (depth.at<ushort>(j, i) != 0)
// 			{
// 				pcl::cuda::PointXYZRGB pt;
// 				pt.z = ((double)depth.at<ushort>(j, i)) / factor;
// 				pt.x = (i - cx) * pt.z / fx;
// 				pt.y = (j - cy) * pt.z / fy;
// 				pt.rgb.b = rgb.at<cv::Vec3b>(j, i)[0];
// 				pt.rgb.g = rgb.at<cv::Vec3b>(j, i)[1];
// 				pt.rgb.r = rgb.at<cv::Vec3b>(j, i)[2];
// 				pt.rgb.alpha = 255;
// 				cloud->points.push_back(pt);
// 			}
// 		}
// 	}
// 
// 	boost::shared_ptr<pcl::cuda::Device<float4>::type> normals;
// 	normals = pcl::cuda::computeFastPointNormals<pcl::cuda::Device>(cloud);
// 
// 	pcl::cuda::SampleConsensusModel1PointPlane<pcl::cuda::Device>::Ptr sac_model(
// 		new pcl::cuda::SampleConsensusModel1PointPlane<pcl::cuda::Device>(cloud));
// 	sac_model->setNormals(normals);
// 
// 	pcl::cuda::MultiRandomSampleConsensus<pcl::cuda::Device> sac(sac_model);
// 	sac.setMinimumCoverage(0.90); // at least 95% points should be explained by planes
// 	sac.setMaximumBatches(5);
// 	sac.setIerationsPerBatch(1000);
// 	sac.setDistanceThreshold(25 / 100.0);
// 
// 	if (!sac.computeModel(0))
// 	{
// 		return false;
// 	}
// 
// 	std::vector<pcl::cuda::SampleConsensusModel1PointPlane<pcl::cuda::Device>::IndicesPtr> planes;
// 	pcl::cuda::Device<int>::type region_mask;
// 	pcl::cuda::markInliers<pcl::cuda::Device>(cloud, region_mask, planes);
// 	thrust::host_vector<int> regions_host;
// 	std::copy(regions_host.begin(), regions_host.end(), std::ostream_iterator<int>(std::cerr, " "));
// 	planes = sac.getAllInliers();
// 
// 	std::vector<int> planes_inlier_counts = sac.getAllInlierCounts();
// 	std::vector<float4> coeffs = sac.getAllModelCoefficients();
// 	std::vector<float3> centroids = sac.getAllModelCentroids();
// 	std::cerr << "Found " << planes_inlier_counts.size() << " planes" << std::endl;
// 
// 	if (planes_inlier_counts.size() == 0)
// 		return false;
// 
// 	if (matches)
// 	{
// 		matches->clear();
// 	}
// 
// 	for (unsigned int i = 0; i < planes.size(); i++)
// 	{
// 		Eigen::Vector4f result_plane(coeffs[i].x, coeffs[i].y, coeffs[i].z, coeffs[i].w);
// 		result_plane.normalize();
// 		result_planes.push_back(result_plane);
// 
// 		if (matches)
// 		{
// 			vector<pair<int, int>> match;
// 			thrust::device_vector<int> iptr = *planes[i];
// 			for (unsigned int j = 0; j < iptr.size(); j++)
// 			{
// 				match.push_back(pair<int, int>(iptr[j] % 640, iptr[j] / 640));
// 			}
// 			matches->push_back(match);
// 		}
// 	}
// 
	return true;
}