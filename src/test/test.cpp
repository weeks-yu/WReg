#include "srba.h"
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

#include <pcl/cuda/point_types.h>
#include <pcl/cuda/point_cloud.h>
#include <pcl/cuda/sample_consensus/multi_ransac.h>
#include <pcl/cuda/sample_consensus/sac_model_1point_plane.h>
#include <pcl/cuda/segmentation/connected_components.h>
#include <pcl/cuda/features/normal_3d.h>

#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include "Config.h"
#include "ICPOdometry.h"
#include "ICPSlowdometry.h"

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
vector<pair<int, int>> pairs;
vector<float> rmses;
vector<pair<int, int>> matches_and_inliers;
vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> trans;
vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> refined_trans;
int pairs_count, now;
vector<bool> is_keyframe_pose_set;
vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> keyframe_poses;
bool show_refined = false;
map<int, int> keyframe_id;
vector<int> keyframe_indices;
vector<PointCloudPtr> downsampled_combined_clouds;
vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> combined_trans;

int plane_ids[640][480];
bool bfs_visited[640][480];

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
	const int icount = 6;
	std::string rname[icount], dname[icount];
	rname[0] = "E:/lab/pcl/kinect data/living_room_1/rgb/00590.jpg";
	rname[1] = "E:/lab/pcl/kinect data/living_room_1/rgb/00591.jpg";
	rname[2] = "E:/lab/pcl/kinect data/living_room_1/rgb/00592.jpg";
	rname[3] = "E:/lab/pcl/kinect data/living_room_1/rgb/00593.jpg";
	rname[4] = "E:/lab/pcl/kinect data/living_room_1/rgb/00594.jpg";
	rname[5] = "E:/lab/pcl/kinect data/living_room_1/rgb/00595.jpg";

	dname[0] = "E:/lab/pcl/kinect data/living_room_1/depth/00590.png";
	dname[1] = "E:/lab/pcl/kinect data/living_room_1/depth/00591.png";
	dname[2] = "E:/lab/pcl/kinect data/living_room_1/depth/00592.png";
	dname[3] = "E:/lab/pcl/kinect data/living_room_1/depth/00593.png";
	dname[4] = "E:/lab/pcl/kinect data/living_room_1/depth/00594.png";
	dname[5] = "E:/lab/pcl/kinect data/living_room_1/depth/00595.png";

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
	double cx = Config::instance()->get<double>("camera_cx");
	double cy = Config::instance()->get<double>("camera_cy");
	double fx = Config::instance()->get<double>("camera_fx");
	double fy = Config::instance()->get<double>("camera_fy");
	double depthFactor = Config::instance()->get<double>("depth_factor");
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
	rname[0] = "G:/kinect data/living_room_1/rgb/00038.jpg";
	rname[1] = "G:/kinect data/living_room_1/rgb/00053.jpg";

	dname[0] = "G:/kinect data/living_room_1/depth/00038.png";
	dname[1] = "G:/kinect data/living_room_1/depth/00053.png";

	cv::Mat r[icount], d[icount];
	PointCloudPtr cloud[icount];

	r[0] = cv::imread(rname[0]);
	d[0] = cv::imread(dname[0], -1);
	cloud[0] = ConvertToPointCloudWithoutMissingData(d[0], r[0], 0, 0);

	Frame *f[icount];
	f[0] = new Frame(r[0], d[0], "SURF", Eigen::Matrix4f::Identity());
	f[0]->f.buildFlannIndex();

	int N = 4, M = 4;
	int keyframeTest[16] = {0};

	for (int i = 1; i < icount; i++)
	{
		r[i] = cv::imread(rname[i]);
		d[i] = cv::imread(dname[i], -1);
		cloud[i] = ConvertToPointCloudWithoutMissingData(d[i], r[i], i, i);

		f[i] = new Frame(r[i], d[i], "SURF", Eigen::Matrix4f::Identity());
		vector<cv::DMatch> matches;
		f[0]->f.findMatchedPairs(matches, f[i]->f, 128, 2);

		Eigen::Matrix4f tran = Eigen::Matrix4f::Identity();
		float rmse;
		vector<cv::DMatch> inliers;
		Feature::getTransformationByRANSAC(tran, rmse, &inliers,
			&f[0]->f, &f[i]->f, matches);
		cout << matches.size() << ", " << inliers.size() << endl;

		for (int j = 0; j < inliers.size(); j++)
		{
			cv::KeyPoint keypoint = f[i]->f.feature_pts[inliers[j].queryIdx];
			int tN = N * keypoint.pt.x / 640;
			int tM = M * keypoint.pt.y / 480;
			tN = tN < 0 ? 0 : (tN >= N ? N - 1 : tN);
			tM = tM < 0 ? 0 : (tM >= M ? M - 1 : tM);

			keyframeTest[tM * N + tN]++;
		}

		pcl::transformPointCloud(*cloud[i], *cloud[i], tran);
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
		if (now > 1 /*0*/)
		{
			viewer->removeAllPointClouds();
			now--;
			cout << now - 1 << " " << now << endl;
			PointCloudPtr cloud_all(new PointCloudT);
			PointCloudPtr tran_cloud(new PointCloudT);
			*cloud_all += *clouds[now - 1];
			pcl::transformPointCloud(*clouds[now], *tran_cloud, trans[now]);
			*cloud_all += *tran_cloud;

			pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud_all);
			viewer->addPointCloud<pcl::PointXYZRGB>(cloud_all, rgb, "cloud");
			viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");

// 			cout << "pair " << now << ": " << pairs[now].first << "\t" << pairs[now].second << "\t"
// 				<< rmses[now] << "\t" << matches_and_inliers[now].first << "\t" << matches_and_inliers[now].second << endl;
// 			show_refined = false;
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
		}
	}
	else if (event.getKeySym() == "x" && event.keyDown())
	{
		if (now < pairs_count - 1)
		{
			viewer->removeAllPointClouds();
			now++;
			cout << now - 1 << " " << now << endl;
			PointCloudPtr cloud_all(new PointCloudT);
			PointCloudPtr tran_cloud(new PointCloudT);
			*cloud_all += *clouds[now - 1];
			pcl::transformPointCloud(*clouds[now], *tran_cloud, trans[now]);
			*cloud_all += *tran_cloud;

			pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud_all);
			viewer->addPointCloud<pcl::PointXYZRGB>(cloud_all, rgb, "cloud");
			viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");

// 			cout << "pair " << now << ": " << pairs[now].first << "\t" << pairs[now].second << "\t"
// 				<< rmses[now] << "\t" << matches_and_inliers[now].first << "\t" << matches_and_inliers[now].second << endl;
// 			show_refined = false;
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

		Eigen::Matrix4f tran = combined_trans[id_end / 50];
		for (int i = int(id_end / 50) * 50; i <= id_end; i++)
		{
			tran = tran * trans[i];
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
	ifstream result_infile("E:/360-ransac.txt");
	result_infile >> pairs_count;
	for (int i = 0; i < pairs_count; i++)
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

		cloud_needed.insert(base_id);
		cloud_needed.insert(target_id);
		pairs.push_back(pair<int, int>(base_id, target_id));
		rmses.push_back(rmse);
		matches_and_inliers.push_back(pair<int, int>(match_count, inlier_count));
		trans.push_back(tran);
	}

	string directory = "E:/lab/pcl/kinect data/rgbd_dataset_freiburg1_360";

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

// 	// test srba
// 	SrbaGraphT rba_graph;
// 	// --------------------------------------------------------------------------------
// 	// Set parameters
// 	// --------------------------------------------------------------------------------
// 	rba_graph.setVerbosityLevel(1);   // 0: None; 1:Important only; 2:Verbose
// 
// 	rba_graph.parameters.srba.use_robust_kernel = false;
// 	//rba.parameters.srba.optimize_new_edges_alone  = false;  // skip optimizing new edges one by one? Relative graph-slam without landmarks should be robust enough, but just to make sure we can leave this to "true" (default)
// 
// 	// Information matrix for relative pose observations:
// 	{
// 		const double STD_NOISE_XYZ = 0.01;
// 		const double STD_NOISE_ANGLES = mrpt::utils::DEG2RAD(0.5);
// 		Eigen::Matrix<double, 6, 6> ObsL;
// 		ObsL.setZero();
// 		// X,Y,Z:
// 		for (int i = 0; i < 3; i++) ObsL(i, i) = 1 / mrpt::utils::square(STD_NOISE_XYZ);
// 		// Yaw,pitch,roll:
// 		for (int i = 0; i < 3; i++) ObsL(3 + i, 3 + i) = 1 / mrpt::utils::square(STD_NOISE_ANGLES);
// 
// 		// Set:
// 		rba_graph.parameters.obs_noise.lambda = ObsL;
// 	}
// 
// 	// =========== Topology parameters ===========
// 	rba_graph.parameters.srba.max_tree_depth = 3;
// 	rba_graph.parameters.srba.max_optimize_depth = 3;
// 	rba_graph.parameters.ecp.submap_size = 5;
// 	rba_graph.parameters.ecp.min_obs_to_loop_closure = 1;
// 	// ===========================================
// 
// 	// srba test;
// 	Eigen::Matrix4f test_tran = trans[0];
// 	Eigen::Vector3f translation = TranslationFromMatrix4f(test_tran);
// 	Eigen::Vector3f yawPitchRoll = YawPitchRollFromMatrix4f(test_tran);
// 	SrbaGraphT::pose_t test_pose;
// 	test_pose.x() = translation(0);
// 	test_pose.y() = translation(1);
// 	test_pose.z() = translation(2);
// 	test_pose.setYawPitchRoll(yawPitchRoll(0), yawPitchRoll(1), yawPitchRoll(2));
// 
// 	Eigen::Vector3f t2(test_pose.x(), test_pose.y(), test_pose.z());
// 	mrpt::math::CQuaternionDouble q;
// 	test_pose.getAsQuaternion(q);
// 	Eigen::Quaternionf quaternion(q.r(), q.x(), q.y(), q.z());
// 	Eigen::Matrix4f rt = transformationFromQuaternionsAndTranslation(quaternion, t2);
// 
// 	if (rt != test_tran)
// 	{
// 		int a = 1;
// 	}
// 
// 	for (int i = 0; i < keyframe_indices.size(); i++)
// 	{
// 		keyframe_poses.push_back(Eigen::Matrix4f::Identity());
// 		is_keyframe_pose_set.push_back(false);
// 
// 		SrbaGraphT::new_kf_observations_t list_obs;
// 		SrbaGraphT::new_kf_observation_t obs_field;
// 
// 		// fake landmark
// 		obs_field.is_fixed = true;
// 		obs_field.obs.feat_id = i; // Feature ID == keyframe ID
// 		obs_field.obs.obs_data.x = 0;   // Landmark values are actually ignored.
// 		obs_field.obs.obs_data.y = 0;
// 		obs_field.obs.obs_data.z = 0;
// 		obs_field.obs.obs_data.yaw = 0;
// 		obs_field.obs.obs_data.pitch = 0;
// 		obs_field.obs.obs_data.roll = 0;
// 		list_obs.push_back(obs_field);
// 
// 		for (int j = 0; j < pairs_count; j++)
// 		{
// 			if (pairs[j].second == keyframe_indices[i])
// 			{
// 				obs_field.is_fixed = false;   // "Landmarks" (relative poses) have unknown relative positions (i.e. treat them as unknowns to be estimated)
// 				obs_field.is_unknown_with_init_val = false; // Ignored, since all observed "fake landmarks" already have an initialized value.
// 
// 				obs_field.obs.feat_id = keyframe_id[pairs[j].first];
// 
// 				Eigen::Matrix4f tran_i = trans[j].inverse();
// 				Eigen::Vector3f translation = TranslationFromMatrix4f(tran_i);
// 				Eigen::Vector3f yawPitchRoll = YawPitchRollFromMatrix4f(tran_i);
// 				obs_field.obs.obs_data.x = translation(0);
// 				obs_field.obs.obs_data.y = translation(1);
// 				obs_field.obs.obs_data.z = translation(2);
// 				obs_field.obs.obs_data.yaw = yawPitchRoll(0);
// 				obs_field.obs.obs_data.pitch = yawPitchRoll(1);
// 				obs_field.obs.obs_data.roll = yawPitchRoll(2);
// 				list_obs.push_back(obs_field);
// 			}
// 		}
// 
// 		SrbaGraphT::TNewKeyFrameInfo new_kf_info;
// 		rba_graph.define_new_keyframe(
// 			list_obs,      // Input observations for the new KF
// 			new_kf_info,   // Output info
// 			true           // Also run local optimization?
// 			);
// 
// 		for (int i = 1; i < is_keyframe_pose_set.size(); i++)
// 		{
// 			is_keyframe_pose_set[i] = false;
// 		}
// 		is_keyframe_pose_set[0] = true;
// 		bfs_visitor_struct bfsvs;
// 		rba_graph.bfs_visitor(0, 1000000, false, bfsvs, bfsvs, bfsvs, bfsvs);
// 	}

// 	keyframe_poses[0] = Eigen::Matrix4f::Identity();
// 	for (int i = 1; i < keyframe_indices.size(); i++)
// 	{
// 		for (int j = 0; j < pairs_count; j++)
// 		{
// 			if (pairs[j].second == keyframe_indices[i] && keyframe_indices[i] > pairs[j].first)
// 			{
// 				keyframe_poses[i] = keyframe_poses[keyframe_id[pairs[j].first]] * trans[j];
// 				break;
// 			}
// 		}
// 	}

	now = 0;
	cout << "pair " << now << ": " << pairs[now].first << "\t" << pairs[now].second << "\t"
		<< rmses[now] << "\t" << matches_and_inliers[now].first << "\t" << matches_and_inliers[now].second << endl;

	ViewerPtr viewer(new pcl::visualization::PCLVisualizer("test"));
	viewer->registerKeyboardCallback(KeyboardEventOccurred, (void*)&viewer);
	viewer->setBackgroundColor(0, 0, 0);
	viewer->addCoordinateSystem(1.0);
	viewer->initCameraParameters();

// 	PointCloudPtr cloud_all(new PointCloudT);
// 	for (int i = 0; i < keyframe_indices.size(); i++)
// 	{
// 		PointCloudPtr tran_cloud(new PointCloudT);
// 		pcl::transformPointCloud(*clouds[keyframe_indices[i]], *tran_cloud, keyframe_poses[i]);
// 		*cloud_all += *tran_cloud;
// 	}
// 
// 	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud_all);
// 	stringstream ss;
// 	ss << 0;
// 	viewer->addPointCloud<pcl::PointXYZRGB>(cloud_all, rgb, ss.str());
// 	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, ss.str());

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
	string directory = "E:/lab/pcl/kinect data/living_room_1";
	int id_end = 0;
	cin >> id_end;

	ifstream cloud_infile(directory + "/read.txt");
	string line;
	int k = 0;
	while (getline(cloud_infile, line))
	{
		if (k <= id_end)
		{
			cout << k << endl;;
			int pos = line.find(' ');
			cv::Mat rgb = cv::imread(directory + "/" + line.substr(0, pos));
			cv::Mat depth = cv::imread(directory + "/" + line.substr(pos + 1, line.length() - pos - 1), -1);
			rgbs[k] = rgb;
			depths[k] = depth;
			PointCloudPtr cloud = ConvertToPointCloudWithoutMissingData(depth, rgb, k, k);
			clouds[k] = cloud;
		}
		else
		{
			break;
		}
		k++;
	}
	cloud_infile.close();

	ICPOdometry *icpcuda = nullptr;
	int threads = Config::instance()->get<int>("icpcuda_threads");
	int blocks = Config::instance()->get<int>("icpcuda_blocks");
	int width = Config::instance()->get<int>("image_width");
	int height = Config::instance()->get<int>("image_height");
	double cx = Config::instance()->get<double>("camera_cx");
	double cy = Config::instance()->get<double>("camera_cy");
	double fx = Config::instance()->get<double>("camera_fx");
	double fy = Config::instance()->get<double>("camera_fy");
	double depthFactor = Config::instance()->get<double>("depth_factor");
	if (icpcuda == nullptr)
		icpcuda = new ICPOdometry(width, height, cx, cy, fx, fy, depthFactor);

	trans.push_back(Eigen::Matrix4f::Identity());
	
	for (int i = 1; i < k; i++)
	{
		cout << i << endl;
		icpcuda->initICPModel((unsigned short *)depths[i - 1].data, 20.0f, Eigen::Matrix4f::Identity());
		icpcuda->initICP((unsigned short *)depths[i].data, 20.0f);

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

	PointCloudPtr cloud_temp(new PointCloudT);
	Eigen::Matrix4f tran = Eigen::Matrix4f::Identity();
	combined_trans.push_back(tran);
	for (int i = 0; i < k; i++)
	{
		tran = tran * trans[i];
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

void read_txt()
{
	ofstream outfile("read.txt");
	for (int i = 0; i < 2870; i++)
	{
		stringstream s1, s2;
		s1 << i;

		for (int j = 0; j < 5 - s1.str().length(); j++)
		{
			s2 << '0';
		}
		s2 << s1.str();
		outfile << "rgb/" << s2.str() << ".jpg depth/" << s2.str() << ".png" << endl;
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
	cv::drawKeypoints(r, f->f.feature_pts, result);
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
	double cx = Config::instance()->get<double>("camera_cx");
	double cy = Config::instance()->get<double>("camera_cy");
	double fx = Config::instance()->get<double>("camera_fx");
	double fy = Config::instance()->get<double>("camera_fy");
	double depthFactor = Config::instance()->get<double>("depth_factor");
	if (icpcuda == nullptr)
		icpcuda = new ICPOdometry(width, height, cx, cy, fx, fy, depthFactor);

	icpcuda->initICP((unsigned short *)d.data, 20.f);
	cv::Mat normals;
	icpcuda->getNMapCurr(normals);

	memset(bfs_visited, 0, sizeof(bool) * 480 * 640);

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
	rname[0] = "G:/kinect data/living_room_1/rgb/00590.jpg";
	rname[1] = "G:/kinect data/living_room_1/rgb/00591.jpg";
// 	rname[2] = "E:/lab/pcl/kinect data/living_room_1/rgb/00592.jpg";
// 	rname[3] = "E:/lab/pcl/kinect data/living_room_1/rgb/00593.jpg";
// 	rname[4] = "E:/lab/pcl/kinect data/living_room_1/rgb/00594.jpg";
// 	rname[5] = "E:/lab/pcl/kinect data/living_room_1/rgb/00595.jpg";

	dname[0] = "G:/kinect data/living_room_1/depth/00590.png";
	dname[1] = "G:/kinect data/living_room_1/depth/00591.png";
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
	double cx = Config::instance()->get<double>("camera_cx");
	double cy = Config::instance()->get<double>("camera_cy");
	double fx = Config::instance()->get<double>("camera_fx");
	double fy = Config::instance()->get<double>("camera_fy");
	double depthFactor = Config::instance()->get<double>("depth_factor");
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

int main()
{
	//keyframe_test();
	//something();
	//icp_test();
	//Ransac_Test();
	//Ransac_Result_Show();
	//Registration_Result_Show();
	//read_txt();
	//feature_test();
	//PlaneFittingTest();
	//continuousPlaneExtractingTest();
	cudaTest();
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
// 	double fx = Config::instance()->get<double>("camera_fx");  // focal length x
// 	double fy = Config::instance()->get<double>("camera_fy");  // focal length y
// 	double cx = Config::instance()->get<double>("camera_cx");  // optical center x
// 	double cy = Config::instance()->get<double>("camera_cy");  // optical center y
// 
// 	double factor = Config::instance()->get<double>("depth_factor");	// for the 16-bit PNG files
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