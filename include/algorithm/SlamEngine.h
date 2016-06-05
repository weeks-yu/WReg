#pragma once

#include "SrbaManager.h"
#include <pcl/registration/gicp.h>
#include <opencv2/core/core.hpp>
#include "RobustManager.h"
#include "HogmanManager.h"
#include "PointCloud.h"
#include "ICPOdometry.h"

class SlamEngine
{
public:
	SlamEngine();
	~SlamEngine();

	void setFrameInterval(int frameInterval) { frame_interval = frameInterval; }
	int getFrameInterval() { return frame_interval; }
	void setFrameStart(int frameStart) { frame_start = frameStart; }
	int getFrameStart() { return frame_start; }
	void setFrameStop(int frameStop) { frame_stop = frameStop; }
	int getFrameStop() { return frame_stop; }

	void setUsingGicp(bool use);
	bool isUsingGicp() { return using_gicp; }
	void setUsingDownsampling(bool use) { using_downsampling = use; }
	bool isUsingDownsampling() { return using_downsampling; }
	void setDownsampleRate(float rate) { downsample_rate = rate; }
	double getDownsampleRate() { return downsample_rate; }
	void setGicpParameters(int max_iter, double max_corr_dist, double epsilon);
	void setGicpMaxIterations(int max_iter) { gicp->setMaximumIterations(max_iter); }
	int getGicpMaxiterations() { return gicp->getMaximumIterations(); }
	void setGicpMaxCorrDist(float max_corr_dist) { gicp->setMaxCorrespondenceDistance(max_corr_dist); }
	double getGicpMaxCorrDist() { return gicp->getMaxCorrespondenceDistance(); }
	void setGicpEpsilon(float epsilon) { gicp->setTransformationEpsilon(epsilon); }
	double getGicpEpsilon() { return gicp->getTransformationEpsilon(); }
	
	void setUsingIcpcuda(bool use);
	bool isUsingIcpcuda() { return using_icpcuda; }

	void setUsingFeature(bool use);
	bool isUsingFeature() { return using_feature; }

	void setUsingHogmanOptimizer(bool use)
	{
		using_hogman_optimizer = use;
		using_optimizer = using_hogman_optimizer || using_srba_optimizer || using_robust_optimizer;
	}
	bool isUsingHogmanOptimizer() { return using_hogman_optimizer; }

	void setUsingSrbaOptimzier(bool use)
	{
		using_srba_optimizer = use;
		using_optimizer = using_hogman_optimizer || using_srba_optimizer || using_robust_optimizer;
	}
	bool isUsingSrbaOptimizer() { return using_srba_optimizer; }

	void setUsingRobustOptimzier(bool use)
	{
		using_robust_optimizer = use;
		using_optimizer = using_hogman_optimizer || using_srba_optimizer || using_robust_optimizer;
	}
	bool isUsingRobustOptimizer() { return using_robust_optimizer; }

	void setGraphFeatureType(string type) { graph_feature_type = type; }
	string getGraphFeatureType() { return graph_feature_type; }

	void setFeatureType(string type) { feature_type = type; }
	string getFeatureType() { return feature_type; }

	void setFeatureMinMatches(int min_macthes) { Config::instance()->set<int>("min_matches", min_macthes); }
	void setFeatureInlierPercentage(float percent) { Config::instance()->set<float>("min_inliers_percent", percent); }

	void RegisterNext(const cv::Mat &imgRGB, const cv::Mat &imgDepth, double timestamp);
	void AddGraph(Frame *frame, PointCloudPtr cloud, bool keyframe, double timestamp);
	void AddGraph(Frame *frame, PointCloudPtr cloud, bool keyframe, bool quad, vector<int> &loop, double timestamp);
	void AddNext(const cv::Mat &imgRGB, const cv::Mat &imgDepth, double timestamp, Eigen::Matrix4f trajectory);
	void Refine();
	PointCloudPtr GetScene();
	int GetFrameID() { return frame_id; }
	vector<pair<double, Eigen::Matrix4f>> GetTransformations();
	vector<pair<int, int>> GetLoop();

	void SaveLogs(ofstream &outfile);
	void ShowStatistics();

private:
	bool IsTransformationBigEnough();

#ifdef SAVE_TEST_INFOS
public:
	vector<int> keyframe_candidates_id;
	vector<pair<cv::Mat, cv::Mat>> keyframe_candidates;
	vector<int> keyframes_id;
	vector<pair<cv::Mat, cv::Mat>> keyframes;
	vector<string> keyframes_inliers_sig;
	vector<string> keyframes_exists_sig;
#endif

private:
	int frame_id;
	int frame_interval;
	int frame_start;
	int frame_stop;

	// results
	Eigen::Matrix4f last_transformation;
	Eigen::Matrix4f last_keyframe_transformation;
	Eigen::Matrix4f accumulated_transformation;
	int accumulated_frame_count;
	vector<Eigen::Matrix4f> transformation_matrix;
	vector<PointCloudPtr> point_clouds;
	vector<double> timestamps;

	// statistics
	clock_t total_start;
	int min_pt_count;
	int max_pt_count;
	float min_icp_time;
	float max_icp_time;
	float total_icp_time;
	float min_fit;
	float max_fit;

	// temporary variables
	PointCloudPtr last_cloud;
	cv::Mat last_depth;

	Frame *last_feature_keyframe;
	Frame *last_feature_frame;
	bool last_keyframe_in_quadtree;
	bool last_feature_frame_is_keyframe;
	float last_rational;

	HogmanManager hogman_manager;
	SrbaManager srba_manager;
	RobustManager robust_manager;

	// parameters - downsampling
	

	// parameters - gicp
	bool using_gicp;
	pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> *gicp;
	bool using_downsampling;
	double downsample_rate;

	// parameters - icpcuda
	bool using_icpcuda;
	ICPOdometry *icpcuda;
	int threads;
	int blocks;

	// parameters - feature
	bool using_feature;
	string feature_type;

	// parameters - optimizer
	bool using_optimizer;

	// parameters - hogman optimizer
	bool using_hogman_optimizer;

	// parameters - srba optimizer
	bool using_srba_optimizer;

	// parameters - robust optimizer
	bool using_robust_optimizer;
	string graph_feature_type;
};