#pragma once

//#include "SrbaManager.h"
#include <pcl/registration/gicp.h>
#include <opencv2/core/core.hpp>
#include "RobustManager.h"
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

	void setUsingIcpcuda(bool use);
	bool isUsingIcpcuda() { return using_icpcuda; }

	void setUsingFeature(bool use);
	bool isUsingFeature() { return using_feature; }

	void updateUsingOptimizer() { using_optimizer = using_robust_optimizer; }

	void setUsingRobustOptimizer(bool use)
	{
		using_robust_optimizer = use;
		updateUsingOptimizer();
	}
	bool isUsingRobustOptimizer() { return using_robust_optimizer; }

	void setGraphFeatureType(string type) { graph_feature_type = type; }
	string getGraphFeatureType() { return graph_feature_type; }

	void setFeatureType(string type) { feature_type = type; }
	string getFeatureType() { return feature_type; }

	void setFeatureMinMatches(int matches) { min_matches = matches; }
	void setFeatureInlierPercentage(float percent) { inlier_percentage = percent; }
	void setFeatureInlierDist(float dist) { inlier_dist = dist; }

	void setGraphMinMatches(int min_macthes) { Config::instance()->set<int>("graph_min_matches", min_macthes); }
	void setGraphInlierPercentage(float percent) { Config::instance()->set<float>("graph_min_inlier_p", percent); }
	void setGraphInlierDist(float dist) { Config::instance()->set<float>("graph_max_inlier_dist", dist); }

	void RegisterNext(const cv::Mat &imgRGB, const cv::Mat &imgDepth, double timestamp);
	void AddGraph(Frame *frame, PointCloudPtr cloud, bool keyframe, double timestamp);
	void AddGraph(Frame *frame, PointCloudPtr cloud, bool keyframe, bool quad, vector<int> &loop, double timestamp);
	void AddNext(const cv::Mat &imgRGB, const cv::Mat &imgDepth, double timestamp, Eigen::Matrix4f trajectory);
	PointCloudPtr GetScene();
	int GetFrameID() { return frame_id; }
	vector<pair<double, Eigen::Matrix4f>> GetTransformations();

	void SaveLogs(ofstream &outfile);
	void ShowStatistics();

	int getLCCandidate()
	{
		if (using_robust_optimizer)
			return robust_manager.keyframe_for_lc.size();
	}

#ifdef SAVE_TEST_INFOS
public:
	vector<int> keyframe_candidates_id;
	vector<pair<cv::Mat, cv::Mat>> keyframe_candidates;
	vector<int> keyframes_id;
	vector<pair<cv::Mat, cv::Mat>> keyframes;
	vector<string> keyframes_inliers_sig;
	vector<string> keyframes_exists_sig;
#endif

public:
	RobustManager robust_manager;

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
	float min_ftof_time;
	float max_ftof_time;
	float total_ftof_time;

	// temporary variables
	vector<PointCloudPtr> cloud_temp;
	PointCloudPtr last_cloud;
	cv::Mat last_depth;

	Frame *last_feature_keyframe;
	Frame *last_feature_frame;
	bool last_keyframe_detect_lc;
	bool last_feature_frame_is_keyframe;
	float last_rational;

	// parameters - downsampling
	double downsample_rate;

	// parameters - icpcuda
	bool using_icpcuda;
	ICPOdometry *icpcuda;
	int threads;
	int blocks;

	// parameters - feature
	bool using_feature;
	string feature_type;
	int min_matches;
	float inlier_percentage;
	float inlier_dist;

	// parameters - optimizer
	bool using_optimizer;

	// parameters - robust optimizer
	bool using_robust_optimizer;
	string graph_feature_type;
};