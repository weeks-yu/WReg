#pragma once

//#include "SrbaManager.h"
#include <pcl/registration/gicp.h>
#include <opencv2/core/core.hpp>
#include "RobustManager.h"
#include "PointCloud.h"

#include "SiftRegister.h"
#include "SurfRegister.h"
#include "OrbRegister.h"
#include "IcpcudaRegister.h"

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

	void setPairwiseRegister(string type);
	void setSecondPairwiseRegister(string type);
	void setGraphRegister(string type);

	void setPairwiseParametersFeature(int min_matches, float inlier_percentage, float inlier_dist);
	void setPairwiseParametersIcpcuda(float dist, float angle, int threads, int blocks);

	void setSecondPairwiseParametersFeature(int min_matches, float inlier_percentage, float inlier_dist);
	void setSecondPairwiseParametersIcpcuda(float dist, float angle, int threads, int blocks);

	void setGraphManager(string type);
	void setGraphParametersFeature(int min_matches, float inlier_percentage, float inlier_dist);

	void RegisterNext(const cv::Mat &imgRGB, const cv::Mat &imgDepth, double timestamp);
	void AddGraph(Frame *frame, PointCloudPtr cloud, bool keyframe, double timestamp);
	void AddGraph(Frame *frame, PointCloudPtr cloud, bool keyframe, bool quad, vector<int> &loop, double timestamp);
	void AddNext(const cv::Mat &imgRGB, const cv::Mat &imgDepth, double timestamp, Eigen::Matrix4f trajectory);
	PointCloudPtr GetScene();
	int GetFrameID() { return frame_id; }
	vector<pair<double, Eigen::Matrix4f>> GetTransformations();

	void SaveLogs(ofstream &outfile);
	void ShowStatistics();

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
	string pairwise_register_type;
	PairwiseRegister *pairwise_register;

	string pairwise_register_type_2;
	PairwiseRegister *pairwise_register_2;

	string graph_manager_type;
	string graph_register_type;
	int graph_min_matches;
	float graph_inlier_percentage;
	float graph_inlier_dist;

	GraphManager *graph_manager;

private:
	int frame_id;
	int frame_interval;
	int frame_start;
	int frame_stop;

	// results
	Eigen::Matrix4f last_tran;
	Eigen::Matrix4f last_keyframe_tran;
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
	cv::Mat last_depth;

	Frame *last_keyframe;
	Frame *last_frame;
	bool is_last_frame_candidate;
	bool is_last_frame_keyframe;
	bool is_last_keyframe_candidate;
	float last_rational;

	bool using_second_register;

	// parameters - downsampling
	double downsample_rate;

	// parameters - icpcuda
	ICPOdometry *icpcuda;

	// parameters - optimizer
	bool using_optimizer;

	// parameters - robust optimizer
	bool using_robust_optimizer;
};