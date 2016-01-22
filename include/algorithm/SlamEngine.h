#pragma once

#include <pcl/registration/gicp.h>
#include <opencv2/core/core.hpp>
#include "GraphManager.h"
#include "PointCloud.h"
#include "ICPOdometry.h"

class SlamEngine
{
public:
	enum FeatureType
	{
		SIFT = 0,
		SURF
	};

public:
	SlamEngine();
	~SlamEngine();

	void setUsingGicp(bool use);
	bool isUsingGicp() { return using_gicp; }
	void setUsingDownsampling(bool use) { using_downsampling = use; }
	bool isUsingDownsampling() { return using_downsampling; }
	void setDownsampleRate(double rate) { downsample_rate = rate; }
	double getDownsampleRate() { return downsample_rate; }
	void setGicpParameters(int max_iter, double max_corr_dist, double epsilon);
	void setGicpMaxIterations(int max_iter) { gicp->setMaximumIterations(max_iter); }
	int getGicpMaxiterations() { return gicp->getMaximumIterations(); }
	void setGicpMaxCorrDist(double max_corr_dist) { gicp->setMaxCorrespondenceDistance(max_corr_dist); }
	double getGicpMaxCorrDist() { return gicp->getMaxCorrespondenceDistance(); }
	void setGicpEpsilon(double epsilon) { gicp->setTransformationEpsilon(epsilon); }
	double getGicpEpsilon() { return gicp->getTransformationEpsilon(); }
	
	void setUsingIcpcuda(bool use);
	bool isUsingIcpcuda() { return using_icpcuda; }

	void setUsingGraphOptimizer(bool use) { using_graph_optimizer = use; }
	bool isUsingGraphOptimizer() { return using_graph_optimizer; }
	void setGraphFeatureType(FeatureType type) { feature_type = type; }
	FeatureType getGraphFeatureType() { return feature_type; }

	void RegisterNext(const cv::Mat &imgRGB, const cv::Mat &imgDepth, double timestamp);
	PointCloudPtr GetScene();
	int GetFrameID() { return frame_id; }
	vector<pair<double, Eigen::Matrix4f>> GetTransformations();

	void ShowStatistics();
	void SaveTestInfo()
	{
		ofstream out("test.txt");
		for (int i = 0; i < graph_manager.baseid.size(); i++)
		{
			out << graph_manager.baseid[i] << endl;
			out << graph_manager.targetid[i] << endl;
			out << graph_manager.ransac_tran[i] << endl;
			out << graph_manager.icp_tran[i] << endl;
		}
		out.close();
	}

public:
	vector<pair<cv::Mat, cv::Mat>> keyframe_candidates;
	vector<pair<cv::Mat, cv::Mat>> keyframes;
	vector<string> keyframes_inliers_sig;
	vector<string> keyframes_exists_sig;

private:
	int frame_id;

	// results
	Eigen::Matrix4f last_transformation;
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
	GraphManager graph_manager;

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

	// parameters - graph optimizer
	bool using_graph_optimizer;
	FeatureType feature_type;
};