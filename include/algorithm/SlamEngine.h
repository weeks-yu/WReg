#pragma once

#include <pcl/registration/gicp.h>
#include <opencv2/core/core.hpp>
#include "GraphManager.h"
#include "PointCloud.h"

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

	void setUsingDownsampling(bool use) { using_downsampling = use; }
	bool isUsingDownsampling() { return using_downsampling; }
	void setDownsampleRate(double rate) { downsample_rate = rate; }
	double getDownsampleRate() { return downsample_rate; }

	void setUsingGicp(bool use) { using_gicp = use; }
	bool isUsingGicp() { return using_gicp; }
	void setGicpParameters(int max_iter, double max_corr_dist, double epsilon);
	void setGicpMaxIterations(int max_iter) { gicp.setMaximumIterations(max_iter); }
	int getGicpMaxiterations() { return gicp.getMaximumIterations(); }
	void setGicpMaxCorrDist(double max_corr_dist) { gicp.setMaxCorrespondenceDistance(max_corr_dist); }
	double getGicpMaxCorrDist() { return gicp.getMaxCorrespondenceDistance(); }
	void setGicpEpsilon(double epsilon) { gicp.setTransformationEpsilon(epsilon); }
	double getGicpEpsilon() { return gicp.getTransformationEpsilon(); }
	
	void setUsingGraphOptimizer(bool use) { using_graph_optimizer = use; }
	bool isUsingGraphOptimizer() { return using_graph_optimizer; }
	void setGraphFeatureType(FeatureType type) { feature_type = type; }
	FeatureType getGraphFeatureType() { return feature_type; }

	void RegisterNext(const cv::Mat &imgRGB, const cv::Mat &imgDepth, double timestamp);
	PointCloudPtr GetScene();

private:
	int frame_id;

	// results
	Eigen::Matrix4f last_transformation;
	vector<PointCloudPtr> point_clouds;

	// temporary variables
	PointCloudPtr last_cloud;
	GraphManager graph_manager;

	// parameters - downsampling
	bool using_downsampling;
	double downsample_rate;

	// parameters - gicp
	bool using_gicp;
	pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> gicp;

	// parameters - graph optimizer
	bool using_graph_optimizer;
	FeatureType feature_type;
};