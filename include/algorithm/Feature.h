#pragma once

#include <vector>
#include <set>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "Config.h"

using namespace std;

class Feature
{
public:
	typedef vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f>> vector_eigen_vector4f;

public:

	vector<cv::KeyPoint> feature_pts;
	vector_eigen_vector4f feature_pts_3d;
	vector_eigen_vector4f feature_pts_3d_real;
	cv::Mat feature_descriptors;
	cv::Mat depth_image;
	vector<int> feature_frame_index;
	bool multiple;

private:

	cv::flann::Index* flann_index;
	int trees;

public:

	Feature(bool multiple = false)
	{
		this->flann_index = nullptr;
		this->trees = Config::instance()->get<int>("kdtree_trees");
		this->multiple = multiple;
	}

	Feature(const cv::Mat &imgRGB, const cv::Mat &imgDepth, string type = "SURF")
	{
		if (type == "SIFT")
		{
			Feature::SIFTExtrator(feature_pts, feature_pts_3d, feature_descriptors, imgRGB, imgDepth);
		}
		else if (type == "SURF")
		{
			Feature::SURFExtrator(feature_pts, feature_pts_3d, feature_descriptors, imgRGB, imgDepth);
		}
		this->flann_index = nullptr;
		this->trees = Config::instance()->get<int>("kdtree_trees");
		this->multiple = false;
	}

	void setMultiple(int frame_index = 0);

	void extract(const cv::Mat &imgRGB, const cv::Mat &imgDepth, string type = "SURF");

	void buildFlannIndex();

	int findMatchedPairs(vector<cv::DMatch> &matches, const Feature &other, int max_leafs = 64);

	bool findMatchedPairsMultiple(vector<int> &frames, vector<vector<cv::DMatch>> &matches, const Feature &other, int k = 30, int max_leafs = 128);

	int getFrameCount();

	void updateFeaturePoints3D(const Eigen::Matrix4f &tran);

public:

	static void SIFTExtrator(vector<cv::KeyPoint> &feature_pts,
		vector_eigen_vector4f &feature_pts_3d,
		cv::Mat &feature_descriptors,
		const cv::Mat &imgRGB, const cv::Mat &imgDepth);

	static void SURFExtrator(vector<cv::KeyPoint> &feature_pts,
		vector_eigen_vector4f &feature_pts_3d,
		cv::Mat &feature_descriptors,
		const cv::Mat &imgRGB, const cv::Mat &imgDepth);

	template <class InputVector>
	static Eigen::Matrix4f getTransformFromMatches(bool &valid,
		const Feature* earlier, const Feature* now,
		const InputVector &matches,
		float max_dist = -1.0);

	static void computeInliersAndError(vector<cv::DMatch> &inliers, double &mean_error, vector<double> *errors,
		const vector<cv::DMatch> &matches,
		const Eigen::Matrix4f &transformation,
		const vector_eigen_vector4f &earlier, const vector_eigen_vector4f &now,
		double squaredMaxInlierDistInM);

	static bool getTransformationByRANSAC(Eigen::Matrix4f &result_transform, float &rmse, vector<cv::DMatch> *matches, 
		const Feature* earlier, const Feature* now,
		const vector<cv::DMatch> &initial_matches,
		unsigned int ransac_iterations = 1000);
};