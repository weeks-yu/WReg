#pragma once

#include <vector>
#include <set>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "Config.h"
#include "PointCloudCuda.h"

using namespace std;

class Feature
{
public:
	typedef vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> vector_eigen_vector3f;

public:
	static unsigned long *new_id;
	static unsigned long NewID()
	{
		if (new_id == nullptr)
		{
			new_id = new unsigned long;
			*new_id = 0;
		}
		*new_id = *new_id + 1;
		return *new_id;
	}

public:
	
	vector<unsigned long> feature_ids;
	vector<cv::KeyPoint> feature_pts;
	vector_eigen_vector3f feature_pts_3d;
	vector_eigen_vector3f feature_pts_3d_real;
	cv::Mat feature_descriptors;
	cv::Mat depth_image;
	vector<int> feature_frame_index;
	bool multiple;
	string type;

private:

	cv::FlannBasedMatcher *flann_matcher;
//	cv::flann::Index* flann_index;

public:

	Feature(bool multiple = false)
	{
		this->flann_matcher = nullptr;
		this->multiple = multiple;
	}

	Feature(const cv::Mat &imgRGB, const cv::Mat &imgDepth, string type = "SURF")
	{
		this->extract(imgRGB, imgDepth, type);
		this->multiple = false;
	}

	~Feature()
	{
		releaseFlannIndex();
	}

	int size() { return feature_pts.size(); }

	void setMultiple(int frame_index = 0);

	void extract(const cv::Mat &imgRGB, const cv::Mat &imgDepth, string type = "SURF");

	void buildFlannIndex();

	void releaseFlannIndex();

	int findMatched(vector<cv::DMatch> &matches, const cv::Mat &descriptor);

	int findMatchedPairs(vector<cv::DMatch> &matches, const Feature *other);

	int findMatchedPairsBruteForce(vector<cv::DMatch> &matches, const Feature *other);

	bool findMatchedPairsMultiple(vector<int> &frames, vector<vector<cv::DMatch>> &matches, const Feature *other, int k = 30);

	int getFrameCount();

	void updateFeaturePoints3DReal(const Eigen::Matrix4f &tran);

public:

	static void SIFTExtractor(vector<cv::KeyPoint> &feature_pts,
		vector_eigen_vector3f &feature_pts_3d,
		cv::Mat &feature_descriptors,
		const cv::Mat &imgRGB, const cv::Mat &imgDepth);

	static void SURFExtractor(vector<cv::KeyPoint> &feature_pts,
		vector_eigen_vector3f &feature_pts_3d,
		cv::Mat &feature_descriptors,
		const cv::Mat &imgRGB, const cv::Mat &imgDepth);

	static void ORBExtractor(vector<cv::KeyPoint> &feature_pts,
		vector_eigen_vector3f &feature_pts_3d,
		cv::Mat &feature_descriptors,
		const cv::Mat &imgRGB, const cv::Mat &imgDepth);

	template <class InputVector>
	static Eigen::Matrix4f getTransformFromMatches(bool &valid,
		const vector_eigen_vector3f &earlier, const vector_eigen_vector3f &now,
		const InputVector &matches,
		float max_dist = -1.0);

	static void computeInliersAndError(vector<cv::DMatch> &inliers, float &mean_error, vector<double> *errors,
		const vector<cv::DMatch> &matches,
		const Eigen::Matrix4f &transformation,
		const vector_eigen_vector3f &earlier, const vector_eigen_vector3f &now,
		float squaredMaxInlierDistInM);

	static void computeInliersAndError(vector<cv::DMatch> &inliers, float &rmse, vector<double> *errors,
		const vector<cv::DMatch> &matches,
		const Eigen::Matrix4f &transformation,
		const Feature* earlier, const Feature* now);

	static bool getTransformationByRANSAC(Eigen::Matrix4f &result_transform,
		Eigen::Matrix<double, 6, 6> &result_information,
		int &point_count, int &point_corr_count,
		float &rmse, vector<cv::DMatch> *matches,
		const Feature* earlier, const Feature* now,
		PointCloudCuda *pcc,
		const vector<cv::DMatch> &initial_matches);

	static bool getTransformationByRANSAC_real(Eigen::Matrix4f &result_transform,
		Eigen::Matrix<double, 6, 6> &result_information,
		int &point_count, int &point_corr_count,
		float &rmse, vector<cv::DMatch> *matches,
		const Feature* earlier, const Feature* now,
		PointCloudCuda *pcc,
		const vector<cv::DMatch> &initial_matches);

	static bool getPlanesByRANSAC(Eigen::Vector4f &result_plane, vector<pair<int, int>> *matches,
		const cv::Mat &depth, const vector<pair<int, int>> &initial_point_indices);

	static float distToPlane(const Eigen::Vector3f &point, const Eigen::Vector4f &plane);
};