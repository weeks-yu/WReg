#include "Feature.h"

#include <algorithm>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/nonfree/gpu.hpp"
#include <pcl/common/transformation_from_correspondences.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include "PointCloud.h"

unsigned long* Feature::new_id = nullptr;

bool sizeCompare(const pair<int, vector<cv::DMatch>> &a, const pair<int, vector<cv::DMatch>> &b)
{
	return a.second.size() > b.second.size();
}

void Feature::SIFTExtractor(vector<cv::KeyPoint> &feature_pts,
	vector_eigen_vector3f &feature_pts_3d,
	cv::Mat &feature_descriptors,
	const cv::Mat &imgRGB, const cv::Mat &imgDepth)
{
	cv::SIFT sift_detector;
	cv::Mat mask, descriptors;
	vector<cv::KeyPoint> fpts;

	sift_detector(imgRGB, mask, fpts, descriptors);

	for (int i = 0; i < fpts.size(); i++)
	{
		if (imgDepth.at<ushort>(cvRound(fpts[i].pt.y), cvRound(fpts[i].pt.x)) == 0)
			continue;
		
		feature_pts.push_back(fpts[i]);
		Eigen::Vector3f pt = ConvertPointTo3D(fpts[i].pt.x, fpts[i].pt.y, imgDepth);
		feature_pts_3d.push_back(pt);

		double norm = 0.0;
		for (int j = 0; j < descriptors.cols; j++)
		{
			norm += descriptors.at<float>(i, j) * descriptors.at<float>(i, j);
		}
		norm = sqrt(norm);
		for (int j = 0; j < descriptors.cols; j++)
		{
			descriptors.at<float>(i, j) /= norm;
		}
		feature_descriptors.push_back(descriptors.row(i));
	}
}

void Feature::SURFExtractor(vector<cv::KeyPoint> &feature_pts,
	vector_eigen_vector3f &feature_pts_3d,
	cv::Mat &feature_descriptors,
	const cv::Mat &imgRGB, const cv::Mat &imgDepth)
{
	cv::SURF surf_detector;
	cv::Mat mask, descriptors;
	vector<cv::KeyPoint> fpts;

	surf_detector(imgRGB, mask, fpts, descriptors);

	for (int i = 0; i < fpts.size(); i++)
	{
		if (imgDepth.at<ushort>(cvRound(fpts[i].pt.y), cvRound(fpts[i].pt.x)) == 0)
			continue;
		feature_pts.push_back(fpts[i]);
		Eigen::Vector3f pt = ConvertPointTo3D(fpts[i].pt.x, fpts[i].pt.y, imgDepth);
		feature_pts_3d.push_back(pt);
		feature_descriptors.push_back(descriptors.row(i));
	}
}

// void Feature::SURFExtractor_GPU(vector<cv::KeyPoint> &feature_pts,
// 	vector_eigen_vector3f &feature_pts_3d,
// 	cv::Mat &feature_descriptors,
// 	const cv::Mat &imgRGB, const cv::Mat &imgDepth)
// {
// 	cv::Mat grey;
// 	cv::cvtColor(imgRGB, grey, CV_BGR2GRAY);
// 	//cv::gpu::printShortCudaDeviceInfo(cv::gpu::getDevice());
// 	cv::gpu::GpuMat gpuRGB, gpuKeypoints, gpuDescriptors;
// 	gpuRGB.upload(grey);
// 	cv::gpu::SURF_GPU surf;
// 
// 	surf(gpuRGB, cv::gpu::GpuMat(), gpuKeypoints, gpuDescriptors);
// 
// 	vector<cv::KeyPoint> fpts;
// 	cv::Mat descriptors;
// 	surf.downloadKeypoints(gpuKeypoints, fpts);
// 	gpuDescriptors.download(descriptors);
// 
// 	for (int i = 0; i < fpts.size(); i++)
// 	{
// 		if (imgDepth.at<ushort>(cvRound(fpts[i].pt.y), cvRound(fpts[i].pt.x)) == 0)
// 			continue;
// 		feature_pts.push_back(fpts[i]);
// 		Eigen::Vector3f pt = ConvertPointTo3D(fpts[i].pt.x, fpts[i].pt.y, imgDepth);
// 		feature_pts_3d.push_back(pt);
// 		feature_descriptors.push_back(descriptors.row(i));
// 	}
// }

void Feature::ORBExtractor(vector<cv::KeyPoint> &feature_pts,
	vector_eigen_vector3f &feature_pts_3d,
	cv::Mat &feature_descriptors,
	const cv::Mat &imgRGB, const cv::Mat &imgDepth)
{
	cv::ORB orb_detector;
	cv::Mat mask, descriptors;
	vector<cv::KeyPoint> fpts;

	orb_detector(imgRGB, mask, fpts, descriptors);
	for (int i = 0; i < fpts.size(); i++)
	{
		if (imgDepth.at<ushort>(cvRound(fpts[i].pt.y), cvRound(fpts[i].pt.x)) == 0)
			continue;
		feature_pts.push_back(fpts[i]);
		Eigen::Vector3f pt = ConvertPointTo3D(fpts[i].pt.x, fpts[i].pt.y, imgDepth);
		feature_pts_3d.push_back(pt);
		feature_descriptors.push_back(descriptors.row(i));
	}
}

void Feature::setMultiple(int frame_index)
{
	if (this->multiple) return;
	this->multiple = true;
	this->feature_frame_index.clear();
	for (int i = 0; i < this->feature_pts.size(); i++)
	{
		this->feature_frame_index.push_back(frame_index);
	}
}

void Feature::extract(const cv::Mat &imgRGB, const cv::Mat &imgDepth, string type)
{
	this->type = type;
	if (type == "sift")
	{
		Feature::SIFTExtractor(feature_pts, feature_pts_3d, feature_descriptors, imgRGB, imgDepth);
	}
	else if (type == "surf")
	{
		Feature::SURFExtractor(feature_pts, feature_pts_3d, feature_descriptors, imgRGB, imgDepth);
		//Feature::SURFExtractor_GPU(feature_pts, feature_pts_3d, feature_descriptors, imgRGB, imgDepth);
	}
	else if (type == "orb")
	{
		Feature::ORBExtractor(feature_pts, feature_pts_3d, feature_descriptors, imgRGB, imgDepth);
	}
	flann_matcher = nullptr;
}

void Feature::buildFlannIndex()
{
	if (this->type != "sift" && this->type != "surf")
	{
		return;
	}
	releaseFlannIndex();
	if (feature_pts.size() > 0)
	{
		int trees = Config::instance()->get<int>("kdtree_trees");
		int max_leaf;
		if (multiple)
			max_leaf = Config::instance()->get<int>("kdtree_max_leaf_mult");
		else
			max_leaf = Config::instance()->get<int>("kdtree_max_leaf");
		this->flann_matcher = new cv::FlannBasedMatcher(new cv::flann::KDTreeIndexParams(trees),
			new cv::flann::SearchParams(max_leaf));
		vector<cv::Mat> ds;
		ds.push_back(feature_descriptors);
		this->flann_matcher->add(ds);
	}
	else
	{
		this->flann_matcher = nullptr;
	}
}

void Feature::releaseFlannIndex()
{
	if (flann_matcher)
	{
		delete flann_matcher;
		flann_matcher = nullptr;
	}
}

int Feature::findMatched(vector<cv::DMatch> &matches, const cv::Mat &descriptor)
{
	if (this->flann_matcher == nullptr)
	{
		return -1;
	}

	vector<vector<cv::DMatch>> matches_;
	// get the best two neighbours
	this->flann_matcher->knnMatch(descriptor, matches_, 2);
	float ratio = Config::instance()->get<float>("matches_criterion");

	for (int i = 0; i < matches_.size(); i++)
	{
		if (matches_[i][0].distance < ratio * matches_[i][1].distance)
		{
			matches.push_back(matches_[i][0]);
		}
	}

	return matches.size();
}

int Feature::findMatchedPairs(vector<cv::DMatch> &matches, const Feature *other)
{
	return findMatched(matches, other->feature_descriptors);
}

int Feature::findMatchedPairsBruteForce(vector<cv::DMatch> &matches, const Feature *other)
{
	cv::Mat od = other->feature_descriptors;
	cv::HammingLUT lut;
	float ratio = Config::instance()->get<float>("matches_criterion");

	for (int i = 0; i < other->feature_descriptors.rows; i++)
	{
		unsigned int min_dist = INT_MAX;
		unsigned int sec_dist = INT_MAX;
		int min_idx = -1, sec_idx = -1;
		uchar *query_feat = od.ptr(i);
		for (int j = 0; j < feature_descriptors.rows; j++)
		{
			unsigned char* train_feat = feature_descriptors.ptr(j);
			unsigned int dist = lut((query_feat), (train_feat), 32);

			if (dist < min_dist)
			{
				sec_dist = min_dist;
				sec_idx = min_idx;
				min_dist = dist;
				min_idx = j;
			}
			else if(dist < sec_dist)
			{
				sec_dist = dist;
				sec_idx = j;
			}
		}

		if (min_dist <= (unsigned int)(sec_dist * ratio))
		{
			matches.push_back(cv::DMatch(i, min_idx, 0, (float)min_dist));
		}
	}

	return matches.size();
}

bool Feature::findMatchedPairsMultiple(vector<int> &frames, vector<vector<cv::DMatch>> &matches,
	const Feature *other, int k, int min_matches)
{
	if (this->flann_matcher == nullptr || !this->multiple)
	{
		return false;
	}

	int frame_count = this->getFrameCount();

	if (frame_count * 4 < k)
	{
		k = frame_count * 4;
	}

	vector<vector<cv::DMatch>> matches_;

	// get the best two neighbours
	this->flann_matcher->knnMatch(other->feature_descriptors, matches_, k);
	float ratio = Config::instance()->get<float>("matches_criterion");
	std::map<int, int> frame_match_count;
	std::map<int, int> result_index;
	std::map<int, vector<cv::DMatch>> matches_map;

	cv::DMatch match;
	for (int i = 0; i < matches_.size(); i++)
	{
		frame_match_count.clear();
		result_index.clear();
		for (int j = 0; j < k; j++)
		{
			if (j >= matches_[i].size()) break;
			int u = this->feature_frame_index[matches_[i][j].trainIdx];
			if (frame_match_count.find(u) != frame_match_count.end() &&
				frame_match_count[u] == 1)
			{
				if (matches_[i][result_index[u]].distance < ratio * matches_[i][j].distance)
				{
					matches_map[u].push_back(matches_[i][result_index[u]]);
				}
				frame_match_count[u]++;
			}
			else if (frame_match_count.find(u) == frame_match_count.end())
			{
				result_index[u] = j;
				frame_match_count[u] = 1;
			}
		}
	}

	vector<pair<int, vector<cv::DMatch>>> candidate;
	for (std::map<int, vector<cv::DMatch>>::iterator it = matches_map.begin(); it !=  matches_map.end(); it++)
	{
		if ((*it).second.size() >= min_matches)
		{
			candidate.push_back(pair<int, vector<cv::DMatch>>(it->first, it->second));
		}
	}

	if (candidate.size() > Config::instance()->get<int>("candidate_number"))
	{
		sort(candidate.begin(), candidate.end(), sizeCompare);
	}

	for (int i = 0; i < (candidate.size() > 10 ? 10 : candidate.size()); i++)
	{
		frames.push_back(candidate[i].first);
		matches.push_back(candidate[i].second);
	}

	return true;
}

int Feature::getFrameCount()
{
	std::set<int> frame_index;

	for (int i = 0; i < this->feature_frame_index.size(); i++)
	{
		frame_index.insert(this->feature_frame_index[i]);
	}
	return frame_index.size();
}

template <class InputVector>
Eigen::Matrix4f Feature::getTransformFromMatches(bool &valid,
	const vector_eigen_vector3f &earlier,
	const vector_eigen_vector3f &now,
	const InputVector &matches,
	float InlierDist /* = -1.0 */)
{
	pcl::TransformationFromCorrespondences tfc;
	valid = true;
	vector<Eigen::Vector3f> t, f;

	for (InputVector::const_iterator it = matches.begin() ; it != matches.end(); it++)
	{
		int this_id = it->queryIdx;
		int earlier_id = it->trainIdx;

		Eigen::Vector3f from(now[this_id][0],
			now[this_id][1],
			now[this_id][2]);
		Eigen::Vector3f to(earlier[earlier_id][0],
			earlier[earlier_id][1],
			earlier[earlier_id][2]);
		if (InlierDist > 0)
		{  
			// storing is only necessary, if max_dist is given
			f.push_back(from);
			t.push_back(to);    
		}
		tfc.add(from, to, 1.0 /*/ to(2)*/);
	}

	// find smallest distance between a point and its neighbour in the same cloud
	if (InlierDist > 0)
	{  
		//float min_neighbour_dist = 1e6;
		Eigen::Matrix4f foo;

		valid = true;
		for (unsigned int i = 0; i < f.size(); i++)
		{
			float d_f = (f.at((i + 1) % f.size()) - f.at(i)).norm();
			float d_t = (t.at((i + 1) % t.size()) - t.at(i)).norm();

			if (abs(d_f - d_t) > InlierDist)
			{
				valid = false;
				return Eigen::Matrix4f();
			}
		}
	}

	// get relative movement from samples
	return tfc.getTransformation().matrix();
}

void Feature::computeInliersAndError(vector<cv::DMatch> &inliers, float &mean_error, vector<double> *errors, // output vars. if errors == nullptr, do not return error for each match
	const vector<cv::DMatch> &matches,
	const Eigen::Matrix4f &transformation,
	const vector_eigen_vector3f &earlier, const vector_eigen_vector3f &now,
	float squaredInlierDist)
{
	Eigen::Affine3f a(transformation);
	inliers.clear();
	if (errors != nullptr) errors->clear();

	mean_error = 0.0;
	for (unsigned int j = 0; j < matches.size(); j++)
	{
		//compute new error and inliers
		unsigned int this_id = matches[j].queryIdx;
		unsigned int earlier_id = matches[j].trainIdx;

		Eigen::Vector3f vec = (a * now[this_id]) - earlier[earlier_id];

		double error = vec.dot(vec);

		if (error > squaredInlierDist)
			continue; //ignore outliers

		inliers.push_back(matches[j]);

		mean_error += error;
		if (errors != nullptr) errors->push_back(error);
	}

	if (inliers.size() < 3)
	{
		//at least the samples should be inliers
		inliers.clear();
		mean_error = 1e9;
	}
	else
	{
		mean_error = sqrt(mean_error / inliers.size());
	}
}

void Feature::computeInliersAndError(vector<cv::DMatch> &inliers, float &rmse, vector<double> *errors,
	const vector<cv::DMatch> &matches,
	const Eigen::Matrix4f &transformation,
	const Feature* earlier, const Feature* now,
	float squaredInlierDist)
{
	Eigen::Affine3f a(transformation);
	inliers.clear();
	if (errors != nullptr) errors->clear();

	rmse = 0.0;
	for (unsigned int j = 0; j < matches.size(); j++)
	{
		//compute new error and inliers
		unsigned int this_id = matches[j].queryIdx;
		unsigned int earlier_id = matches[j].trainIdx;

		Eigen::Vector3f vec = (a * now->feature_pts_3d[this_id]) - earlier->feature_pts_3d[earlier_id];

		double error = vec.dot(vec);

		if (error > squaredInlierDist)
			continue; //ignore outliers

		inliers.push_back(matches[j]);

		rmse += error;
		if (errors != nullptr) errors->push_back(error);
	}

	if (inliers.size() < 3)
	{
		//at least the samples should be inliers
		inliers.clear();
		rmse = 1e9;
	}
	else
	{
		rmse = sqrt(rmse / inliers.size());
	}
}

bool Feature::getTransformationByRANSAC(Eigen::Matrix4f &result_transform,
	float &rmse, vector<cv::DMatch> *matches, // output vars. if matches == nullptr, do not return inlier match
	const Feature* earlier, const Feature* now,
	const vector<cv::DMatch> &initial_matches,
	int min_matches, float lnlier_percentage, float inlier_dist)
{
	if (matches != nullptr)	matches->clear();

	if (initial_matches.size() < min_matches)
	{
		return false;
	}

	unsigned int min_inlier_threshold = (unsigned int)(initial_matches.size() * lnlier_percentage);
	std::vector<cv::DMatch> inlier; //holds those feature correspondences that support the transformation
	float inlier_error; //all squared errors
	srand((long)std::clock());

	// a point is an inlier if it's no more than max_dist_m m from its partner apart
	const float squared_max_dist_m = inlier_dist * inlier_dist;

	// best values of all iterations (including invalids)
	float best_error = 1e6, best_error_invalid = 1e6;
	unsigned int best_inlier_invalid = 0, best_inlier_cnt = 0, valid_iterations = 0;
	Eigen::Matrix4f transformation;

	const unsigned int sample_size = 3;// chose this many randomly from the correspondences:
	unsigned int ransac_iterations = Config::instance()->get<int>("ransac_max_iteration");

	bool *used = new bool[initial_matches.size()];
	memset(used, 0, initial_matches.size() * sizeof(bool));
	vector<int> sample_matches_indices(3);
	vector<cv::DMatch> sample_matches_vector(3);

	for (unsigned int n_iter = 0; n_iter < ransac_iterations; n_iter++)
	{
		for (int i = 0; i < sample_matches_indices.size(); i++)
		{
			used[sample_matches_indices[i]] = false;
		}
		sample_matches_indices.clear();
		sample_matches_vector.clear();
		while (sample_matches_indices.size() < sample_size)
		{
			int id = rand() % initial_matches.size();
			if (!used[id])
			{
				used[id] = true;
				sample_matches_indices.push_back(id);
				sample_matches_vector.push_back(initial_matches.at(id));
			}
		}

		bool valid; // valid is false iff the sampled points clearly aren't inliers themself 
		transformation = Feature::getTransformFromMatches(valid,
			earlier->feature_pts_3d, now->feature_pts_3d,
			sample_matches_vector, inlier_dist);

		if (!valid) continue; // valid is false iff the sampled points aren't inliers themself 
		if (transformation != transformation) continue; //Contains NaN

		//test whether samples are inliers (more strict than before)
		Feature::computeInliersAndError(inlier, inlier_error, nullptr,
			sample_matches_vector, transformation,
			earlier->feature_pts_3d, now->feature_pts_3d,
			squared_max_dist_m);
		if (inlier_error > 1000) continue; //most possibly a false match in the samples

		Feature::computeInliersAndError(inlier, inlier_error, nullptr,
			initial_matches, transformation,
			earlier->feature_pts_3d, now->feature_pts_3d,
			squared_max_dist_m);

		if (inlier.size() < min_inlier_threshold || inlier_error > inlier_dist)
		{
			continue;
		}

		valid_iterations++;

		//Performance hacks:
		///Iterations with more than half of the initial_matches inlying, count twice
		if (inlier.size() > initial_matches.size() * 0.5) n_iter++;
		///Iterations with more than 80% of the initial_matches inlying, count threefold
		if (inlier.size() > initial_matches.size() * 0.8) n_iter++;

		if (inlier_error < best_error)
		{ //copy this to the result
			result_transform = transformation;
			if (matches != nullptr) *matches = inlier;
			best_inlier_cnt = inlier.size();
			rmse = inlier_error;
			best_error = inlier_error;
		}

		float new_inlier_error;

		transformation = Feature::getTransformFromMatches(valid,
			earlier->feature_pts_3d, now->feature_pts_3d, inlier); // compute new trafo from all inliers:
		if (transformation != transformation) continue; //Contains NaN
		Feature::computeInliersAndError(inlier, new_inlier_error, nullptr,
			initial_matches, transformation,
			earlier->feature_pts_3d, now->feature_pts_3d,
			squared_max_dist_m);

		if (inlier.size() < min_inlier_threshold || new_inlier_error > inlier_dist)
		{
			continue;
		}

		if (new_inlier_error < best_error) 
		{
			result_transform = transformation;
			if (matches != nullptr) *matches = inlier;
			best_inlier_cnt = inlier.size();
			rmse = new_inlier_error;
			best_error = new_inlier_error;
		}
	} //iterations
	
	return best_inlier_cnt >= min_inlier_threshold;
}
