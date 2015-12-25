#include "Feature.h"

#include <algorithm>
#include <opencv2/nonfree/features2d.hpp>
// #include <opencv2/nonfree/nonfree.hpp>
#include <pcl/common/transformation_from_correspondences.h>
#include "PointCloud.h"

bool sizeCompare(const pair<int, vector<cv::DMatch>> &a, const pair<int, vector<cv::DMatch>> &b)
{
	return a.second.size() > b.second.size();
}

void Feature::SIFTExtrator(vector<cv::KeyPoint> &feature_pts, vector_eigen_vector4f &feature_pts_3d, cv::Mat &feature_descriptors,
	const cv::Mat &imgRGB, const cv::Mat &imgDepth, const Eigen::Matrix4f tran)
{
	cv::SIFT sift_detector;
	cv::Mat mask, descriptors;
	vector<cv::KeyPoint> fpts;

	sift_detector(imgRGB, mask, fpts, descriptors);

	for (int i = 0; i < fpts.size(); i++)
	{
		Eigen::Vector3f pt;
		pt  = ConvertPointTo3D(cvRound(fpts[i].pt.x), cvRound(fpts[i].pt.y), imgDepth);
		if (pt.isZero())
		{
			continue;
		}
		feature_pts.push_back(fpts[i]);
		feature_pts_3d.push_back(tran * Eigen::Vector4f(pt(0), pt(1) ,pt(2), 1.0));

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

void Feature::SURFExtrator(vector<cv::KeyPoint> &feature_pts, vector_eigen_vector4f &feature_pts_3d, cv::Mat &feature_descriptors,
	const cv::Mat &imgRGB, const cv::Mat &imgDepth, const Eigen::Matrix4f tran)
{
	cv::SURF surf_detector;
	cv::Mat mask, descriptors;
	vector<cv::KeyPoint> fpts;

	surf_detector(imgRGB, mask, fpts, descriptors);

	for (int i = 0; i < fpts.size(); i++)
	{
		Eigen::Vector3f pt;
		pt  = ConvertPointTo3D(cvRound(fpts[i].pt.x), cvRound(fpts[i].pt.y), imgDepth);
		if (pt.isZero())
		{
			continue;
		}
		feature_pts.push_back(fpts[i]);
		feature_pts_3d.push_back(tran * Eigen::Vector4f(pt(0), pt(1), pt(2), 1.0));
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

void Feature::extract(const cv::Mat &imgRGB, const cv::Mat &imgDepth, Eigen::Matrix4f tran, string type)
{
	if (type == "SIFT")
	{
		Feature::SIFTExtrator(feature_pts, feature_pts_3d, feature_descriptors, imgRGB, imgDepth, tran);
	}
	else if (type == "SURF")
	{
		Feature::SURFExtrator(feature_pts, feature_pts_3d, feature_descriptors, imgRGB, imgDepth, tran);
	}
	flann_index = nullptr;
	this->trees = Config::instance()->get<int>("kdtree_trees");
}

void Feature::buildFlannIndex()
{
	this->flann_index = new cv::flann::Index(feature_descriptors, cv::flann::KDTreeIndexParams(this->trees));
}

int Feature::findMatchedPairs(vector<cv::DMatch> &matches, const Feature &other, int max_leafs)
{
	if (this->flann_index == nullptr || this->multiple)
	{
		return -1;
	}

	const int k = 2;

	cv::Mat indices(other.feature_descriptors.rows, k, CV_32S);
	cv::Mat dists(other.feature_descriptors.rows, k, CV_32F);

	// get the best two neighbours
	this->flann_index->knnSearch(other.feature_descriptors, indices, dists, k, cv::flann::SearchParams(max_leafs));

	int* indices_ptr = indices.ptr<int> (0);
	float* dists_ptr = dists.ptr<float> (0);

	float ratio = Config::instance()->get<float>("matches_criterion");

	cv::DMatch match;
	for (int i = 0; i < indices.rows; i++)
	{
		if (dists_ptr[2 * i] < ratio * dists_ptr[2 * i + 1])
		{
			match.queryIdx = i;
			match.trainIdx = indices_ptr[2 * i];
			match.distance = dists_ptr[2 * i];
			matches.push_back(match);
		}
	}
	
	return matches.size();
}

bool Feature::findMatchedPairsMultiple(vector<int> &frames, vector<vector<cv::DMatch>> &matches, const Feature &other, int k, int max_leafs)
{
	if (this->flann_index == nullptr || !this->multiple)
	{
		return false;
	}

	int frame_count = this->getFrameCount();

	if (frame_count * 3 < k)
	{
		k = frame_count * 3;
	}

	cv::Mat indices(other.feature_descriptors.rows, k, CV_32S);
	cv::Mat dists(other.feature_descriptors.rows, k, CV_32F);

	// get the best two neighbours
	this->flann_index->knnSearch(other.feature_descriptors, indices, dists, k, cv::flann::SearchParams(max_leafs));

	int* indices_ptr = indices.ptr<int> (0);
	float* dists_ptr = dists.ptr<float> (0);

	float ratio = Config::instance()->get<float>("matches_criterion");
	int* frame_match_count = new int[this->getFrameCount()];
	std::map<int, int> result_index;
	std::map<int, vector<cv::DMatch>> matches_map;

	cv::DMatch match;
	for (int i = 0; i < indices.rows; i++)
	{
		memset(frame_match_count, 0, frame_count * sizeof(int));
		for (int j = 0; j < k; j++)
		{
			int u = this->feature_frame_index[indices_ptr[k * i + j]];
			if (frame_match_count[u] == 1)
			{
				if (dists_ptr[result_index[u]] < ratio * dists_ptr[k * i + j])
				{
					match.queryIdx = i;
					match.trainIdx = indices_ptr[result_index[u]];
					match.distance = dists_ptr[result_index[u]];
					matches_map[u].push_back(match);
				}
				frame_match_count[u]++;
			}
			else if (frame_match_count[u] == 0)
			{
				result_index[u] = k * i + j;
				frame_match_count[u]++;
			}
		}
	}
	delete [] frame_match_count;

	vector<pair<int, vector<cv::DMatch>>> candidate;
	unsigned int min_count = (unsigned int) Config::instance()->get<int>("min_matches");
	for (std::map<int, vector<cv::DMatch>>::iterator it = matches_map.begin(); it !=  matches_map.end(); it++)
	{
		if ((*it).second.size() >= min_count)
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

void Feature::transform(const Eigen::Matrix4f tran, int frame_index)
{
	for (int i = 0; i < this->feature_pts_3d.size(); i++)
	{
		if (!this->multiple || frame_index < 0 || this->feature_frame_index[i] == frame_index)
		{
			this->feature_pts_3d[i] = tran * this->feature_pts_3d[i];
		}
	}
}

void Feature::append(const Feature &other, int frame_index)
{
	for (int i = 0; i < other.feature_pts.size(); i++)
	{
		this->feature_pts.push_back(other.feature_pts[i]);
		this->feature_pts_3d.push_back(other.feature_pts_3d[i]);
		this->feature_descriptors.push_back(other.feature_descriptors.row(i));

		if (multiple)
		{
			this->feature_frame_index.push_back(frame_index);
		}
	}
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
	const Feature* earlier, const Feature* now,
	const InputVector &matches,
	float max_dist)
{
	pcl::TransformationFromCorrespondences tfc;
	valid = true;
	vector<Eigen::Vector3f> t, f;

	for (InputVector::const_iterator it = matches.begin() ; it != matches.end(); it++)
	{
		int this_id = it->queryIdx;
		int earlier_id = it->trainIdx;

		Eigen::Vector3f from(now->feature_pts_3d[this_id][0],
			now->feature_pts_3d[this_id][1],
			now->feature_pts_3d[this_id][2]);
		Eigen::Vector3f to(earlier->feature_pts_3d[earlier_id][0],
			earlier->feature_pts_3d[earlier_id][1],
			earlier->feature_pts_3d[earlier_id][2]);
		if (max_dist > 0)
		{  
			// storing is only necessary, if max_dist is given
			f.push_back(from);
			t.push_back(to);    
		}
		tfc.add(from, to, 1.0 / to(0)); //the further, the less weight b/c of accuracy decay
	}

	// find smallest distance between a point and its neighbour in the same cloud
	if (max_dist > 0)
	{  
		//float min_neighbour_dist = 1e6;
		Eigen::Matrix4f foo;

		valid = true;
		for (unsigned int i = 0; i < f.size(); i++)
		{
			float d_f = (f.at((i + 1) % f.size()) - f.at(i)).norm();
			float d_t = (t.at((i + 1) % t.size()) - t.at(i)).norm();

			if (abs(d_f - d_t) > max_dist)
			{
				valid = false;
				return Eigen::Matrix4f();
			}
		}
	}

	// get relative movement from samples
	return tfc.getTransformation().matrix();
}

void Feature::computeInliersAndError(vector<cv::DMatch> &inliers, double &mean_error, vector<double> *errors, // output vars. if errors == nullptr, do not return error for each match
	const vector<cv::DMatch> &matches,
	const Eigen::Matrix4f &transformation,
	const vector_eigen_vector4f &earlier, const vector_eigen_vector4f &now,
	double squaredMaxInlierDistInM)
{
	inliers.clear();
	if (errors != nullptr) errors->clear();

	vector<pair<float,int> > dists;
	std::vector<cv::DMatch> inliers_temp;

//	assert(matches.size() > 0);
	mean_error = 0.0;
	for (unsigned int j = 0; j < matches.size(); j++)
	{
		//compute new error and inliers
		unsigned int this_id = matches[j].queryIdx;
		unsigned int earlier_id = matches[j].trainIdx;

		Eigen::Vector4f vec = (transformation * now[this_id]) - earlier[earlier_id];

		double error = vec.dot(vec);

		if (error > squaredMaxInlierDistInM)
			continue; //ignore outliers

// 		if (!(error >= 0.0)){
// 
// 		}
		error = sqrt(error);
		dists.push_back(pair<float, int>(error, j));
		inliers_temp.push_back(matches[j]); //include inlier

		mean_error += error;
		if (errors != nullptr) errors->push_back(error);
	}

	if (inliers_temp.size() < 3)
	{
		//at least the samples should be inliers
		mean_error = 1e9;
	}
	else
	{
		mean_error /= inliers_temp.size();

		// sort inlier ascending according to their error
		sort(dists.begin(), dists.end());

		inliers.resize(inliers_temp.size());
		for (unsigned int i = 0; i < inliers_temp.size(); i++)
		{
			inliers[i] = matches[dists[i].second];
		}
	}
}

bool Feature::getTransformationByRANSAC(Eigen::Matrix4f &result_transform, float &rmse, vector<cv::DMatch> *matches, // output vars. if matches == nullptr, do not return inlier match
	const Feature* earlier, const Feature* now,
	const vector<cv::DMatch> &initial_matches,
	unsigned int ransac_iterations)
{
	if (matches != nullptr)	matches->clear();

	if (initial_matches.size() <= (unsigned int) Config::instance()->get<int>("min_matches"))
	{
		return false;
	}

	unsigned int min_inlier_threshold = int(initial_matches.size() * 0.2);
	std::vector<cv::DMatch> inlier; //holds those feature correspondences that support the transformation
	double inlier_error; //all squared errors
	srand((long)std::clock());

	// a point is an inlier if it's no more than max_dist_m m from its partner apart
	const float max_dist_m = Config::instance()->get<double>("max_dist_for_inliers");
	const float squared_max_dist_m = max_dist_m * max_dist_m;

	// best values of all iterations (including invalids)
	double best_error = 1e6, best_error_invalid = 1e6;
	unsigned int best_inlier_invalid = 0, best_inlier_cnt = 0, valid_iterations = 0;
	Eigen::Matrix4f transformation;

	const unsigned int sample_size = 3;// chose this many randomly from the correspondences:
	for (unsigned int n_iter = 0; n_iter < ransac_iterations; n_iter++)
	{
		//generate a map of samples. Using a map solves the problem of drawing a sample more than once
		std::set<cv::DMatch> sample_matches;
		std::vector<cv::DMatch> sample_matches_vector;
		while(sample_matches.size() < sample_size){
			int id = rand() % initial_matches.size();
			sample_matches.insert(initial_matches.at(id));
			sample_matches_vector.push_back(initial_matches.at(id));
		}

		bool valid; // valid is false iff the sampled points clearly aren't inliers themself 
		transformation = Feature::getTransformFromMatches(valid, earlier, now, sample_matches, max_dist_m);

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

		// check also invalid iterations
		if (inlier.size() > best_inlier_invalid)
		{
			best_inlier_invalid = inlier.size();
			best_error_invalid = inlier_error;
		}

		if (inlier.size() < min_inlier_threshold || inlier_error > max_dist_m)
		{
			continue;
		}

		valid_iterations++;
//		assert(inlier_error>0);

		//Performance hacks:
		///Iterations with more than half of the initial_matches inlying, count twice
		if (inlier.size() > initial_matches.size() * 0.5) n_iter++;
		///Iterations with more than 80% of the initial_matches inlying, count threefold
		if (inlier.size() > initial_matches.size() * 0.8) n_iter++;

		if (inlier_error < best_error) { //copy this to the result
			result_transform = transformation;
			if (matches != nullptr) *matches = inlier;
//			assert(matches.size() >= min_inlier_threshold);
//			assert(matches.size()>= ((float)initial_matches.size()) * min_inlier_ratio);
			best_inlier_cnt = inlier.size();
			rmse = inlier_error;
			best_error = inlier_error;
		}
// 		else
// 		{
// 			
// 		}

		//int max_ndx = min((int) min_inlier_threshold,30); //? What is this 30?
		double new_inlier_error;

		transformation = Feature::getTransformFromMatches(valid, earlier, now, inlier); // compute new trafo from all inliers:
		if (transformation != transformation) continue; //Contains NaN
		Feature::computeInliersAndError(inlier, new_inlier_error, nullptr,
			initial_matches, transformation,
			earlier->feature_pts_3d, now->feature_pts_3d,
			squared_max_dist_m);

		// check also invalid iterations
		if (inlier.size() > best_inlier_invalid)
		{
			best_inlier_invalid = inlier.size();
			best_error_invalid = inlier_error;
		}

		if(inlier.size() < min_inlier_threshold || new_inlier_error > max_dist_m)
		{
			continue;
		}

//		assert(new_inlier_error > 0);

		if (new_inlier_error < best_error) 
		{
			result_transform = transformation;
			if (matches != nullptr) *matches = inlier;
//			assert(matches->size() >= min_inlier_threshold);
//			assert(matches.size()>= ((float)initial_matches->size())*min_inlier_ratio);
			best_inlier_cnt = inlier.size();
			rmse = new_inlier_error;
			best_error = new_inlier_error;
		}
// 		else
// 		{
// 
// 		}
	} //iterations

	return best_inlier_cnt >= min_inlier_threshold;
}