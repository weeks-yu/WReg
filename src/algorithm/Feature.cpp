#include "Feature.h"

#include <algorithm>
#include <opencv2/nonfree/features2d.hpp>
// #include <opencv2/nonfree/nonfree.hpp>
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

void Feature::ORBExtractor(vector<cv::KeyPoint> &feature_pts,
	vector_eigen_vector3f &feature_pts_3d,
	cv::Mat &feature_descriptors,
	const cv::Mat &imgRGB, const cv::Mat &imgDepth)
{
	cv::ORB orb_detector;
	cv::Mat mask, descriptors;
	vector<cv::KeyPoint> fpts;

	orb_detector(imgRGB, mask, fpts, descriptors);
	descriptors.convertTo(descriptors, CV_32F);

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
	if (type == "SIFT")
	{
		Feature::SIFTExtractor(feature_pts, feature_pts_3d, feature_descriptors, imgRGB, imgDepth);
	}
	else if (type == "SURF")
	{
		Feature::SURFExtractor(feature_pts, feature_pts_3d, feature_descriptors, imgRGB, imgDepth);
	}
	else if (type == "ORB")
	{
		Feature::ORBExtractor(feature_pts, feature_pts_3d, feature_descriptors, imgRGB, imgDepth);
	}
	for (int i = 0; i < feature_pts.size(); i++)
	{
		feature_ids.push_back(0);
	}
	depth_image = imgDepth;
	flann_index = nullptr;
	this->trees = Config::instance()->get<int>("kdtree_trees");
}

void Feature::buildFlannIndex()
{
	releaseFlannIndex();
	if (feature_pts.size() > 0)
	{
		this->flann_index = new cv::flann::Index(feature_descriptors, cv::flann::KDTreeIndexParams(this->trees));
	}
	else
	{
		this->flann_index = nullptr;
	}
}

void Feature::releaseFlannIndex()
{
	if (flann_index)
	{
		delete flann_index;
		flann_index = nullptr;
	}
}

int Feature::findMatched(vector<cv::DMatch> &matches, const cv::Mat &descriptor, int max_leafs /* = 64 */, int k /* = 2 */)
{
	if (this->flann_index == nullptr || k != 1 && k != 2)
	{
		return -1;
	}

	cv::Mat indices(descriptor.rows, k, CV_32S);
	cv::Mat dists(descriptor.rows, k, CV_32F);

	// get the best two neighbours
	this->flann_index->knnSearch(descriptor, indices, dists, k, cv::flann::SearchParams(max_leafs));

	int* indices_ptr = indices.ptr<int>(0);
	float* dists_ptr = dists.ptr<float>(0);

	float ratio = Config::instance()->get<float>("matches_criterion");

	cv::DMatch match;
	for (int i = 0; i < indices.rows; i++)
	{
		if (k == 1 || dists_ptr[k * i] < ratio * dists_ptr[k * i + 1])
		{
			match.queryIdx = i;
			match.trainIdx = indices_ptr[k * i];
			match.distance = dists_ptr[k * i];
			matches.push_back(match);
		}
	}

	return matches.size();
}

int Feature::findMatchedPairs(vector<cv::DMatch> &matches, const Feature *other, int max_leafs, int k)
{
	return findMatched(matches, other->feature_descriptors, max_leafs, k);
}

bool Feature::findMatchedPairsMultiple(vector<int> &frames, vector<vector<cv::DMatch>> &matches, const Feature *other, int k, int max_leafs)
{
	if (this->flann_index == nullptr || !this->multiple)
	{
		return false;
	}

	int frame_count = this->getFrameCount();

	if (frame_count * 4 < k)
	{
		k = frame_count * 4;
	}

	cv::Mat indices(other->feature_descriptors.rows, k, CV_32S);
	cv::Mat dists(other->feature_descriptors.rows, k, CV_32F);

	// get the best two neighbours
	this->flann_index->knnSearch(other->feature_descriptors, indices, dists, k, cv::flann::SearchParams(max_leafs));

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

// void Feature::transform(const Eigen::Matrix4f tran, int frame_index)
// {
// 	for (int i = 0; i < this->feature_pts_3d.size(); i++)
// 	{
// 		if (!this->multiple || frame_index < 0 || this->feature_frame_index[i] == frame_index)
// 		{
// 			this->feature_pts_3d[i] = tran * this->feature_pts_3d[i];
// 		}
// 	}
// }

// void Feature::append(const Feature &other, int frame_index)
// {
// 	for (int i = 0; i < other.feature_pts.size(); i++)
// 	{
// 		this->feature_pts.push_back(other.feature_pts[i]);
// 		this->feature_descriptors.push_back(other.feature_descriptors.row(i));
// 
// 		if (multiple)
// 		{
// 			this->feature_frame_index.push_back(frame_index);
// 		}
// 	}
// }

int Feature::getFrameCount()
{
	std::set<int> frame_index;

	for (int i = 0; i < this->feature_frame_index.size(); i++)
	{
		frame_index.insert(this->feature_frame_index[i]);
	}
	return frame_index.size();
}

void Feature::updateFeaturePoints3D(const Eigen::Matrix4f &tran)
{
	Eigen::Affine3f a(tran);
	feature_pts_3d_real.clear();
	for (int i = 0; i < feature_pts.size(); i++)
	{
		feature_pts_3d_real.push_back(a * feature_pts_3d[i]);
	}
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
		tfc.add(from, to, 1.0 / to(2)); //the further, the less weight b/c of accuracy decay
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
	const vector_eigen_vector3f &earlier, const vector_eigen_vector3f &now,
	double squaredMaxInlierDistInM)
{
	Eigen::Affine3f a(transformation);
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

		Eigen::Vector3f vec = (a * now[this_id]) - earlier[earlier_id];

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

bool Feature::getTransformationByRANSAC(Eigen::Matrix4f &result_transform,
	Eigen::Matrix<double, 6, 6> &result_information,
	float &result_coresp, 
	float &rmse, vector<cv::DMatch> *matches, // output vars. if matches == nullptr, do not return inlier match
	const Feature* earlier, const Feature* now,
	PointCloudCuda *pcc,
	const vector<cv::DMatch> &initial_matches,
	unsigned int ransac_iterations)
{
	if (matches != nullptr)	matches->clear();

	if (initial_matches.size() < (unsigned int) Config::instance()->get<int>("min_matches"))
	{
		return false;
	}

	unsigned int min_inlier_threshold = (unsigned int)(initial_matches.size() * Config::instance()->get<float>("min_inliers_percent"));
	std::vector<cv::DMatch> inlier; //holds those feature correspondences that support the transformation
	double inlier_error; //all squared errors
	srand((long)std::clock());

	// a point is an inlier if it's no more than max_dist_m m from its partner apart
	const float max_dist_m = Config::instance()->get<float>("max_dist_for_inliers");
	const float squared_max_dist_m = max_dist_m * max_dist_m;

	// best values of all iterations (including invalids)
	double best_error = 1e6, best_error_invalid = 1e6;
	unsigned int best_inlier_invalid = 0, best_inlier_cnt = 0, valid_iterations = 0;
	Eigen::Matrix4f transformation;

	const unsigned int sample_size = 3;// chose this many randomly from the correspondences:

	int threads = Config::instance()->get<int>("icpcuda_threads");
	int blocks = Config::instance()->get<int>("icpcuda_blocks");
	float corr_percent = Config::instance()->get<float>("coresp_percent");
	Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Identity();
	float temp_coresp = 0;

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

		if (inlier_error < best_error)
		{ //copy this to the result
			if (pcc != nullptr)
			{
				Eigen::Vector3f t = transformation.topRightCorner(3, 1);
				Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rot = transformation.topLeftCorner(3, 3);
				int point_count, point_corr_count;
				pcc->getCoresp(t, rot, information, point_count, point_corr_count, threads, blocks);
				temp_coresp = (float)point_corr_count / point_count;
				if (temp_coresp < corr_percent) continue;
			}

			result_transform = transformation;
			result_information = information;
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
			if (pcc != nullptr)
			{
				Eigen::Vector3f t = transformation.topRightCorner(3, 1);
				Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rot = transformation.topLeftCorner(3, 3);
				int point_count, point_corr_count;
				pcc->getCoresp(t, rot, information, point_count, point_corr_count, threads, blocks);
				temp_coresp = (float)point_corr_count / point_count;
				if (temp_coresp < corr_percent) continue;
			}

			result_transform = transformation;
			if (pcc != nullptr)
				result_information = information;
			result_coresp = temp_coresp;
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
	
	if (pcc == nullptr)
		result_information = Eigen::Matrix<double, 6, 6>::Identity();
	return best_inlier_cnt >= min_inlier_threshold;
}

bool Feature::getPlanesByRANSAC(Eigen::Vector4f &result_plane, vector<pair<int, int>> *matches,
	const cv::Mat &depth, const vector<pair<int, int>> &initial_point_indices)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

	// Generate the data
	for (size_t i = 0; i < initial_point_indices.size(); ++i)
	{
		Eigen::Vector3f point = ConvertPointTo3D(initial_point_indices[i].second, initial_point_indices[i].first, depth);
		cloud->push_back(pcl::PointXYZ(point(0), point(1), point(2)));
	}

	pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
	pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
	// Create the segmentation object
	
	pcl::SACSegmentation<pcl::PointXYZ> seg;
	// Optional
	seg.setOptimizeCoefficients(true);
	// Mandatory
	seg.setModelType(pcl::SACMODEL_PLANE);
	seg.setMethodType(pcl::SAC_RANSAC);
	//seg.setMaxIterations(Config::instance()->get<int>("plane_max_iteration"));
	seg.setDistanceThreshold(Config::instance()->get<float>("plane_dist_threshold"));

	seg.setInputCloud(cloud);
	seg.segment(*inliers, *coefficients);

	if (inliers->indices.size() == 0)
	{
		return false;
	}

	result_plane(0) = coefficients->values[0];
	result_plane(1) = coefficients->values[1];
	result_plane(2) = coefficients->values[2];
	result_plane(3) = coefficients->values[3];
	result_plane /= sqrt(result_plane(0) * result_plane(0) + result_plane(1) * result_plane(1) + result_plane(2) * result_plane(2));

	if (matches != nullptr)
	{
		matches->clear();
		for (size_t i = 0; i < inliers->indices.size(); ++i)
		{
			matches->push_back(initial_point_indices[inliers->indices[i]]);
		}
	}

	return true;
}

float Feature::distToPlane(const Eigen::Vector3f &point, const Eigen::Vector4f &plane)
{
	float a = plane(0) * point(0) + plane(1) * point(1) + plane(2) * point(2) + plane(3);
	return fabs(a);
}