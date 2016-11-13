#include "OrbRegister.h"
#include "Frame.h"

OrbRegister::OrbRegister()
{
	this->min_matches = Config::instance()->get<int>("min_matches");
	this->inlier_percentage = Config::instance()->get<float>("min_inlier_p");
	this->inlier_dist = Config::instance()->get<float>("max_inlier_dist");
}

OrbRegister::OrbRegister(int min_matches, float inlier_percentage, float inlier_dist)
{
	this->min_matches = min_matches;
	this->inlier_percentage = inlier_percentage;
	this->inlier_dist = inlier_dist;
}

OrbRegister::~OrbRegister()
{

}

bool OrbRegister::getTransformation(void *prev, void *now, Eigen::Matrix4f &tran)
{
	Frame *prev_frame = static_cast<Frame *>(prev);
	Frame *now_frame = static_cast<Frame *>(now);

	vector<cv::DMatch> matches, inliers;

	Eigen::Matrix<double, 6, 6> information;
	float rmse;

	prev_frame->f->findMatchedPairsBruteForce(matches, now_frame->f);
	if (Feature::getTransformationByRANSAC(tran, rmse, &inliers,
		prev_frame->f, now_frame->f, matches, min_matches, inlier_percentage, inlier_dist))
	{
		return true;
	}
	return false;
}

float OrbRegister::getCorrespondencePercent(void *last, void *now, Eigen::Matrix4f &estimatedTran)
{
	Frame *last_frame = static_cast<Frame *>(last);
	Frame *now_frame = static_cast<Frame *>(now);
	vector<cv::DMatch> matches, inliers;
	float rmse;

	last_frame->f->findMatchedPairsBruteForce(matches, now_frame->f);
	Feature::computeInliersAndError(inliers, rmse, nullptr, matches,
		estimatedTran,
		last_frame->f, now_frame->f, inlier_dist * inlier_dist);

	return matches.size() > 0 ? (float)inliers.size() / matches.size() : 0;
}

void OrbRegister::setParameters(void **parameters)
{
	min_matches = *static_cast<int *>(parameters[0]);
	inlier_percentage = *static_cast<float *>(parameters[1]);
	inlier_dist = *static_cast<float *>(parameters[2]);
}