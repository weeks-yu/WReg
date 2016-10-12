#include "RGBDReader.h"

RGBDReader::RGBDReader() : intr(), max_depth(1.0f)
{

}

RGBDReader::~RGBDReader()
{

}

bool RGBDReader::getNextColorFrame(cv::Mat &rgb)
{
	if (!color_frames.empty())
	{
		cv::Mat r = color_frames.front();
		r.copyTo(rgb);
		color_frames.pop();
		return true;
	}
	return false;
}

bool RGBDReader::getNextDepthFrame(cv::Mat &depth)
{
	if (!depth_frames.empty())
	{
		cv::Mat d = depth_frames.front();
		d.copyTo(depth);
		depth_frames.pop();
		return true;
	}
	return false;
}

unsigned short RGBDReader::getMaxDepth()
{
	return max_depth;
}