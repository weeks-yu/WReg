#pragma once

#include <opencv2/core/core.hpp>
#include <queue>

struct Intrinsic
{
	float rx, ry;
	float fx, fy;
	float cx, cy;
	float zFactor;
	Intrinsic() : rx(0), ry(0), fx(0), fy(0), cx(0), cy(0), zFactor(1.0) {}
	Intrinsic(float rx_, float ry_, float fx_, float fy_, float cx_, float cy_, float zFactor_) 
		: rx(rx_), ry(ry_)
		, fx(fx_), fy(fy_)
		, cx(cx_), cy(cy_)
		, zFactor(zFactor_) { }

	Intrinsic operator()(int level) const
	{
		int div = 1 << level;
		return (Intrinsic(rx / div, ry / div, fx / div, fy / div, cx / div, cy / div, zFactor));
	}
};

class RGBDReader
{
// public:
// 	typedef std::pair<cv::Mat, cv::Mat> RGBDFrame;

public:
	RGBDReader();
	virtual ~RGBDReader();

	virtual bool getNextColorFrame(cv::Mat &rgb);
	virtual bool getNextDepthFrame(cv::Mat &depth);
	virtual unsigned short getMaxDepth();

	virtual bool create(const char* mode) = 0;
	virtual bool start() = 0;
	virtual void stop() = 0;

public:
	Intrinsic intr;

public:
	std::queue<cv::Mat> color_frames, depth_frames;
	unsigned short max_depth;
};