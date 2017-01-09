#pragma once;

#include "RGBDReader.h"
#include <OpenNI.h>

using namespace openni;

class ImageReader : public RGBDReader
{
public:
	ImageReader();
	virtual ~ImageReader();

	virtual bool getNextFrame(cv::Mat &rgb, cv::Mat &depth, double &timestamp);
	virtual void registerColorToDepth(const cv::Mat &rgb, const cv::Mat &depth, cv::Mat &rgbRegistered);
	virtual void registerDepthToColor(const cv::Mat &rgb, const cv::Mat &depth, cv::Mat &depthRegistered);

	virtual bool create(const char* filename = NULL);
	virtual void start();
	virtual void stop();

private:
	std::string filename;

	std::ifstream *fileInput;
	std::queue<std::string> rgbStream, depthStream;
};