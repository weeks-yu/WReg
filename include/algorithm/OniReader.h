#pragma once;

#include "RGBDReader.h"
#include <OpenNI.h>

using namespace openni;

class OniReader : public RGBDReader
{
public:
	OniReader();
	virtual ~OniReader();

	virtual bool getNextFrame(cv::Mat &rgb, cv::Mat &depth, double &timestamp);
	virtual void registerColorToDepth(const cv::Mat &rgb, const cv::Mat &depth, cv::Mat &rgbRegistered);
	virtual void registerDepthToColor(const cv::Mat &rgb, const cv::Mat &depth, cv::Mat &depthRegistered);

	virtual bool create(const char* filename = NULL);
	virtual void start();
	virtual void stop();

private:
	std::string filename;

	Device oniDevice;
	VideoStream depthStream;
	VideoStream colorStream;
	PlaybackControl *control;

	int rgbNow;
	int rgbCount;
	int depthNow;
	int depthCount;
};