#pragma once;

#include "RGBDReader.h"
#include <OpenNI.h>

using namespace openni;

class OniReader : public RGBDReader
{
public:
	OniReader();
	virtual ~OniReader();

	virtual bool getNextColorFrame(cv::Mat &rgb);
	virtual bool getNextDepthFrame(cv::Mat &depth);
	virtual bool getNextFrame(cv::Mat &rgb, cv::Mat &depth);

	virtual bool create(const char* filename);
	virtual void start();
	virtual void stop();

private:
	std::string filename;

	Device oniDevice;
	VideoStream depthStream;
	VideoStream colorStream;
	PlaybackControl *control;
};