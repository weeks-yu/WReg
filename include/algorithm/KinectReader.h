#pragma once;

#include "RGBDReader.h"
#include <OpenNI.h>

using namespace openni;

class KinectReader : public RGBDReader, public VideoStream::NewFrameListener
{
public:
	KinectReader();
	virtual ~KinectReader();

	virtual bool create();
	virtual bool start();
	virtual void stop();
	virtual void onNewFrame(VideoStream &vs);

private:
	std::string filename;

	Device kinectDevice;
	VideoStream depthStream;
	VideoStream colorStream;
};