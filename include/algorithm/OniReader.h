#pragma once;

#include "RGBDReader.h"
#include <OpenNI.h>

using namespace openni;

class OniReader : public RGBDReader
{
public:
	OniReader();
	virtual ~OniReader();

	virtual bool create(const char* filename);
	virtual bool start();
	virtual void stop();

private:
	std::string filename;

	Device oniDevice;
	VideoStream depthStream;
	VideoStream colorStream;
	PlaybackControl *control;
};