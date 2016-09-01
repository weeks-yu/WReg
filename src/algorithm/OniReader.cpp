#include "OniReader.h"
#include <opencv2/imgproc/imgproc.hpp>

OniReader::OniReader()
{
	filename = "";
}

OniReader::~OniReader()
{

}

bool OniReader::create(const char* filename_)
{
	filename = std::string(filename_);
	Status result = STATUS_OK;

	result = OpenNI::initialize();
	if (result != STATUS_OK) return false;

	result = oniDevice.open(filename_);
	if (result != STATUS_OK) return false;

	// depthstream and color stream
	result = depthStream.create(oniDevice, SENSOR_DEPTH);
	if (result != STATUS_OK) return false;
	max_depth = (unsigned short)depthStream.getMaxPixelValue();

	result = oniDevice.setImageRegistrationMode(IMAGE_REGISTRATION_DEPTH_TO_COLOR);
	if (result != STATUS_OK) return false;

	result = colorStream.create(oniDevice, SENSOR_COLOR);
	if (result != STATUS_OK) return false;

	// Intrinsic
	VideoMode videoMode = depthStream.getVideoMode();
	float hFov = depthStream.getHorizontalFieldOfView();
	float vFov = depthStream.getVerticalFieldOfView();

	intr.rx = videoMode.getResolutionX();
	intr.ry = videoMode.getResolutionY();
	intr.fx = intr.rx / 2 / tan(hFov / 2);
	intr.fy = intr.ry / 2 / tan(vFov / 2);
	intr.cx = 0.5 * intr.rx;
	intr.cy = 0.5 * intr.ry;
	switch (videoMode.getPixelFormat())
	{
	case ONI_PIXEL_FORMAT_DEPTH_1_MM:
		intr.zFactor = 1.0f;
		break;
	case ONI_PIXEL_FORMAT_DEPTH_100_UM:
		intr.zFactor = 0.1f;
		break;
	}

	// playback control
	control = oniDevice.getPlaybackControl();
	control->setSpeed(-1);

	color_frames.clear();
	depth_frames.clear();

	return true;
}

bool OniReader::start()
{
	Status result;

	result = depthStream.start();
	if (result != STATUS_OK) return false;

	result = colorStream.start();
	if (result != STATUS_OK) return false;
	
	VideoFrameRef oniDepth, oniColor;
	
	for (int i = 0; i < control->getNumberOfFrames(colorStream); i++)
	{
		if (colorStream.readFrame(&oniColor) == STATUS_OK)
		{
			cv::Mat r(oniColor.getHeight(), oniColor.getWidth(), CV_8UC3, (void *)oniColor.getData());
			cv::cvtColor(r, r, CV_RGB2BGR);
			color_frames.push_back(r);
		}
	}

	for (int i = 0; i < control->getNumberOfFrames(depthStream); i++)
	{
		if (depthStream.readFrame(&oniDepth) == STATUS_OK)
		{
			cv::Mat d(oniDepth.getHeight(), oniDepth.getWidth(), CV_16UC1, (void *)oniDepth.getData());
			depth_frames.push_back(d);
		}
	}

	return true;
}

void OniReader::stop()
{
	
}