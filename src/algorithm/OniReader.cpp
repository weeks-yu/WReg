#include "OniReader.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

OniReader::OniReader()
{
	filename = "";
}

OniReader::~OniReader()
{

}

bool OniReader::getNextFrame(cv::Mat &rgb, cv::Mat &depth, double &timestamp)
{
	VideoFrameRef oniColor, oniDepth;
	if (colorStream.readFrame(&oniColor) == STATUS_OK && depthStream.readFrame(&oniDepth) == STATUS_OK)
	{
		cv::Mat r(oniColor.getHeight(), oniColor.getWidth(), CV_8UC3, (void *)oniColor.getData());
		cv::cvtColor(r, rgb, CV_RGB2BGR);

		cv::Mat d(oniDepth.getHeight(), oniDepth.getWidth(), CV_16UC1, (void *)oniDepth.getData());
		d.copyTo(depth);
		return true;
	}
	return false;
}

void OniReader::registerColorToDepth(const cv::Mat &rgb, const cv::Mat &depth, cv::Mat &rgbRegistered)
{
	throw std::exception("Exception : not implemented.");
}

void OniReader::registerDepthToColor(const cv::Mat &rgb, const cv::Mat &depth, cv::Mat &depthRegistered)
{
	depth.copyTo(depthRegistered);
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

	intrDepth.rx = videoMode.getResolutionX();
	intrDepth.ry = videoMode.getResolutionY();
	intrDepth.fx = intrDepth.rx / 2 / tan(hFov / 2);
	intrDepth.fy = intrDepth.ry / 2 / tan(vFov / 2);
	intrDepth.cx = 0.5 * intrDepth.rx;
	intrDepth.cy = 0.5 * intrDepth.ry;
	switch (videoMode.getPixelFormat())
	{
	case ONI_PIXEL_FORMAT_DEPTH_1_MM:
		intrDepth.zFactor = 1000.0f;
		break;
	case ONI_PIXEL_FORMAT_DEPTH_100_UM:
		intrDepth.zFactor = 10000.0f;
		break;
	}

	// playback control
	control = oniDevice.getPlaybackControl();
	control->setSpeed(-1);

	return true;
}

void OniReader::start()
{
	Status result;

	result = depthStream.start();
	if (result != STATUS_OK) return;

	result = colorStream.start();
	if (result != STATUS_OK) return;

	return;
}

void OniReader::stop()
{
	
}