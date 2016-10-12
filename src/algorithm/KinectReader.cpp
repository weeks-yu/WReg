#include "KinectReader.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

KinectReader::KinectReader()
{
	filename = "";
}

KinectReader::~KinectReader()
{

}

bool KinectReader::create()
{
	Status result = STATUS_OK;

	result = OpenNI::initialize();
	if (result != STATUS_OK) return false;

	result = kinectDevice.open(openni::ANY_DEVICE);
	if (result != STATUS_OK) return false;

	// depthstream and color stream
	result = depthStream.create(kinectDevice, SENSOR_DEPTH);
	if (result != STATUS_OK) return false;
	max_depth = (unsigned short)depthStream.getMaxPixelValue();

	result = kinectDevice.setImageRegistrationMode(IMAGE_REGISTRATION_DEPTH_TO_COLOR);
	if (result != STATUS_OK) return false;

	result = colorStream.create(kinectDevice, SENSOR_COLOR);
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

	while (!color_frames.empty())
		color_frames.pop();
	while (!depth_frames.empty())
		depth_frames.pop();

	return true;
}

bool KinectReader::start()
{
	Status result;

	depthStream.addNewFrameListener(this);
	colorStream.addNewFrameListener(this);

	result = depthStream.start();
	if (result != STATUS_OK) return false;

	result = colorStream.start();
	if (result != STATUS_OK) return false;

	return true;
}

void KinectReader::stop()
{
	
}

void KinectReader::onNewFrame(VideoStream &vs)
{
	SensorType st = vs.getSensorInfo().getSensorType();
	if (st == SENSOR_DEPTH)
	{
		VideoFrameRef vfr;
		if (vs.readFrame(&vfr) == STATUS_OK)
		{
			cv::Mat d(vfr.getHeight(), vfr.getWidth(), CV_16UC1, (void *)vfr.getData());
			cv::Mat d2;
			d.copyTo(d2);
			depth_frames.push(d2);
		}
	}
	else if (st == SENSOR_COLOR)
	{
		VideoFrameRef vfr;
		if (colorStream.readFrame(&vfr) == STATUS_OK)
		{
			cv::Mat r(vfr.getHeight(), vfr.getWidth(), CV_8UC3, (void *)vfr.getData());
			cv::Mat r2;
			cv::cvtColor(r, r2, CV_RGB2BGR);
			color_frames.push(r2);
		}
	}
}