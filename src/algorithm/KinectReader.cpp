#include "KinectReader.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>

KinectReader::KinectReader()
{
	running = false;
}

KinectReader::~KinectReader()
{
	SafeRelease(sensor);
	SafeRelease(mapper);
	SafeRelease(colorSource);
	SafeRelease(colorReader);
	SafeRelease(depthSource);
	SafeRelease(depthReader);
	SafeRelease(infraredSource);
	SafeRelease(infraredReader);
}

bool KinectReader::getNextColorFrame(cv::Mat &rgb)
{
	IColorFrame* colorFrame = nullptr;
	HRESULT result = colorReader->AcquireLatestFrame(&colorFrame);
	if (FAILED(result))
	{
		return false;
	}
	// Retrieved Color Data
	std::vector<RGBQUAD> data(intrColor.rx * intrColor.ry);
	result = colorFrame->CopyConvertedFrameDataToArray(intrColor.rx * intrColor.ry * sizeof(RGBQUAD),
		reinterpret_cast<BYTE*>(&data[0]), ColorImageFormat::ColorImageFormat_Bgra);
	if (FAILED(result))
	{
		throw std::exception("Exception : IColorFrame::CopyConvertedFrameDataToArray()");
		return false;
	}
	cv::Mat r(intrColor.ry, intrColor.rx, CV_8UC4, (void *)data.data());
	r.copyTo(rgb);
	SafeRelease(colorFrame);
	return true;

// 	VideoFrameRef oniColor;
// 	if (colorStream.readFrame(&oniColor) == STATUS_OK)
// 	{
// 		ICoordinateMapper *cMapper = 
// 		cv::Mat r(oniColor.getHeight(), oniColor.getWidth(), CV_8UC3, (void *)oniColor.getData());
// 		cv::cvtColor(r, rgb, CV_RGB2BGR);
// // 		if (intrDepth.rx != intrColor.rx && intrDepth.ry != intrColor.ry)
// // 		{
// // 			//cv::resize(rgb, rgb, cv::Size(intrDepth.rx, intrDepth.ry));
// // 		}
// 		return true;
// 	}
// 	return false;
}

bool KinectReader::getNextDepthFrame(cv::Mat &depth)
{
	IDepthFrame* depthFrame = nullptr;
	HRESULT result = depthReader->AcquireLatestFrame(&depthFrame);
	if (FAILED(result))
	{
		return false;
	}
	// Retrieved Depth Data
	std::vector<UINT16> data(intrDepth.rx * intrDepth.ry);
	result = depthFrame->CopyFrameDataToArray(intrDepth.rx * intrDepth.ry, &data[0]);
	if (FAILED(result))
	{
		throw std::exception("Exception : IDepthFrame::CopyFrameDataToArray()");
		return false;
	}
	cv::Mat d(intrDepth.ry, intrDepth.rx, CV_16UC1, (void *)data.data());
	d.copyTo(depth);
	SafeRelease(depthFrame);
	return true;
// 	VideoFrameRef oniDepth;
// 	if (depthStream.readFrame(&oniDepth) == STATUS_OK)
// 	{
// 		cv::Mat d(oniDepth.getHeight(), oniDepth.getWidth(), CV_16UC1, (void *)oniDepth.getData());
// 		d.copyTo(depth);
// // 		if (intrDepth.rx != intrColor.rx && intrDepth.ry != intrColor.ry)
// // 		{
// // 			cv::resize(depth, depth, cv::Size(intrColor.rx, intrColor.ry));
// // 		}
// 		return true;
// 	}
// 	return false;
}

bool KinectReader::getNextFrame(cv::Mat &rgb, cv::Mat &depth)
{
	std::vector<RGBQUAD> colorData;
	std::vector<UINT16> depthData;
	bool isColorSucceeded = false, isDepthSucceeded = false;

	IColorFrame* colorFrame = nullptr;
	HRESULT result = colorReader->AcquireLatestFrame(&colorFrame);
	if (SUCCEEDED(result))
	{
		colorData.resize(intrColor.rx * intrColor.ry);
		result = colorFrame->CopyConvertedFrameDataToArray(intrColor.rx * intrColor.ry * sizeof(RGBQUAD),
			reinterpret_cast<BYTE*>(&colorData[0]), ColorImageFormat::ColorImageFormat_Bgra);
		if (FAILED(result))
		{
			throw std::exception("Exception : IColorFrame::CopyConvertedFrameDataToArray()");
			return false;
		}
		isColorSucceeded = true;
	}
	SafeRelease(colorFrame);

	IDepthFrame* depthFrame = nullptr;
	result = depthReader->AcquireLatestFrame(&depthFrame);
	if (SUCCEEDED(result))
	{
		depthData.resize(intrDepth.rx * intrDepth.ry);
		result = depthFrame->CopyFrameDataToArray(intrDepth.rx * intrDepth.ry, &depthData[0]);
		if (FAILED(result))
		{
			throw std::exception("Exception : IDepthFrame::CopyFrameDataToArray()");
			return false;
		}
		isDepthSucceeded = true;
	}
	SafeRelease(depthFrame);

	if (!isColorSucceeded || !isDepthSucceeded)
	{
		return false;
	}

	cv::Mat d(intrDepth.ry, intrDepth.rx, CV_16UC1, (void *)depthData.data());
	d.copyTo(depth);

	// registration
	std::vector<DepthSpacePoint> mappedData(intrColor.rx * intrColor.ry);
	mapper->MapColorFrameToDepthSpace(intrDepth.rx * intrDepth.ry, depthData.data(), intrColor.rx * intrColor.ry, mappedData.data());
	cv::Mat r = cv::Mat::zeros(intrDepth.ry, intrDepth.rx, CV_8UC3);
	for (int i = 0; i < intrColor.ry; i++)
	{
		for (int j = 0; j < intrColor.rx; j++)
		{
			float u = mappedData[intrColor.rx * i + j].X;
			float v = mappedData[intrColor.rx * i + j].Y;
			if (u >= 0 &&  u < intrDepth.rx && v >= 0 && v < intrDepth.ry)
			{
				r.at<cv::Vec3b>(v, u)[0] = colorData[intrColor.rx * i + j].rgbBlue;
				r.at<cv::Vec3b>(v, u)[1] = colorData[intrColor.rx * i + j].rgbGreen;
				r.at<cv::Vec3b>(v, u)[2] = colorData[intrColor.rx * i + j].rgbRed;
			}
		}
	}
	r.copyTo(rgb);

	SafeRelease(colorFrame);
	SafeRelease(depthFrame);

	return true;
}

bool KinectReader::create(const char* mode)
{
	// Create Sensor Instance
	HRESULT result = GetDefaultKinectSensor(&sensor);
	if (FAILED(result))
	{
		throw std::exception("Exception : GetDefaultKinectSensor()");
	}

	// Open Sensor
	result = sensor->Open();
	if (FAILED(result)){
		throw std::exception("Exception : IKinectSensor::Open()");
	}

	// Retrieved Coordinate Mapper
	result = sensor->get_CoordinateMapper(&mapper);
	if (FAILED(result)){
		throw std::exception("Exception : IKinectSensor::get_CoordinateMapper()");
	}

	// Retrieved Color Frame Source
	result = sensor->get_ColorFrameSource(&colorSource);
	if (FAILED(result)){
		throw std::exception("Exception : IKinectSensor::get_ColorFrameSource()");
	}

	// Retrieved Depth Frame Source
	result = sensor->get_DepthFrameSource(&depthSource);
	if (FAILED(result)){
		throw std::exception("Exception : IKinectSensor::get_DepthFrameSource()");
	}

	// Retrieved Infrared Frame Source
	result = sensor->get_InfraredFrameSource(&infraredSource);
	if (FAILED(result)){
		throw std::exception("Exception : IKinectSensor::get_InfraredFrameSource()");
	}

	// Retrieved Color Frame Size
	IFrameDescription* colorDescription;
	result = colorSource->get_FrameDescription(&colorDescription);
	if (FAILED(result)){
		throw std::exception("Exception : IColorFrameSource::get_FrameDescription()");
	}

	result = colorDescription->get_Width(&intrColor.rx); // 1920
	if (FAILED(result)){
		throw std::exception("Exception : IFrameDescription::get_Width()");
	}

	result = colorDescription->get_Height(&intrColor.ry); // 1080
	if (FAILED(result)){
		throw std::exception("Exception : IFrameDescription::get_Height()");
	}

	float hFov, vFov;
	result = colorDescription->get_HorizontalFieldOfView(&hFov);
	if (FAILED(result)){
		throw std::exception("Exception : IFrameDescription::get_HorizontalFieldOfView()");
	}

	result = colorDescription->get_VerticalFieldOfView(&vFov);
	if (FAILED(result)){
		throw std::exception("Exception : IFrameDescription::get_VerticalFieldOfView()");
	}
	intrColor.fx = intrColor.rx / 2 / tan(hFov / 2);
	intrColor.fy = intrColor.ry / 2 / tan(vFov / 2);
	intrColor.cx = 0.5 * intrColor.rx;
	intrColor.cy = 0.5 * intrColor.ry;

	SafeRelease(colorDescription);

	// To Reserve Color Frame Buffer
/*	colorBuffer.resize(colorWidth * colorHeight);*/

	// Retrieved Depth Frame Size
	IFrameDescription* depthDescription;
	result = depthSource->get_FrameDescription(&depthDescription);
	if (FAILED(result)){
		throw std::exception("Exception : IDepthFrameSource::get_FrameDescription()");
	}

	result = depthDescription->get_Width(&intrDepth.rx); // 512
	if (FAILED(result)){
		throw std::exception("Exception : IFrameDescription::get_Width()");
	}

	result = depthDescription->get_Height(&intrDepth.ry); // 424
	if (FAILED(result)){
		throw std::exception("Exception : IFrameDescription::get_Height()");
	}

	result = depthDescription->get_HorizontalFieldOfView(&hFov);
	if (FAILED(result)){
		throw std::exception("Exception : IFrameDescription::get_HorizontalFieldOfView()");
	}

	result = depthDescription->get_VerticalFieldOfView(&vFov);
	if (FAILED(result)){
		throw std::exception("Exception : IFrameDescription::get_VerticalFieldOfView()");
	}
	intrDepth.fx = intrDepth.rx / 2 / tan(hFov / 2);
	intrDepth.fy = intrDepth.ry / 2 / tan(vFov / 2);
	intrDepth.cx = 0.5 * intrDepth.rx;
	intrDepth.cy = 0.5 * intrDepth.ry;

	SafeRelease(depthDescription);

	// To Reserve Depth Frame Buffer
/*	depthBuffer.resize(depthWidth * depthHeight);*/

	// Retrieved Infrared Frame Size
	IFrameDescription* infraredDescription;
	result = infraredSource->get_FrameDescription(&infraredDescription);
	if (FAILED(result)){
		throw std::exception("Exception : IInfraredFrameSource::get_FrameDescription()");
	}

	result = infraredDescription->get_Width(&intrInfrared.rx); // 512
	if (FAILED(result)){
		throw std::exception("Exception : IFrameDescription::get_Width()");
	}

	result = infraredDescription->get_Height(&intrInfrared.ry); // 424
	if (FAILED(result)){
		throw std::exception("Exception : IFrameDescription::get_Height()");
	}

	result = infraredDescription->get_HorizontalFieldOfView(&hFov);
	if (FAILED(result)){
		throw std::exception("Exception : IFrameDescription::get_HorizontalFieldOfView()");
	}

	result = infraredDescription->get_VerticalFieldOfView(&vFov);
	if (FAILED(result)){
		throw std::exception("Exception : IFrameDescription::get_VerticalFieldOfView()");
	}
	intrInfrared.fx = intrInfrared.rx / 2 / tan(hFov / 2);
	intrInfrared.fy = intrInfrared.ry / 2 / tan(vFov / 2);
	intrInfrared.cx = 0.5 * intrInfrared.rx;
	intrInfrared.cy = 0.5 * intrInfrared.ry;

	SafeRelease(infraredDescription);

	return true;
	// To Reserve Infrared Frame Buffer
/*	infraredBuffer.resize(infraredWidth * infraredHeight);*/

// 	signal_PointXYZ = createSignal<signal_Kinect2_PointXYZ>();
// 	signal_PointXYZI = createSignal<signal_Kinect2_PointXYZI>();
// 	signal_PointXYZRGB = createSignal<signal_Kinect2_PointXYZRGB>();
// 	signal_PointXYZRGBA = createSignal<signal_Kinect2_PointXYZRGBA>();


// 	Status result = STATUS_OK;
// 
// 	std::cout << "Initializing OpenNI...";
// 	result = OpenNI::initialize();
// 	std::cout << result << std::endl;
// 	if (result != STATUS_OK) return false;
// 
// 	Sleep(3000);
// 	openni::Array<openni::DeviceInfo> lists;
// 	while (lists.getSize() <= 0)
// 	{
// 		OpenNI::enumerateDevices(&lists);
// 		if (lists.getSize() <= 0)
// 		{
// 			std::cout << "No device found!" << std::endl;
// 			Sleep(1000);
// 		}
// 	}
// 	
// 	std::cout << "Connecting Kinect/Kinect2...";
// 	result = kinectDevice.open(openni::ANY_DEVICE);
// 	std::cout << OpenNI::getExtendedError() << std::endl;
// 	if (result != STATUS_OK) return false;
// 	
// 
// 	// depthstream and color stream
// 	std::cout << "Preparing depth sensor...";
// 	result = depthStream.create(kinectDevice, SENSOR_DEPTH);
// 	std::cout << result << std::endl;
// 	if (result != STATUS_OK) return false;
// 	max_depth = (unsigned short)depthStream.getMaxPixelValue();
// 
// 	if (kinectDevice.isImageRegistrationModeSupported(openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR))
// 	{
// 		std::cout << "Setting ImageRegistrationMode...";
// 		result = kinectDevice.setImageRegistrationMode(IMAGE_REGISTRATION_DEPTH_TO_COLOR);
// 		std::cout << result << std::endl;
// 		if (result != STATUS_OK) return false;
// 	}
// 
// 	std::cout << "Preparing color sensor...";
// 	result = colorStream.create(kinectDevice, SENSOR_COLOR);
// 	std::cout << result << std::endl;
// 	if (result != STATUS_OK) return false;
// 
// // 	std::cout << "depth: " << std::endl;
// // 	for (int i = 0; i < depthStream.getSensorInfo().getSupportedVideoModes().getSize(); i++)
// // 	{
// // 		std::cout << depthStream.getSensorInfo().getSupportedVideoModes()[i].getFps() << std::endl
// // 			<< depthStream.getSensorInfo().getSupportedVideoModes()[i].getPixelFormat() << std::endl
// // 			<< depthStream.getSensorInfo().getSupportedVideoModes()[i].getResolutionX() << "\t"
// // 			<< depthStream.getSensorInfo().getSupportedVideoModes()[i].getResolutionY() << std::endl << std::endl;
// // 	}
// // 
// 	std::cout << "color: " << std::endl;
// 	for (int i = 0; i < colorStream.getSensorInfo().getSupportedVideoModes().getSize(); i++)
// 	{
// 		std::cout << colorStream.getSensorInfo().getSupportedVideoModes()[i].getFps() << std::endl
// 			<< colorStream.getSensorInfo().getSupportedVideoModes()[i].getPixelFormat() << std::endl
// 			<< colorStream.getSensorInfo().getSupportedVideoModes()[i].getResolutionX() << "\t"
// 			<< colorStream.getSensorInfo().getSupportedVideoModes()[i].getResolutionY() << std::endl << std::endl;
// 	}
// 	// Intrinsic
// 	VideoMode videoModeDepth = depthStream.getVideoMode();
// 	float hFov = depthStream.getHorizontalFieldOfView();
// 	float vFov = depthStream.getVerticalFieldOfView();
// 
// 	intrDepth.rx = videoModeDepth.getResolutionX();
// 	intrDepth.ry = videoModeDepth.getResolutionY();
// 	intrDepth.fx = intrDepth.rx / 2 / tan(hFov / 2);
// 	intrDepth.fy = intrDepth.ry / 2 / tan(vFov / 2);
// 	intrDepth.cx = 0.5 * intrDepth.rx;
// 	intrDepth.cy = 0.5 * intrDepth.ry;
// 	switch (videoModeDepth.getPixelFormat())
// 	{
// 	case ONI_PIXEL_FORMAT_DEPTH_1_MM:
// 		intrDepth.zFactor = 1.0f;
// 		break;
// 	case ONI_PIXEL_FORMAT_DEPTH_100_UM:
// 		intrDepth.zFactor = 1000.0f;
// 		break;
// 	}
// 
// 	VideoMode videoModeColor = colorStream.getVideoMode();
// 	intrColor.rx = videoModeColor.getResolutionX();
// 	intrColor.ry = videoModeColor.getResolutionY();
// 	intrColor.fx = intrColor.rx / 2 / tan(hFov / 2);
// 	intrColor.fy = intrColor.ry / 2 / tan(vFov / 2);
// 	intrColor.cx = 0.5 * intrColor.rx;
// 	intrColor.cy = 0.5 * intrColor.ry;
// 
// 	return true;
}

void KinectReader::start()
{
	// Open Color Frame Reader
	HRESULT result = colorSource->OpenReader(&colorReader);
	if (FAILED(result)){
		throw std::exception("Exception : IColorFrameSource::OpenReader()");
	}

	// Open Depth Frame Reader
	result = depthSource->OpenReader(&depthReader);
	if (FAILED(result)){
		throw std::exception("Exception : IDepthFrameSource::OpenReader()");
	}

	// Open Infrared Frame Reader
	result = infraredSource->OpenReader(&infraredReader);
	if (FAILED(result)){
		throw std::exception("Exception : IInfraredFrameSource::OpenReader()");
	}

// 	running = true;
// 
// 	thread = boost::thread(&Kinect2Grabber::threadFunction, this);

// 	Status result;
// 
// 	result = depthStream.start();
// 	if (result != STATUS_OK) return false;
// 
// 	result = colorStream.start();
// 	if (result != STATUS_OK) return false;
//
//	return true;
}

void KinectReader::stop()
{
/*	boost::unique_lock<boost::mutex> lock(mutex);*/

/*	quit = true;*/
	running = false;
	if (sensor)
	{
		sensor->Close();
	}

/*	lock.unlock();*/

// 	depthStream.destroy();
// 	colorStream.destroy();
// 	kinectDevice.close();
}

bool KinectReader::isRunning()
{
/*	boost::unique_lock<boost::mutex> lock(mutex);*/
	return running;
/*	lock.unlock();*/
}

void KinectReader::shutdown()
{
	//OpenNI::shutdown();
}

// void KinectReader::onNewFrame(VideoStream &vs)
// {
// 	SensorType st = vs.getSensorInfo().getSensorType();
// 	if (st == SENSOR_DEPTH)
// 	{
// 		VideoFrameRef vfr;
// 		if (vs.readFrame(&vfr) == STATUS_OK)
// 		{
// 			cv::Mat d(vfr.getHeight(), vfr.getWidth(), CV_16UC1, (void *)vfr.getData());
// 			cv::Mat d2;
// 			d.copyTo(d2);
// 			depth_frames.push(d2);
// 		}
// 	}
// 	else if (st == SENSOR_COLOR)
// 	{
// 		VideoFrameRef vfr;
// 		if (colorStream.readFrame(&vfr) == STATUS_OK)
// 		{
// 			cv::Mat r(vfr.getHeight(), vfr.getWidth(), CV_8UC3, (void *)vfr.getData());
// 			cv::Mat r2;
// 			cv::cvtColor(r, r2, CV_RGB2BGR);
// 			color_frames.push(r2);
// 		}
// 	}
// }