#pragma once;

#include "RGBDReader.h"
#include <Kinect.h>
/*#include <OpenNI.h>*/
#include <boost/thread/thread.hpp>

/*using namespace openni;*/

class KinectReader : public RGBDReader/*, public VideoStream::NewFrameListener*/
{
public:
	KinectReader();
	virtual ~KinectReader();

// 	virtual bool getNextColorFrame(cv::Mat &rgb);
// 	virtual bool getNextDepthFrame(cv::Mat &depth);
	virtual bool getNextFrame(cv::Mat &rgb, cv::Mat &depth, double &timestamp);
	virtual void registerColorToDepth(const cv::Mat &rgb, const cv::Mat &depth, cv::Mat &rgbRegistered);
	virtual void registerDepthToColor(const cv::Mat &rgb, const cv::Mat &depth, cv::Mat &depthRegistered);

	virtual bool create(const char* mode = NULL);
	virtual void start();
	virtual void stop();
//	virtual void onNewFrame(VideoStream &vs);
	virtual bool isRunning();
//	virtual void threadFunction();

	static void shutdown();

protected:
	template<class Interface>
	inline void SafeRelease(Interface *& IRealese)
	{
		if (IRealese != NULL)
		{
			IRealese->Release();
			IRealese = NULL;
		}
	}

protected:
// 	Device kinectDevice;
// 	VideoStream depthStream;
// 	VideoStream colorStream;

	IKinectSensor *sensor;
	ICoordinateMapper *mapper;
	IColorFrameSource *colorSource;
	IColorFrameReader *colorReader;
	IDepthFrameSource *depthSource;
	IDepthFrameReader *depthReader;
	IInfraredFrameSource *infraredSource;
	IInfraredFrameReader *infraredReader;
// 	std::vector<RGBQUAD> colorBuffer;
// 	std::vector<UINT16> depthBuffer;
// 	std::vector<UINT16> infraredBuffer;

	Intrinsic intrInfrared;
 	bool running;
// 	bool quit;

// 	boost::thread thread;
// 	mutable boost::mutex mutex;
};