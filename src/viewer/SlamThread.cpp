#include "SlamThread.h"
#include <QFile>
#include <QFileInfo>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <fstream>
#include "ImageReader.h"
#include "OniReader.h"
#include "KinectReader.h"
#include "RgbdDatas.h"

SlamThread::SlamThread(SensorType st, const QString &dir, SlamEngine *eng, int *parameters)
	// for SENSOR_IMAGE and SENSOR_ONI:
	//		parameters[0] is frameInterval
	//		parameters[1] is frameStart
	//		parameters[2] is frameStop
{
	shouldStop = false;
	directory = dir;
	engine = eng;
	frameInterval = 1;
	frameStart = -1;
	frameStop = -1;

	sensorType = st;
	Intrinsic intr;
	switch (sensorType)
	{
	case SENSOR_IMAGE:
		frameInterval = parameters[0];
		frameStart = parameters[1];
		frameStop = parameters[2];

		reader = new ImageReader();
		intr.rx = Config::instance()->get<int>("image_width");
		intr.ry = Config::instance()->get<int>("image_height");
		intr.cx = Config::instance()->get<float>("camera_cx");
		intr.cy = Config::instance()->get<float>("camera_cy");
		intr.fx = Config::instance()->get<float>("camera_fx");
		intr.fy = Config::instance()->get<float>("camera_fy");
		intr.zFactor = Config::instance()->get<float>("depth_factor");
		reader->setIntrinsicDepth(intr);
		reader->create(dir.toStdString().c_str());
		reader->start();
		break;
	case SENSOR_ONI:
		frameInterval = parameters[0];
		frameStart = parameters[1];
		frameStop = parameters[2];

		reader = new OniReader();
		reader->create(dir.toStdString().c_str());
		reader->start();
		break;
	case SENSOR_KINECT:
		frameInterval = 1;
		frameStart = -1;
		frameStop = -1;

		reader = new KinectReader();
		reader->create(NULL);
		reader->start();
		break;
	}

// 	engine->setFrameInterval(frameInterval);
// 	engine->setFrameStart(frameStart);
// 	engine->setFrameStop(frameStop);
}

SlamThread::~SlamThread()
{
	engine = nullptr;
}

void SlamThread::setParameters(int *parameters)
{
	switch (sensorType)
	{
	case SENSOR_IMAGE:
		frameInterval = parameters[0];
		frameStart = parameters[1];
		frameStop = parameters[2];
		break;
	case SENSOR_ONI:
		frameInterval = parameters[0];
		frameStart = parameters[1];
		frameStop = parameters[2];
		break;
	case SENSOR_KINECT:
		frameInterval = 1;
		frameStart = -1;
		frameStop = -1;
		break;
	}
}

void SlamThread::run()
{
	emit InitDone(reader->intrColor, reader->intrDepth);

	int k = 0;
	int id = 0;
	while (!shouldStop)
	{
		cv::Mat r, d;
		double ts;
		if (!reader->getNextFrame(r, d, ts))
		{
			break;
		}

		if (shouldRegister)
		{
			if (frameStart > -1 && k < frameStart)
			{
				k++;
				continue;
			}

			if (frameStop > -1 && k > frameStop)
				break;

			if ((k - frameStart > -1 ? frameStart : 0) % frameInterval != 0)
			{
				k++;
				continue;
			}
		}

		switch (sensorType)
		{
		case SENSOR_KINECT:
			reader->registerColorToDepth(r, d, r);
			break;
		case SENSOR_IMAGE:
			break;
		case SENSOR_ONI:
			break;
		}

		RGBDDatas::push(r, d, ts);
		Eigen::Matrix4f tran = Eigen::Matrix4f::Identity();
		if (shouldRegister)
		{
			tran = engine->RegisterNext(r, d, ts);
			k++;
		}
		emit OneIterationDone(id, tran);
		id++;
	}
	
	reader->stop();
	engine->ShowStatistics();
	//engine->SaveTestInfo();
	emit RegistrationDone();
}