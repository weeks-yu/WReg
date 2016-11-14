#include "SlamThread.h"
#include <QFile>
#include <QFileInfo>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <fstream>
#include "ImageReader.h"
#include "OniReader.h"
#include "KinectReader.h"

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
	switch (sensorType)
	{
	case SENSOR_IMAGE:
		frameInterval = parameters[0];
		frameStart = parameters[1];
		frameStop = parameters[2];

		reader = new ImageReader();
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

	engine->setFrameInterval(frameInterval);
	engine->setFrameStart(frameStart);
	engine->setFrameStop(frameStop);
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
	int k = 0;
	while (!shouldStop)
	{
		cv::Mat r, d;
		if (!reader->getNextFrame(r, d))
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

		if (shouldRegister)
		{
			engine->RegisterNext(r, d, k);
			k++;
		}
		emit OneIterationDone(r, d);
	}
	
	reader->stop();
	engine->ShowStatistics();
	//engine->SaveTestInfo();
	emit RegistrationDone();
}