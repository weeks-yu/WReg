#ifndef SLAMTHREAD_H
#define SLAMTHREAD_H

#include "SlamEngine.h"
#include "RGBDReader.h"
#include <QThread>
#include <QTextStream>

class SlamThread : public QThread
{
	Q_OBJECT
public:
	typedef enum
	{
		SENSOR_IMAGE = 0,
		SENSOR_KINECT,
		SENSOR_ONI
	} SensorType;
public:
	SlamThread(SensorType st, const QString &dir, SlamEngine *eng, int *parameters);
	~SlamThread();

signals:
	void OneIterationDone(const cv::Mat &rgb, const cv::Mat &depth);
	void RegistrationDone();

protected:
	void run();

private:
	bool shouldStop;

	SensorType sensorType;

	//image or oni
	QString directory;
	int frameInterval;
	int frameStart;
	int frameStop;

	SlamEngine *engine;

	RGBDReader *reader;
};

#endif