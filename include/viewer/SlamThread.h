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

	void setParameters(int *parameters);
	void setShouldRegister(bool reg) { shouldRegister = reg; }
	SlamEngine* getEngine() { return engine; }
	void setEngine(SlamEngine* eng) { engine = eng; }

signals:
	void OneIterationDone(const cv::Mat &rgb, const cv::Mat &depth, const bool showPointCloud = false);
	void RegistrationDone();

protected:
	void run();

private:
	bool shouldStop;
	bool shouldRegister;

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