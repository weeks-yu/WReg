#ifndef SLAMTHREAD_H
#define SLAMTHREAD_H

#include "SlamEngine.h"
#include "RGBDReader.h"
#include "TsdfModel.h"
#include <QThread>
#include <QTextStream>

class SlamThread : public QThread
{
	Q_OBJECT
public:
	typedef enum
	{
		SENSOR_IMAGE = 0,
		SENSOR_ONI,
		SENSOR_KINECT
	} SensorType;
public:
	SlamThread(SensorType st, const QString &dir, SlamEngine *eng, int *parameters);
	~SlamThread();

	void setParameters(int *parameters); 
	void setShouldRegister(bool reg) { shouldRegister = reg; }
	SlamEngine* getEngine() { return engine; }
	void setEngine(SlamEngine* eng) { engine = eng; }

signals:
	void InitDone(const Intrinsic &intrColor, const Intrinsic &intrDepth);
	void OneIterationDone(int id, const Eigen::Matrix4f &tran);
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
	TsdfModel *tsdf;

	RGBDReader *reader;
};

#endif