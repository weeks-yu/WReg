#ifndef SLAMTHREAD_H
#define SLAMTHREAD_H

#include <QThread>
#include <QTextStream>
#include "SlamEngine.h"

class SlamThread : public QThread
{
	Q_OBJECT
public:
	SlamThread(const QString &dir, SlamEngine *eng, int interval = 1, int start = -1, int stop = -1);
	~SlamThread();

signals:
	void OneIterationDone(const cv::Mat &rgb, const cv::Mat &depth);
	void RegistrationDone();

protected:
	void run();

private:
	bool shouldStop;
	QString directory;
	ifstream *fileInput;
	int frameInterval;
	int frameStart;
	int frameStop;
	SlamEngine *engine;
};

#endif