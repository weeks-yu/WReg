#ifndef SLAMTHREAD_H
#define SLAMTHREAD_H

#include <QThread>
#include <QTextStream>
#include "SlamEngine.h"

class SlamThread : public QThread
{
	Q_OBJECT
public:
	SlamThread(const QString &dir, SlamEngine *eng);
	~SlamThread();

signals:
	void OneIterationDone(const cv::Mat &rgb, const cv::Mat &depth);

protected:
	void run();

private:
	bool shouldStop;
	QString directory;
	ifstream *fileInput;
	SlamEngine *engine;
};

#endif