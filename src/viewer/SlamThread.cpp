#include "SlamThread.h"
#include <QFile>
#include <QFileInfo>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <fstream>

SlamThread::SlamThread(const QString &dir, SlamEngine *eng, int interval /* = 1 */, int start /* = -1 */, int stop /* = -1 */)
{
	directory = dir;
	shouldStop = false;
	fileInput = new ifstream((directory + "/read.txt").toStdString());
	engine = eng;
	frameInterval = interval;
	frameStart = start;
	frameStop = stop;
}

SlamThread::~SlamThread()
{
	engine = nullptr;
}

void SlamThread::run()
{
	int k = 0;
	string line;
	while (!shouldStop && fileInput != nullptr && getline(*fileInput, line))
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

		QStringList lists = QString(line.data()).split(' ');
		cv::Mat rgb = cv::imread((directory + "/" + lists[0]).toStdString());
		cv::Mat depth = cv::imread((directory + "/" + lists[1]).toStdString(), -1);

		QFileInfo fi(lists[0]);
		QString tname = fi.completeBaseName();
		double ttime = tname.toDouble();
		engine->RegisterNext(rgb, depth, fi.completeBaseName().toDouble());
		emit OneIterationDone(rgb, depth);
		k++;
	}
	if (fileInput)
		fileInput->close();
	emit RegistrationDone();
}