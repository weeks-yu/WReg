#include "SlamThread.h"
#include <QFile>
#include <QFileInfo>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <fstream>

SlamThread::SlamThread(const QString &dir, SlamEngine *eng)
{
	directory = dir;
	shouldStop = false;

	QFile file();
	fileInput = new ifstream((directory + "/read.txt").toStdString());

	engine = eng;
}

SlamThread::~SlamThread()
{
	engine = nullptr;
}

void SlamThread::run()
{
	std::string line;
	while (!shouldStop && fileInput != nullptr && getline(*fileInput, line))
	{
		QStringList lists = QString(line.data()).split(' ');
		cv::Mat rgb = cv::imread((directory + "/" + lists[0]).toStdString());
		cv::Mat depth = cv::imread((directory + "/" + lists[1]).toStdString(), -1);

		QFileInfo fi(lists[0]);
		engine->RegisterNext(rgb, depth, fi.baseName().toDouble());
		emit OneIterationDone(rgb, depth);
	}
}