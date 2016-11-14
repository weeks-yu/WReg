#include "SlamThread.h"

#include "MainWindow.h"
#include "ui_MainWindow.h"
#include "ui_DockRegistration.h"
#include "PclViewer.h"
#include "Parser.h"
#include "Transformation.h"

#include <QFileDialog>
#include <QMessageBox>

#include <pcl/io/pcd_io.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
	uiDockRegistration = nullptr;
	dockRegistration = nullptr;

	mdiArea = new QMdiArea(this);
	this->setCentralWidget(mdiArea);
	registrationViewer = nullptr;

	engine = nullptr;
	thread = nullptr;
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::ShowPointCloudFiles(const QString &filename)
{
	QFileInfo fi(filename);
	if (!fi.isFile())
	{
		return;
	}
	PointCloudT::Ptr cloud(new PointCloudT);

	// load point cloud file
	if (fi.suffix() == "pcd")
	{
		if (pcl::io::loadPCDFile(filename.toStdString(), *cloud) == -1)
		{
			// error while loading pcd file
			return;
		}
	}

	QMdiSubWindow *subWindow = new QMdiSubWindow(mdiArea);
	PclViewer *pclViewer = new PclViewer(cloud, subWindow);
	subWindow->setWidget(pclViewer);
	subWindow->setAttribute(Qt::WA_DeleteOnClose);
	mdiArea->addSubWindow(subWindow);
	subWindow->showMaximized();
}

// void MainWindow::ShowBenchmarkTest(const QString &directory)
// {
// 	// left dock
// 	if (dockBenchmark == nullptr)
// 	{
// 		dockBenchmark = new QDockWidget(this);
// 		if (uiDockBenchmark == nullptr)
// 			uiDockBenchmark = new Ui::DockBenchmark;
// 		uiDockBenchmark->setupUi(dockBenchmark);
// 	}
// 	uiDockBenchmark->lineEditDirectory->setText(directory);
// 	uiDockBenchmark->pushButtonSaveTrajectories->setDisabled(true);
// 	this->addDockWidget(Qt::LeftDockWidgetArea, dockBenchmark);
// 	dockBenchmark->show();
// 
// 	connect(uiDockBenchmark->pushButtonDirectory, &QPushButton::clicked, this, &MainWindow::onBenchmarkPushButtonDirectoryClicked);
// 	connect(uiDockBenchmark->pushButtonRun, &QPushButton::clicked, this, &MainWindow::onBenchmarkPushButtonRunClicked);
// 	connect(uiDockBenchmark->pushButtonSaveKeyframes, &QPushButton::clicked, this, &MainWindow::onBenchmarkPushButtonSaveKeyframesClicked);
// 	connect(uiDockBenchmark->pushButtonSaveTrajectories, &QPushButton::clicked, this, &MainWindow::onBenchmarkPushButtonSaveTrajectoriesClicked);
// 	connect(uiDockBenchmark->pushButtonSaveLogs, &QPushButton::clicked, this, &MainWindow::onBenchmarkPushButtonSaveLogsClicked);
// 
// 	// center
// 	registrationViewer = new RegistrationViewer(this);
// 	QMdiSubWindow *subWindow = new QMdiSubWindow(mdiArea);
// 	subWindow->setWidget(registrationViewer);
// 	subWindow->setAttribute(Qt::WA_DeleteOnClose);
// 	mdiArea->addSubWindow(subWindow);
// 	subWindow->showMaximized();
// }
// 
// void MainWindow::ShowBenchmarkResult(const QString &filename, int fi, int fst, int fed)
// {
// 	if (engine != nullptr)
// 		delete engine;
// 	engine = new SlamEngine();
// 	engine->setUsingIcpcuda(false);
// 	engine->setUsingRobustOptimizer(false);
// 
// 	string directory;
// 	ifstream input(filename.toStdString());
// 	getline(input, directory);
// 
// 	stringstream ss;
// 	ss << directory << "/read.txt";
// 	ifstream fileInput(ss.str());
// 
// 	int k = 0;
// 	string line;
// 	getline(input, line);
// 	while (getline(fileInput, line))
// 	{
// 		if (fst > -1 && k < fst)
// 		{
// 			k++;
// 			continue;
// 		}
// 
// 		if (fed > -1 && k > fed)
// 			break;
// 
// 		if ((k - fst > -1 ? fst : 0) % fi != 0)
// 		{
// 			k++;
// 			continue;
// 		}
// 
// 		double timestamp;
// 		Eigen::Vector3f t;
// 		Eigen::Quaternionf q;
// 		input >> timestamp >> t(0) >> t(1) >> t(2) >> q.x() >> q.y() >> q.z() >> q.w();
// 
// 		QStringList lists = QString(line.data()).split(' ');
// 
// 		QString timestamp_string = lists[1].mid(6, lists[1].length() - 10);
// 		double timestamp_this = timestamp_string.toDouble();
// 
// 		if (!(fabs(timestamp_this - timestamp) < 1e-4))
// 		{
// 			if (timestamp_this < timestamp)
// 			{
// 				continue;
// 			}
// 			while (!(fabs(timestamp_this - timestamp) < 1e-4) &&
// 				timestamp_this > timestamp && !input.eof())
// 			{
// 				input >> timestamp >> t(0) >> t(1) >> t(2) >> q.x() >> q.y() >> q.z() >> q.w();
// 			}
// 		}
// 
// 		Eigen::Matrix4f tran = transformationFromQuaternionsAndTranslation(q, t);
// 
// 		ss.str("");
// 		ss << directory << "/" << lists[0].toStdString();
// 		cv::Mat rgb = cv::imread(ss.str());
// 		ss.str("");
// 		ss << directory << "/" << lists[1].toStdString();
// 		cv::Mat depth = cv::imread(ss.str(), -1);
// 
// 		engine->AddNext(rgb, depth, timestamp, tran);
// 		k++;
// 	}
// 	fileInput.close();
// 	input.close();
// 
// 	registrationViewer->ShowPointCloud(engine->GetScene());
// }

void MainWindow::ShowRegistration()
{
	// left dock
	if (dockRegistration == nullptr)
	{
		dockRegistration = new QDockWidget(this);
		if (uiDockRegistration == nullptr)
			uiDockRegistration = new Ui::DockRegistration;
		uiDockRegistration->setupUi(dockRegistration);
	}

	uiDockRegistration->pushButtonSaveTrajectories->setDisabled(true);
	this->addDockWidget(Qt::LeftDockWidgetArea, dockRegistration);
	dockRegistration->show();

	onRegistrationComboBoxSensorTypeCurrentIndexChanged(0);

	connect(uiDockRegistration->pushButtonDirectory, &QPushButton::clicked, this, &MainWindow::onRegistrationPushButtonDirectoryClicked);
	connect(uiDockRegistration->pushButtonRun, &QPushButton::clicked, this, &MainWindow::onRegistrationPushButtonRunClicked);
	connect(uiDockRegistration->pushButtonSaveKeyframes, &QPushButton::clicked, this, &MainWindow::onRegistrationPushButtonSaveKeyframesClicked);
	connect(uiDockRegistration->pushButtonSaveTrajectories, &QPushButton::clicked, this, &MainWindow::onRegistrationPushButtonSaveTrajectoriesClicked);
	connect(uiDockRegistration->pushButtonSaveLogs, &QPushButton::clicked, this, &MainWindow::onRegistrationPushButtonSaveLogsClicked);
	connect(uiDockRegistration->comboBoxSensorType,
		static_cast<void (QComboBox::*)(int index)>(&QComboBox::currentIndexChanged),
		this,
		&MainWindow::onRegistrationComboBoxSensorTypeCurrentIndexChanged);
	connect(uiDockRegistration->pushButtonConnectKinect, &QPushButton::clicked, this, &MainWindow::onRegistrationPushButtonConnectKinectClicked);

	// center
	registrationViewer = new RegistrationViewer(this);
	QMdiSubWindow *subWindow = new QMdiSubWindow(mdiArea);
	subWindow->setWidget(registrationViewer);
	subWindow->setAttribute(Qt::WA_DeleteOnClose);
	mdiArea->addSubWindow(subWindow);
	subWindow->showMaximized();
}

// slots
void MainWindow::onActionOpenTriggered()
{
	QString filename = QFileDialog::getOpenFileName(this, tr("Open PCD File"), "", tr("pcd files(*.pcd)"));
	if (!filename.isNull())
	{
		ShowPointCloudFiles(filename);
	}
}

void MainWindow::onActionSaveTriggered()
{
	QString filename = QFileDialog::getSaveFileName(this, tr("Save pcd fild"), "", tr("pcd files(*.pcd)"));
	if (!filename.isNull())
	{
		QWidget *w = mdiArea->currentSubWindow()->widget();
		QString s = w->metaObject()->className();
		if (s == "BenchmarkViewer")
		{
			RegistrationViewer *bv = (RegistrationViewer *)w;
			pcl::io::savePCDFile(filename.toStdString(), *(bv->GetPointCloud()), true);
		}
		else if (s == "PclViewer")
		{
			PclViewer *pv = (PclViewer *)w;
			pcl::io::savePCDFile(filename.toStdString(), *(pv->GetPointCloud()), true);
		}
	}
}

// void MainWindow::onActionOpenReadTxtTriggered()
// {
// 	QString directory = QFileDialog::getExistingDirectory(this, tr("Open Benchmark Directory"), "");
// 	if (!directory.isNull())
// 	{
// 		QFileInfo fi(directory);
// 		if (!fi.isDir())
// 		{
// 			QMessageBox::warning(this, tr("Invalid directory"), tr("Please check whether the directory is valid."));
// 			return;
// 		}
// 		QFileInfo fi2(fi.absoluteFilePath() + "/read.txt");
// 		if (!fi2.isFile())
// 		{
// 			QMessageBox::warning(this, tr("Read.txt not found"), tr("Please check whether \"read.txt\" exists or not"));
// 			return;
// 		}
// 		ShowBenchmarkTest(directory);
// 	}
// }
// 
// void MainWindow::onActionShowResultFromFileTriggered()
// {
// 	QString filename = QFileDialog::getOpenFileName(this, tr("Open Result File"), "", tr("all files(*.*)"));
// 	if (!filename.isNull())
// 	{
// 		QFileInfo fi(filename);
// 		if (!fi.isFile())
// 		{
// 			QMessageBox::warning(this, tr("File not found"), tr("Please check whether selected file exists or not"));
// 			return;
// 		}
// 		ifstream input(filename.toStdString());
// 		string directory, line;
// 		getline(input, directory);
// 		getline(input, line);
// 		QStringList lists = QString(line.data()).split(' ');
// 		int fit, fst, fed;
// 		fit = lists[0].toInt();
// 		fst = lists[1].toInt();
// 		fed = lists[2].toInt();
// 
// 		ShowBenchmarkTest(QString(directory.data()));
// 		ShowBenchmarkResult(filename, fit, fst, fed);
// 	}
// }

void MainWindow::onActionRegistrationTriggered()
{
	ShowRegistration();
}

void MainWindow::onRegistrationPushButtonRunClicked(bool checked)
{
	// sensor type
	// 0 - image
	// 1 - ONI
	// 2 - Kinect
	SlamThread::SensorType sensorType = static_cast<SlamThread::SensorType>(uiDockRegistration->comboBoxSensorType->currentIndex());
	int method = uiDockRegistration->comboBoxMethod->currentIndex();
	bool usingICPCUDA = (method == 0 || method == 1) && (sensorType == 0 || sensorType == 1);
	bool usingFeature = (method == 2 || method == 3) && (sensorType == 0 || sensorType == 1);
	bool usingRobustOptimizer = (method == 1 || method == 3) && (sensorType == 0 || sensorType == 1);

	if (!Parser::isFloat(uiDockRegistration->lineEditICPDist->text()))
	{
		QMessageBox::warning(this, "Parameter Error", "Group ICP: Parameter \"Dist Threshold\" should be float.");
		return;
	}
	if (!Parser::isFloat(uiDockRegistration->lineEditICPAngle->text()))
	{
		QMessageBox::warning(this, "Parameter Error", "Group ICP: Parameter \"Angle Threshold\" should be float.");
		return;
	}
	if (!Parser::isFloat(uiDockRegistration->lineEditFeatureInlierPercentage->text()))
	{
		QMessageBox::warning(this, "Parameter Error", "Group Feature: Parameter \"Inlier Percentage\" should be float.");
		return;
	}
	if (!Parser::isFloat(uiDockRegistration->lineEditFeatureInlierDist->text()))
	{
		QMessageBox::warning(this, "Parameter Error", "Group Feature: Parameter \"Inlier Dist\" should be float.");
		return;
	}
	if (!Parser::isFloat(uiDockRegistration->lineEditGraphInlierDist->text()))
	{
		QMessageBox::warning(this, "Parameter Error", "Group Feature: Parameter \"Graph Inlier Dist\" should be float.");
		return;
	}

	int *p = nullptr;

	switch (sensorType)
	{
	case SlamThread::SENSOR_IMAGE:
		p = new int[3];
		p[0] = uiDockRegistration->spinBoxFrameInterval->value();
		p[1] = uiDockRegistration->spinBoxStartFrame->value();
		p[2] = uiDockRegistration->spinBoxStopFrame->value();
		break;
	case SlamThread::SENSOR_ONI:
		p = new int[3];
		p[0] = uiDockRegistration->spinBoxFrameInterval->value();
		p[1] = uiDockRegistration->spinBoxStartFrame->value();
		p[2] = uiDockRegistration->spinBoxStopFrame->value();
		break;
	case SlamThread::SENSOR_KINECT:
		break;
	}

	// run benchmark test
	if (thread != nullptr)
	{
		engine = thread->getEngine();
		if (engine == nullptr)
		{
			engine = new SlamEngine();
			thread->setEngine(engine);
		}
		thread->setParameters(p);
	}
	else
	{
		engine = new SlamEngine();
		thread = new SlamThread(sensorType, uiDockRegistration->lineEditDirectory->text(), engine, p);
		qRegisterMetaType<cv::Mat>("cv::Mat");
		connect(thread, &SlamThread::OneIterationDone, this, &MainWindow::onBenchmarkOneIterationDone);
		connect(thread, &SlamThread::RegistrationDone, this, &MainWindow::onBenchmarkRegistrationDone);
	}

	engine->setUsingIcpcuda(usingICPCUDA);
	engine->setUsingFeature(usingFeature);
	if (usingFeature)
	{
		QString type = uiDockRegistration->comboBoxFeatureType->currentText();
		engine->setFeatureType(type.toStdString());
		engine->setFeatureMinMatches(uiDockRegistration->spinBoxFeatureMinMatches->text().toInt());
		engine->setFeatureInlierPercentage(uiDockRegistration->lineEditFeatureInlierPercentage->text().toFloat());
		engine->setFeatureInlierDist(uiDockRegistration->lineEditFeatureInlierDist->text().toFloat());
	}
	engine->setUsingRobustOptimizer(usingRobustOptimizer);
	if (usingRobustOptimizer)
	{
		QString type = uiDockRegistration->comboBoxGraphFeatureType->currentText();
		engine->setGraphFeatureType(type.toStdString());
		engine->setGraphMinMatches(uiDockRegistration->spinBoxGraphMinMatches->text().toInt());
		engine->setGraphInlierPercentage(uiDockRegistration->lineEditGraphInlierPercentage->text().toFloat());
		engine->setGraphInlierDist(uiDockRegistration->lineEditGraphInlierDist->text().toFloat());
	}

	thread->setShouldRegister(true);
	if (!thread->isRunning())
	{
		thread->start();
	}
	uiDockRegistration->pushButtonRun->setDisabled(true);
	uiDockRegistration->pushButtonSaveTrajectories->setDisabled(true);
}

void MainWindow::onRegistrationPushButtonDirectoryClicked(bool checked)
{
	QString directory = QFileDialog::getExistingDirectory(this, tr("Open Benchmark Directory"), "");
	if (!directory.isNull())
	{
		QFileInfo fi(directory);
		if (!fi.isDir())
		{
			QMessageBox::warning(this, tr("Invalid directory"), tr("Please check whether the directory is valid."));
			return;
		}
		QFileInfo fi2(fi.absoluteFilePath() + "/read.txt");
		if (!fi2.isFile())
		{
			QMessageBox::warning(this, tr("Read.txt not found"), tr("Please check whether \"read.txt\" exists or not"));
			return;
		}
		uiDockRegistration->lineEditDirectory->setText(directory);
	}
}

void MainWindow::onRegistrationPushButtonSaveTrajectoriesClicked(bool checked)
{
	QString filename = QFileDialog::getSaveFileName(this, tr("Save Transformations"), "", tr("txt files(*.txt)"));
	if (!filename.isNull())
	{
		if (!engine)
			return;

		int fi = engine->getFrameInterval();
		int fst = engine->getFrameStart();
		int fed = engine->getFrameStop();
		vector<pair<double, Eigen::Matrix4f>> trans = engine->GetTransformations();
		ofstream outfile(filename.toStdString());
		outfile << uiDockRegistration->lineEditDirectory->text().toStdString() << endl;
		outfile << fi << ' ' << fst << ' ' << fed << endl;
		for (int i = 0; i < trans.size(); i++)
		{
			Eigen::Vector3f t = TranslationFromMatrix4f(trans[i].second);
			Eigen::Quaternionf q = QuaternionFromMatrix4f(trans[i].second);

			outfile << fixed << setprecision(6) << trans[i].first
				<< ' ' << t(0) << ' ' << t(1) << ' ' << t(2)
				<< ' ' << q.x() << ' ' << q.y() << ' ' << q.z() << ' ' << q.w() << endl;
		}
		outfile.close();
	}
}

void MainWindow::onRegistrationPushButtonSaveKeyframesClicked(bool checked)
{
	QString directory = QFileDialog::getExistingDirectory(this, tr("Select Directory"), "");
	if (!directory.isNull())
	{
		QFileInfo fi(directory);
		if (!fi.isDir())
		{
			QMessageBox::warning(this, tr("Invalid directory"), tr("Please check whether the directory is valid."));
			return;
		}

		if (!engine)
			return;

#ifdef SAVE_TEST_INFOS
		cout << engine->keyframe_candidates.size() << endl;
		for (int i = 0; i < engine->keyframe_candidates.size(); i++)
		{
			QString fn = fi.absoluteFilePath() + "/keyframe_candidate_" + QString::number(engine->keyframe_candidates_id[i]) + "_rgb.png";
			cv::imwrite(fn.toStdString(), engine->keyframe_candidates[i].first);
		}

		cout << engine->keyframes.size() << endl;
		for (int i = 0; i < engine->keyframes.size(); i++)
		{
			QString inliers_sig = QString::fromStdString(engine->keyframes_inliers_sig[i]);
			QString exists_sig = QString::fromStdString(engine->keyframes_exists_sig[i]);
			QString fn = fi.absoluteFilePath() + "/keyframe_" + QString::number(engine->keyframes_id[i]) + "_rgb_" +
				inliers_sig + "_" + exists_sig + ".png";
			cv::imwrite(fn.toStdString(), engine->keyframes[i].first);
		}
#endif

	}
}

void MainWindow::onRegistrationPushButtonSaveLogsClicked(bool checked)
{
	QString filename = QFileDialog::getSaveFileName(this, tr("Save Transformations"), "", tr("txt files(*.txt)"));
	if (!filename.isNull())
	{
		
		ofstream outfile(filename.toStdString());
		if (engine)
		{
			engine->SaveLogs(outfile);
		}
		outfile.close();
	}
}

void MainWindow::onRegistrationComboBoxSensorTypeCurrentIndexChanged(int index)
{
	if (index == 0 || index == 1)
	{
		uiDockRegistration->groupBoxImageOni->setEnabled(true);
		uiDockRegistration->groupBoxImageOni->setVisible(true);
		uiDockRegistration->groupBoxKinect->setEnabled(false);
		uiDockRegistration->groupBoxKinect->setVisible(false);
		uiDockRegistration->groupBoxMethodImageOni->setEnabled(true);
		uiDockRegistration->groupBoxMethodImageOni->setVisible(true);
		uiDockRegistration->groupBoxMethodKinect->setEnabled(false);
		uiDockRegistration->groupBoxMethodKinect->setVisible(false);
	}
	else if (index == 2)
	{
		uiDockRegistration->groupBoxImageOni->setEnabled(false);
		uiDockRegistration->groupBoxImageOni->setVisible(false);
		uiDockRegistration->groupBoxKinect->setEnabled(true);
		uiDockRegistration->groupBoxKinect->setVisible(true);
		uiDockRegistration->groupBoxMethodImageOni->setEnabled(false);
		uiDockRegistration->groupBoxMethodImageOni->setVisible(false);
		uiDockRegistration->groupBoxMethodKinect->setEnabled(true);
		uiDockRegistration->groupBoxMethodKinect->setVisible(true);
	}
}

void MainWindow::onRegistrationPushButtonConnectKinectClicked(bool checked)
{

}

void MainWindow::onBenchmarkOneIterationDone(const cv::Mat &rgb, const cv::Mat &depth, bool showPointCloud)
{
	cv::Mat tempRgb, tempDepth;
	QImage *rgbImage, *depthImage;
	cv::cvtColor(rgb, tempRgb, CV_BGR2RGB);
	rgbImage = new QImage((const unsigned char*)(tempRgb.data),
		tempRgb.cols, tempRgb.rows,
		tempRgb.cols * tempRgb.channels(),
		QImage::Format_RGB888);
	registrationViewer->ShowRGBImage(rgbImage);

	depth *= 255.0 / 65535.0;
	depth.convertTo(tempDepth, CV_8U);
	cv::cvtColor(tempDepth, tempDepth, CV_GRAY2RGB);
	depthImage = new QImage((const unsigned char*)(tempDepth.data),
		tempDepth.cols, tempDepth.rows,
		tempDepth.cols * tempDepth.channels(),
		QImage::Format_RGB888);
	registrationViewer->ShowDepthImage(depthImage);

	if (showPointCloud)
	{
		registrationViewer->ShowPointCloud(engine->GetScene());
	}
}

void MainWindow::onBenchmarkRegistrationDone()
{
	registrationViewer->ShowPointCloud(engine->GetScene());
	uiDockRegistration->pushButtonRun->setDisabled(false);
	uiDockRegistration->pushButtonSaveTrajectories->setDisabled(false);
}