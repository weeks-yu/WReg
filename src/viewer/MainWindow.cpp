#include "MainWindow.h"
#include "ui_MainWindow.h"
#include "ui_DockBenchmark.h"
#include "PclViewer.h"
#include "Parser.h"
#include "SlamThread.h"
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
	uiDockBenchmark = nullptr;
	dockBenchmark = nullptr;

	mdiArea = new QMdiArea(this);
	this->setCentralWidget(mdiArea);
	benchmarkViewer = nullptr;

	engine = nullptr;
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

void MainWindow::ShowBenchmarkTest(const QString &directory)
{
	// left dock
	if (dockBenchmark == nullptr)
	{
		dockBenchmark = new QDockWidget(this);
		if (uiDockBenchmark == nullptr)
			uiDockBenchmark = new Ui::DockBenchmark;
		uiDockBenchmark->setupUi(dockBenchmark);
	}
	uiDockBenchmark->lineEditDirectory->setText(directory);
	uiDockBenchmark->pushButtonSave->setDisabled(true);
	this->addDockWidget(Qt::LeftDockWidgetArea, dockBenchmark);

	connect(uiDockBenchmark->pushButtonRun, &QPushButton::clicked, this, &MainWindow::onBenchmarkPushButtonRunClicked);
	connect(uiDockBenchmark->pushButtonDirectory, &QPushButton::clicked, this, &MainWindow::onBenchmarkPushButtonDirectoryClicked);
	connect(uiDockBenchmark->pushButtonSave, &QPushButton::clicked, this, &MainWindow::onBenchmarkPushButtonSaveClicked);
	connect(uiDockBenchmark->pushButtonSaveKeyframes, &QPushButton::clicked, this, &MainWindow::onBenchmarkPushButtonSaveKeyframesClicked);

	// center
	benchmarkViewer = new BenchmarkViewer(this);
	QMdiSubWindow *subWindow = new QMdiSubWindow(mdiArea);
	subWindow->setWidget(benchmarkViewer);
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

}

void MainWindow::onActionOpenReadTxtTriggered()
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
		ShowBenchmarkTest(directory);
	}
}

void MainWindow::onBenchmarkPushButtonRunClicked(bool checked)
{
	if (!Parser::isDouble(uiDockBenchmark->lineEditDownsampleRate->text()))
	{
		QMessageBox::warning(this, "Parameter Error", "Downsampling: parameter \"Downsample Rate\" should be double.");
		return;
	}
	if (!Parser::isDouble(uiDockBenchmark->lineEditGICPCorrDist->text()))
	{
		QMessageBox::warning(this, "Parameter Error", "GICP: parameter \"Max Correspondence Dist\" should be double.");
		return;
	}
	if (!Parser::isDouble(uiDockBenchmark->lineEditGICPEpsilon->text()))
	{
		QMessageBox::warning(this, "Parameter Error", "GICP: parameter \"Transformation Epsilon\" should be double.");
		return;
	}

	// run benchmark test
	if (engine != nullptr)
		delete engine;
	engine = new SlamEngine();
	int method = uiDockBenchmark->comboBoxMethod->currentIndex();
	bool usingGICP = method == 0 || method == 1;
	bool usingICPCUDA = method == 2 || method == 3;
	bool usingGraphOptimizer = method == 0 || method == 2;
	bool usingDownSampling = usingGICP && uiDockBenchmark->checkBoxDownSampling->isChecked();
	
	engine->setUsingDownsampling(usingDownSampling);
	if (usingDownSampling)
	{
		engine->setDownsampleRate(uiDockBenchmark->lineEditDownsampleRate->text().toDouble());
	}
	engine->setUsingGicp(usingGICP);
	if (usingGICP)
	{
		engine->setGicpMaxIterations(uiDockBenchmark->spinBoxGICPIterations->text().toDouble());
		engine->setGicpMaxCorrDist(uiDockBenchmark->lineEditGICPCorrDist->text().toDouble());
		engine->setGicpEpsilon(uiDockBenchmark->lineEditGICPEpsilon->text().toDouble());
	}
	engine->setUsingIcpcuda(usingICPCUDA);
	if (usingICPCUDA)
	{

	}
	engine->setUsingGraphOptimizer(usingGraphOptimizer);
	if (usingGraphOptimizer)
	{
		QString type = uiDockBenchmark->comboBoxGraphFeatureType->currentText();
		engine->setGraphFeatureType(type == "SIFT" ? SlamEngine::SIFT : SlamEngine::SURF);
	}

	SlamThread *thread = new SlamThread(uiDockBenchmark->lineEditDirectory->text(), engine,
		uiDockBenchmark->spinBoxFrameInterval->value(),
		uiDockBenchmark->spinBoxStartFrame->value(),
		uiDockBenchmark->spinBoxStopFrame->value());
	qRegisterMetaType<cv::Mat>("cv::Mat");
	connect(thread, &SlamThread::OneIterationDone, this, &MainWindow::onBenchmarkOneIterationDone);
	connect(thread, &SlamThread::RegistrationDone, this, &MainWindow::onBenchmarkRegistrationDone);
	thread->start();
	uiDockBenchmark->pushButtonRun->setDisabled(true);
	uiDockBenchmark->pushButtonSave->setDisabled(true);
}

void MainWindow::onBenchmarkPushButtonDirectoryClicked(bool checked)
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
		uiDockBenchmark->lineEditDirectory->setText(directory);
	}
}

void MainWindow::onBenchmarkPushButtonSaveClicked(bool checked)
{
	QString filename = QFileDialog::getSaveFileName(this, tr("Save Transformations"), "", tr("txt files(*.txt)"));
	if (!filename.isNull())
	{
		vector<pair<double, Eigen::Matrix4f>> trans = engine->GetTransformations();
		ofstream outfile(filename.toStdString());
		for (int i = 0; i < trans.size(); i++)
		{
			Eigen::Vector3f t = TranslationFromMatrix4f(trans[i].second);
			Eigen::Quaternionf q = QuaternionsFromMatrix4f(trans[i].second);

			outfile << fixed << setprecision(6) << trans[i].first
				<< ' ' << t(0) << ' ' << t(1) << ' ' << t(2)
				<< ' ' << q.x() << ' ' << q.y() << ' ' << q.z() << ' ' << q.w() << endl;
		}
		outfile.close();
	}
}

void MainWindow::onBenchmarkPushButtonSaveKeyframesClicked(bool checked)
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

		for (int i = 0; i < engine->keyframe_candidates.size(); i++)
		{
			QString fn = fi.absoluteFilePath() + "/keyframe_candidate_" + QString::number(i) + "_rgb.png";
			cv::imwrite(fn.toStdString(), engine->keyframe_candidates[i].first);
// 			fn = fi.absoluteFilePath() + "/keyframe_candidate_" + QString::number(i) + "_depth.png";
// 			cv::imwrite(fn.toStdString(), engine->keyframe_candidates[i].second);
		}

		for (int i = 0; i < engine->keyframes.size(); i++)
		{
			QString inliers_sig = QString::fromStdString(engine->keyframes_inliers_sig[i]);
			QString exists_sig = QString::fromStdString(engine->keyframes_exists_sig[i]);
			QString fn = fi.absoluteFilePath() + "/keyframe_" + QString::number(i) + "_rgb_" +
				inliers_sig + "_" + exists_sig + ".png";
			cv::imwrite(fn.toStdString(), engine->keyframe_candidates[i].first);
// 			fn = fi.absoluteFilePath() + "/keyframe_" + QString::number(i) + "_depth.png";
// 			cv::imwrite(fn.toStdString(), engine->keyframe_candidates[i].second);
		}
	}
}

void MainWindow::onBenchmarkOneIterationDone(const cv::Mat &rgb, const cv::Mat &depth)
{
	cv::Mat tempRgb, tempDepth;
	QImage *rgbImage, *depthImage;
	cv::cvtColor(rgb, tempRgb, CV_BGR2RGB);
	rgbImage = new QImage((const unsigned char*)(tempRgb.data),
		tempRgb.cols, tempRgb.rows,
		tempRgb.cols * tempRgb.channels(),
		QImage::Format_RGB888);
	benchmarkViewer->ShowRGBImage(rgbImage);

	depth *= 255.0 / 65535.0;
	depth.convertTo(tempDepth, CV_8U);
	cv::cvtColor(tempDepth, tempDepth, CV_GRAY2RGB);
	depthImage = new QImage((const unsigned char*)(tempDepth.data),
		tempDepth.cols, tempDepth.rows,
		tempDepth.cols * tempDepth.channels(),
		QImage::Format_RGB888);
	benchmarkViewer->ShowDepthImage(depthImage);
	if (engine->GetFrameID() % 30 == 0)
	{
		benchmarkViewer->ShowPointCloud(engine->GetScene());
	}
}

void MainWindow::onBenchmarkRegistrationDone()
{
	benchmarkViewer->ShowPointCloud(engine->GetScene());
	uiDockBenchmark->pushButtonRun->setDisabled(false);
	uiDockBenchmark->pushButtonSave->setDisabled(false);
}