#include "MainWindow.h"
#include "ui_MainWindow.h"
#include "ui_DockBenchmark.h"
#include "PclViewer.h"
#include "Parser.h"
#include "SlamThread.h"

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
	this->addDockWidget(Qt::LeftDockWidgetArea, dockBenchmark);

	connect(uiDockBenchmark->pushButtonRun, &QPushButton::clicked, this, &MainWindow::onBenchmarkPushButtonRunClicked);
	connect(uiDockBenchmark->pushButtonDirectory, &QPushButton::clicked, this, &MainWindow::onBenchmarkPushButtonDirectoryClicked);

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
	bool usingDownSampling = uiDockBenchmark->checkBoxDownSampling->isChecked();
	int method = uiDockBenchmark->comboBoxMethod->currentIndex();
	bool usingGICP = method == 0;
	bool usingGraphOptimizer = method == 0;
	
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
	engine->setUsingGraphOptimizer(usingGraphOptimizer);
	if (usingGraphOptimizer)
	{
		QString type = uiDockBenchmark->comboBoxGraphFeatureType->currentText();
		engine->setGraphFeatureType(type == "SIFT" ? SlamEngine::SIFT : SlamEngine::SURF);
	}

	SlamThread *thread = new SlamThread(uiDockBenchmark->lineEditDirectory->text(), engine);
	qRegisterMetaType<cv::Mat>("cv::Mat");
	connect(thread, &SlamThread::OneIterationDone, this, &MainWindow::onBenchmarkOneIterationDone);
	thread->start();
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
}