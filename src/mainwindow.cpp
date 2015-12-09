#include "MainWindow.h"
#include "ui_MainWindow.h"
#include "ui_DockBenchmark.h"
#include "PclViewer.h"
#include "BenchmarkViewer.h"
#include "Parser.h"

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

// 	cloud.reset(new PointCloudT);
// 	// The number of points in the cloud
// 	cloud->points.resize(200);
// 
// 	// Fill the cloud with some points
// 	for (size_t i = 0; i < cloud->points.size(); ++i)
// 	{
// 		cloud->points[i].x = 1024 * rand() / (RAND_MAX + 1.0f);
// 		cloud->points[i].y = 1024 * rand() / (RAND_MAX + 1.0f);
// 		cloud->points[i].z = 1024 * rand() / (RAND_MAX + 1.0f);
// 
// 		cloud->points[i].r = 0;
// 		cloud->points[i].g = 255;
// 		cloud->points[i].b = 0;
// 	}
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

void MainWindow::ShowBenchmarkTest(const QString &filename)
{
	QFileInfo fi(filename);
	if (!fi.isFile() || fi.fileName() != "read.txt")
	{
		return;
	}

	cv::Mat imgDepth16u(480, 640, CV_16UC1);
	cv::Mat imgRGB8u(480, 640, CV_8UC3);
	imgRGB8u = cv::imread("rgb.png");
	imgDepth16u = cv::imread("depth.png", -1);

	cv::Mat rgb, depth;
	QImage *rgbImage, *depthImage;
	cv::cvtColor(imgRGB8u, rgb, CV_BGR2RGB);
	rgbImage = new QImage((const unsigned char*)(rgb.data),
		rgb.cols, rgb.rows,
		rgb.cols*rgb.channels(),
		QImage::Format_RGB888);

	// left dock
	if (dockBenchmark == nullptr)
	{
		dockBenchmark = new QDockWidget(this);
		if (uiDockBenchmark == nullptr)
			uiDockBenchmark = new Ui::DockBenchmark;
		uiDockBenchmark->setupUi(dockBenchmark);
	}
	this->addDockWidget(Qt::LeftDockWidgetArea, dockBenchmark);

	connect(uiDockBenchmark->pushButtonRun, &QPushButton::clicked, this, &MainWindow::onBenchmarkPushButtonRunClicked);

	// center
	BenchmarkViewer *benchmarkViewer = new BenchmarkViewer(this);
	QMdiSubWindow *subWindow = new QMdiSubWindow(mdiArea);
	subWindow->setWidget(benchmarkViewer);
	subWindow->setAttribute(Qt::WA_DeleteOnClose);
	mdiArea->addSubWindow(subWindow);
	subWindow->showMaximized();
	benchmarkViewer->ShowRGBImage(rgbImage);
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
	QString filename = QFileDialog::getOpenFileName(this, tr("Open read.txt"), "", tr("read.txt(read.txt)"));
	if (!filename.isNull())
	{
		ShowBenchmarkTest(filename);
	}
}

void MainWindow::onBenchmarkPushButtonRunClicked(bool checked)
{
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
}