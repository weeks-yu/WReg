#include "PclViewer.h"
#include "ui_PclViewer.h"

PclViewer::PclViewer(PointCloudT::Ptr cloud, QWidget *parent) :
    QWidget(parent),
	ui(new Ui::PclViewer)
{
	ui->setupUi(this);

	this->cloud = cloud;

 	viewer.reset(new pcl::visualization::PCLVisualizer("viewer", false));
	ui->qvtkWidget->SetRenderWindow(viewer->getRenderWindow());
	
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
	viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "cloud");

	viewer->resetCamera();
}

PclViewer::~PclViewer()
{
	delete ui;
}

PointCloudPtr PclViewer::GetPointCloud()
{
	return cloud;
}