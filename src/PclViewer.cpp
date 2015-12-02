#include "PclViewer.h"

PclViewer::PclViewer(PointCloudT::Ptr cloud, QWidget *parent) :
    QVTKWidget(parent)
{
	this->cloud = cloud;

 	viewer.reset(new pcl::visualization::PCLVisualizer("viewer", true));
	this->SetRenderWindow(viewer->getRenderWindow());
	
	//pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(this->cloud);
	viewer->addPointCloud(this->cloud, "cloud");

	viewer->resetCamera();
}

PclViewer::~PclViewer()
{
}