#ifndef PCLVIEWER_H
#define PCLVIEWER_H

#include <QWidget>
#include <QVBoxLayout>
#include <QVTKWidget.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <vtkRenderWindow.h>
#include "PointCloud.h"

namespace Ui {
	class PclViewer;
}

class PclViewer : public QWidget
{
    Q_OBJECT

public:
	explicit PclViewer(PointCloudT::Ptr cloud, QWidget *parent = 0);
	~PclViewer();

	PointCloudPtr GetPointCloud();

protected:
	Ui::PclViewer *ui;

	pcl::visualization::PCLVisualizer::Ptr viewer;
	PointCloudPtr cloud;
};

#endif // PCLVIEWER_H
