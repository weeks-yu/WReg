#ifndef PCLVIEWER_H
#define PCLVIEWER_H

#include <QWidget>
#include <QVBoxLayout>
#include <QVTKWidget.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <vtkRenderWindow.h>

typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

namespace Ui {
	class PclViewer;
}

class PclViewer : public QWidget
{
    Q_OBJECT

public:
	explicit PclViewer(PointCloudT::Ptr cloud, QWidget *parent = 0);
	~PclViewer();

protected:
	Ui::PclViewer *ui;

	pcl::visualization::PCLVisualizer::Ptr viewer;
	PointCloudT::Ptr cloud;
};

#endif // PCLVIEWER_H
