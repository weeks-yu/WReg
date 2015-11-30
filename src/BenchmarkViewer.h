#ifndef BENCHMARKVIEWER_H
#define BENCHMARKVIEWER_H

#include <QWidget>
#include <QSplitter>
#include <QVTKWidget.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <vtkRenderWindow.h>

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

namespace Ui {
	class BenchmarkViewer;
}

class BenchmarkViewer : public QSplitter
{
    Q_OBJECT

public:
	explicit BenchmarkViewer(QWidget *parent = 0);
	~BenchmarkViewer();

	virtual QSize sizeHint() const { return QSize(640, 480); }

protected:
	pcl::visualization::PCLVisualizer::Ptr viewer;
	PointCloudT::Ptr cloud;

private:
	QVTKWidget *qvtkWidget;
};

#endif // BENCHMARKVIEWER_H
