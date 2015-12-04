#ifndef BENCHMARKVIEWER_H
#define BENCHMARKVIEWER_H

#include <QWidget>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <vtkRenderWindow.h>

typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

namespace Ui {
	class BenchmarkViewer;
}

class BenchmarkViewer : public QWidget
{
    Q_OBJECT

public:
	explicit BenchmarkViewer(QWidget *parent = 0);
	~BenchmarkViewer();

	void ShowRGBImage(QImage *rgb);
	void ShowDepthImage(QImage *depth);

protected:
	virtual void resizeEvent(QResizeEvent *event);

protected:
	pcl::visualization::PCLVisualizer::Ptr viewer;
	PointCloudT::Ptr cloud;

private:
	Ui::BenchmarkViewer *ui;
	QImage *currRGB;
	QImage *currDepth;
};

#endif // BENCHMARKVIEWER_H
