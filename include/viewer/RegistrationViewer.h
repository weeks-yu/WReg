#ifndef BENCHMARKVIEWER_H
#define BENCHMARKVIEWER_H

#include "SlamThread.h"

#include <QWidget>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <vtkRenderWindow.h>

#include "PointCloud.h"
#include "TriangleMesh.h"
#include "OpenGLWindow.h"

namespace Ui {
	class RegistrationViewer;
}

class RegistrationViewer : public QWidget
{
    Q_OBJECT

public:
	enum
	{
		USING_PCL_VIEWER = 0,
		USING_OPENGL_VIEWER
	};

public:
	explicit RegistrationViewer(QWidget *parent = 0);
	~RegistrationViewer();

	void ShowRGBImage(QImage *rgb);
	void ShowDepthImage(QImage *depth);
	void ShowPointCloud(PointCloudPtr cloud_);
	void ShowMesh(GLMesh *mesh_);
	PointCloudPtr GetPointCloud();

	void SetViewerMode(int mode);
	int GetViewerMode();

protected:
	virtual void resizeEvent(QResizeEvent *event);

private:
	QPixmap getPixmap(QImage image, QSize size, bool keepAspectRatio = true);

private slots:
	void onSplitterVerticalMoved(int pos, int index);
	void onSplitterHorizontalMoved(int pos, int index);

protected:
	pcl::visualization::PCLVisualizer::Ptr viewer;
	pcl::visualization::Camera camera;
	PointCloudPtr cloud;

	OpenGLWindow *glwindow;
	QWidget *glwidget;
	GLMesh *mesh;

	int mode;

private:
	Ui::RegistrationViewer *ui;
	QImage currRGB;
	QImage currDepth;
};

#endif // BENCHMARKVIEWER_H
