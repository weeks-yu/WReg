#include "BenchmarkViewer.h"

BenchmarkViewer::BenchmarkViewer(QWidget *parent) :
	QSplitter(parent)
{
	qvtkWidget = new QVTKWidget(this);
	viewer.reset(new pcl::visualization::PCLVisualizer("viewer", true));
	qvtkWidget->SetRenderWindow(viewer->getRenderWindow());
	viewer->resetCamera();
	qvtkWidget->update();

	this->setOrientation(Qt::Vertical);
	QSplitter *splitter_horizontal = new QSplitter(Qt::Horizontal, this);
	this->addWidget(splitter_horizontal);
	this->addWidget(qvtkWidget);
}

BenchmarkViewer::~BenchmarkViewer()
{
}