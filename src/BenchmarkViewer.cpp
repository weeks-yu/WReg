#include "BenchmarkViewer.h"
#include "ui_BenchmarkViewer.h"

BenchmarkViewer::BenchmarkViewer(QWidget *parent) :
	QWidget(parent),
	ui(new Ui::BenchmarkViewer)
{
	ui->setupUi(this);
	this->setAutoFillBackground(true);
	QPalette palette;
	palette.setColor(QPalette::Background, Qt::gray);
	this->setPalette(palette);

	viewer.reset(new pcl::visualization::PCLVisualizer("viewer", false));
	ui->qvtkWidget->SetRenderWindow(viewer->getRenderWindow());
	viewer->resetCamera();
}

BenchmarkViewer::~BenchmarkViewer()
{
	delete ui;
}

void BenchmarkViewer::ShowRGBImage(QImage *rgb)
{
	if (rgb == nullptr)
		return;

	ui->labelRGB->setPixmap(QPixmap::fromImage(*rgb));
	this->update();
}

void BenchmarkViewer::ShowDepthImage(QImage *depth)
{
	if (depth == nullptr)
		return;

	ui->labelDepth->setPixmap(QPixmap::fromImage(*depth));
	this->update();
}