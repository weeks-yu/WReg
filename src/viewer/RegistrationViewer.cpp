#include "RegistrationViewer.h"
#include "ui_RegistrationViewer.h"

#include <QPainter>

RegistrationViewer::RegistrationViewer(QWidget *parent) :
	QWidget(parent),
	ui(new Ui::RegistrationViewer)
{
	ui->setupUi(this);
	this->setAutoFillBackground(true);
	QPalette palette;
	palette.setColor(QPalette::Background, Qt::gray);
	this->setPalette(palette);

	viewer.reset(new pcl::visualization::PCLVisualizer("viewer", false));
	ui->qvtkWidget->SetRenderWindow(viewer->getRenderWindow());
	viewer->resetCamera();
	viewer->setCameraPosition(0, 0, -10, 0, 0, 5, 0, 1, 0);

	cloud = PointCloudPtr(new PointCloudT);
}

RegistrationViewer::~RegistrationViewer()
{
	delete ui;
}

void RegistrationViewer::ShowRGBImage(QImage *rgb)
{
	if (rgb == nullptr)
		return;

	currRGB = rgb->copy(0, 0, rgb->width(), rgb->height());
	ui->labelRGB->setPixmap(getPixmap(currRGB, ui->labelRGB->size()));
}

void RegistrationViewer::ShowDepthImage(QImage *depth)
{
	if (depth == nullptr)
		return;

	currDepth = depth->copy(0, 0, depth->width(), depth->height());
	ui->labelDepth->setPixmap(getPixmap(currDepth, ui->labelDepth->size()));
}

void RegistrationViewer::ShowPointCloud(PointCloudPtr result)
{
	*cloud = *result;
	viewer->removeAllPointClouds();
	pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb(cloud);
	viewer->addPointCloud<PointT>(cloud, rgb, "cloud");
	ui->qvtkWidget->update();
}

PointCloudPtr RegistrationViewer::GetPointCloud()
{
	return cloud;
}

void RegistrationViewer::resizeEvent(QResizeEvent *event)
{
	if (!currRGB.isNull())
	{
		ui->labelRGB->setPixmap(getPixmap(currRGB, ui->labelRGB->size()));
	}
	if (!currDepth.isNull())
	{
		ui->labelDepth->setPixmap(getPixmap(currDepth, ui->labelDepth->size()));
	}
}

QPixmap RegistrationViewer::getPixmap(QImage image, QSize size, bool keepAspectRatio /* = true */)
{
	QPixmap pix(size);
	QPainter painter(&pix);
	QImage img = image.scaled(size, Qt::KeepAspectRatio, Qt::SmoothTransformation);
	painter.fillRect(0, 0, size.width(), size.height(), Qt::gray);
	painter.drawImage(QPoint((size.width() - img.width()) / 2, (size.height() - img.height()) / 2), img);
	return pix;
}

void RegistrationViewer::onSplitterVerticalMoved(int pos, int index)
{
	if (!currRGB.isNull())
		ui->labelRGB->setPixmap(getPixmap(currRGB, ui->labelRGB->size()));
	if (!currDepth.isNull())
		ui->labelDepth->setPixmap(getPixmap(currDepth, ui->labelDepth->size()));
}

void RegistrationViewer::onSplitterHorizontalMoved(int pos, int index)
{
	if (!currRGB.isNull())
		ui->labelRGB->setPixmap(getPixmap(currRGB, ui->labelRGB->size()));
	if (!currDepth.isNull())
		ui->labelDepth->setPixmap(getPixmap(currDepth, ui->labelDepth->size()));
}