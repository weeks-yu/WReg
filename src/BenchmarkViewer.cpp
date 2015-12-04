#include "BenchmarkViewer.h"
#include "ui_BenchmarkViewer.h"

#include <QPainter>

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

	currRGB = nullptr;
	currDepth = nullptr;
}

BenchmarkViewer::~BenchmarkViewer()
{
	delete ui;
}

void BenchmarkViewer::ShowRGBImage(QImage *rgb)
{
	if (rgb == nullptr)
		return;

	currRGB = new QImage(*rgb);

	QSize size = ui->labelRGB->size();
	QPixmap pix(size);
	QPainter painter(&pix);
	QImage img = currRGB->scaled(size, Qt::KeepAspectRatio, Qt::SmoothTransformation);
	painter.drawImage(QPoint((size.width() - img.width()) / 2, (size.height() - img.height()) / 2), img);
	ui->labelRGB->setPixmap(pix);
}

void BenchmarkViewer::ShowDepthImage(QImage *depth)
{
	if (depth == nullptr)
		return;

	ui->labelDepth->setPixmap(QPixmap::fromImage(*depth));
	this->update();
}

void BenchmarkViewer::resizeEvent(QResizeEvent *event)
{
	if (currRGB != nullptr)
	{
		QSize size = ui->labelRGB->size();
		QPixmap pix(size);
		QPainter painter(&pix);
		QImage img = currRGB->scaled(size, Qt::KeepAspectRatio, Qt::SmoothTransformation);
		painter.drawImage(QPoint((size.width() - img.width()) / 2, (size.height() - img.height()) / 2), img);
		ui->labelRGB->setPixmap(pix);
	}
}