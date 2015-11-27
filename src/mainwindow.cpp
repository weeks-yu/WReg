#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

	cloud.reset(new PointCloudT);
	// The number of points in the cloud
	cloud->points.resize(200);

	// Fill the cloud with some points
	for (size_t i = 0; i < cloud->points.size(); ++i)
	{
		cloud->points[i].x = 1024 * rand() / (RAND_MAX + 1.0f);
		cloud->points[i].y = 1024 * rand() / (RAND_MAX + 1.0f);
		cloud->points[i].z = 1024 * rand() / (RAND_MAX + 1.0f);

		cloud->points[i].r = 0;
		cloud->points[i].g = 255;
		cloud->points[i].b = 0;
	}
}

MainWindow::~MainWindow()
{
    delete ui;
}
