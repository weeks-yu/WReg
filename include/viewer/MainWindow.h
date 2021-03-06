#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include "SlamEngine.h"

#include <QMainWindow>
#include <QMdiArea>
#include <QMdiSubWindow>
#include <QDockWidget>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <opencv2/core/core.hpp>

#include <vtkRenderWindow.h>

#include "BenchmarkViewer.h"

typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

namespace Ui {
	class MainWindow;
	class DockBenchmark;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT
public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private:
	void ShowPointCloudFiles(const QString &filename);
	void ShowBenchmarkTest(const QString &directory);
	void ShowBenchmarkResult(const QString &filename, int fi, int fst, int fed);

private slots:
	void onActionOpenTriggered();
	void onActionSaveTriggered();
	void onActionOpenReadTxtTriggered();
	void onActionShowResultFromFileTriggered();

	// benchmark
	void onBenchmarkPushButtonRunClicked(bool checked);
	void onBenchmarkPushButtonDirectoryClicked(bool checked);
	void onBenchmarkPushButtonSaveClicked(bool checked);
	void onBenchmarkPushButtonSaveKeyframesClicked(bool checked);
	void onBenchmarkPushButtonSaveLogsClicked(bool checked);
	void onBenchmarkPushButtonRefineClicked(bool checked);

	void onBenchmarkOneIterationDone(const cv::Mat &rgb, const cv::Mat &depth);
	void onBenchmarkRegistrationDone();

protected:
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
	PointCloudT::Ptr cloud;
	
private:
    Ui::MainWindow *ui;
	Ui::DockBenchmark *uiDockBenchmark;

	QDockWidget *dockBenchmark;
	QMdiArea *mdiArea;
	BenchmarkViewer *benchmarkViewer;

	SlamEngine *engine;

	bool inBenchmarkMode;
};

#endif // MAINWINDOW_H
