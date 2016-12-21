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

#include "RegistrationViewer.h"

typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

namespace Ui {
	class MainWindow;
	class DockRegistration;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT
public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private:
	void ShowPointCloudFiles(const QString &filename);
// 	void ShowBenchmarkTest(const QString &directory);
// 	void ShowBenchmarkResult(const QString &filename, int fi, int fst, int fed);
	void ShowRegistration();

private slots:
	void onActionOpenTriggered();
	void onActionSaveTriggered();
// 	void onActionOpenReadTxtTriggered();
// 	void onActionShowResultFromFileTriggered();
	void onActionRegistrationTriggered();

	// registration
	void onRegistrationPushButtonRunClicked(bool checked);
	void onRegistrationPushButtonDirectoryClicked(bool checked);
	void onRegistrationPushButtonSaveTrajectoriesClicked(bool checked);
	void onRegistrationPushButtonSaveKeyframesClicked(bool checked);
	void onRegistrationPushButtonSaveLogsClicked(bool checked);
	void onRegistrationComboBoxSensorTypeCurrentIndexChanged(int index);
	void onRegistrationPushButtonConnectKinectClicked(bool checked);
	void onRegistrationRadioButtonModeToggled(bool checked);

	void onBenchmarkOneIterationDone(const cv::Mat &rgb, const cv::Mat &depth, const Eigen::Matrix4f &tran);
	void onBenchmarkRegistrationDone();

protected:
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
	PointCloudT::Ptr cloud;
	
private:
    Ui::MainWindow *ui;
	Ui::DockRegistration *uiDockRegistration;

	QDockWidget *dockRegistration;
	QMdiArea *mdiArea;
	RegistrationViewer *registrationViewer;

	TsdfModel *tsdf;
	SlamEngine *engine;
	SlamThread *thread;

	bool inRegistrationMode;
};

#endif // MAINWINDOW_H
