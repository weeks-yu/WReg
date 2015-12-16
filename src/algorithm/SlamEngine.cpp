#include "SlamEngine.h"
#include "PointCloud.h"
#include "Transformation.h"

SlamEngine::SlamEngine()
{
	frame_id = 0;

	using_downsampling = true;
	downsample_rate = 0.02;

	using_graph_optimizer = true;
	feature_type = SURF;

	using_gicp = true;
	gicp.setMaximumIterations(50);
	gicp.setMaxCorrespondenceDistance(0.1);
	gicp.setTransformationEpsilon(1e-4);
}

SlamEngine::~SlamEngine()
{

}

void SlamEngine::RegisterNext(const cv::Mat &imgRGB, const cv::Mat &imgDepth, double timestamp)
{
	PointCloudPtr cloud_new = ConvertToPointCloudWithoutMissingData(imgDepth, imgRGB, timestamp, frame_id);

	if (frame_id == 0)
	{
		last_cloud = cloud_new;
		if (using_downsampling)
		{
			last_cloud = DownSamplingByVoxelGrid(cloud_new, downsample_rate, downsample_rate, downsample_rate);
		}
		point_clouds.push_back(last_cloud);

		if (using_graph_optimizer)
		{
			Frame *frame;
			if (feature_type == SIFT)
			{
				frame = new Frame(imgRGB, imgDepth, Eigen::Matrix4f::Identity(), "SIFT");
			}
			else if (feature_type == SURF)
			{
				frame = new Frame(imgRGB, imgDepth, Eigen::Matrix4f::Identity(), "SURF");
			}
			graph_manager.buildQuadTree(0.0, 0.0, Config::instance()->get<float>("quadtree_size"), 4);
			graph_manager.addNode(frame, Eigen::Matrix4f::Identity(), 1.0, true);
		}
		last_transformation = Eigen::Matrix4f::Identity();
	}
	else
	{
		PointCloudPtr cloud_for_registration = cloud_new;
		PointCloudPtr cloud_transformed(new PointCloudType());
		Eigen::Matrix4f relative_tran = Eigen::Matrix4f::Identity();
		Eigen::Matrix4f now_tran = Eigen::Matrix4f::Identity();
		float weight = 1.0;

		clock_t step_start = 0;
		float step_time;

		if (using_downsampling)
		{
			cloud_for_registration = DownSamplingByVoxelGrid(cloud_new, downsample_rate, downsample_rate, downsample_rate);
		}
		point_clouds.push_back(cloud_for_registration);
		int m_size = cloud_for_registration->size();
// 		if (m_size < min_pt_count) min_pt_count = m_size;
// 		if (m_size > max_pt_count) max_pt_count = m_size;
		cout << ", size: " << m_size;

		if (using_gicp)
		{
			step_start = clock();
			pcl::transformPointCloud(*cloud_for_registration, *cloud_transformed, last_transformation);

			gicp.setInputSource(cloud_transformed);
			gicp.setInputTarget(last_cloud);
			gicp.align(*cloud_transformed);

			Eigen::Matrix4f tran = gicp.getFinalTransformation();
			step_time = (clock() - step_start) / 1000.0;
// 			if (step_time < min_gicp_time) min_gicp_time = step_time;
// 			if (step_time > max_gicp_time) max_gicp_time = step_time;
// 			total_gicp_time += step_time;
			cout << ", gicp time: " << fixed << setprecision(3) << step_time;

			weight = sqrt(1.0 / gicp.getFitnessScore());
// 			if (weight < min_fit) min_fit = weight;
// 			if (weight > max_fit) max_fit = weight;
			cout << ", Weight: " << fixed << setprecision(3) << weight;

			relative_tran = tran * last_transformation;
			now_tran = relative_tran * graph_manager.getLastTransformation();
		}
		if (using_graph_optimizer)
		{
			Frame *frame_now;
			if (IsTransformationBigEnough(graph_manager.getLastKeyframeTransformation().inverse() * now_tran))
			{
				step_start = clock();
				if (feature_type == SIFT)
				{
					frame_now = new Frame(imgRGB, imgDepth, Eigen::Matrix4f::Identity(), "SIFT");
				}
				else if (feature_type == SURF)
				{
					frame_now = new Frame(imgRGB, imgDepth, Eigen::Matrix4f::Identity(), "SURF");
				}
				step_time = (clock() - step_start) / 1000.0;
				cout << endl;
				cout << "Feature: " << fixed << setprecision(3) << step_time;
				graph_manager.addNode(frame_now, relative_tran, weight, true);
			}
			else
			{
				frame_now = new Frame();
				frame_now->tran = now_tran;
				graph_manager.addNode(frame_now, relative_tran, weight, false);
			}
		}
		last_transformation = relative_tran;
	}
	frame_id++;
}

PointCloudPtr SlamEngine::GetScene()
{
	PointCloudPtr cloud(new PointCloudType());
	for (int i = 0; i < frame_id; i++)
	{
		PointCloudPtr tc(new PointCloudType());
		pcl::transformPointCloud(*point_clouds[i], *tc, graph_manager.getTransformation(i));
		*cloud += *tc;
		if ((i + 1) % 20 == 0)
		{
			cloud = DownSamplingByVoxelGrid(cloud, downsample_rate, downsample_rate, downsample_rate);
		}
	}
	return cloud;
}