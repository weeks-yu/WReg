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
	timestamps.push_back(timestamp);
	PointCloudPtr cloud_new = ConvertToPointCloudWithoutMissingData(imgDepth, imgRGB, timestamp, frame_id);

	std::cout << "Frame " << frame_id << ": ";

	if (frame_id == 0)
	{
		last_cloud = cloud_new;
		if (using_downsampling)
		{
			last_cloud = DownSamplingByVoxelGrid(cloud_new, downsample_rate, downsample_rate, downsample_rate);
		}
		point_clouds.push_back(last_cloud);

		int m_size = last_cloud->size();
		//if (m_size < min_pt_count) min_pt_count = m_size;
		//if (m_size > max_pt_count) max_pt_count = m_size;
		std::cout << "size: " << m_size;

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
		else
		{
			transformation_matrix.push_back(Eigen::Matrix4f::Identity());
		}
		last_transformation = Eigen::Matrix4f::Identity();
	}
	else
	{
		PointCloudPtr cloud_for_registration = cloud_new;
		PointCloudPtr cloud_transformed(new PointCloudT);
		Eigen::Matrix4f relative_tran = Eigen::Matrix4f::Identity();
		Eigen::Matrix4f global_tran = Eigen::Matrix4f::Identity();
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
		std::cout << "size: " << m_size;

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
			std::cout << ", gicp time: " << fixed << setprecision(3) << step_time;

			weight = sqrt(1.0 / gicp.getFitnessScore());
// 			if (weight < min_fit) min_fit = weight;
// 			if (weight > max_fit) max_fit = weight;
			std::cout << ", Weight: " << fixed << setprecision(3) << weight;

			relative_tran = tran * last_transformation;	
		}
		if (using_graph_optimizer)
		{
			global_tran = relative_tran * graph_manager.getLastTransformation();
			Frame *frame_now;
			if (IsTransformationBigEnough(graph_manager.getLastKeyframeTransformation().inverse() * global_tran))
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
				std::cout << endl;
				std::cout << "Feature: " << fixed << setprecision(3) << step_time;
				graph_manager.addNode(frame_now, relative_tran, weight, true);
			}
			else
			{
				frame_now = new Frame();
				frame_now->tran = global_tran;
				graph_manager.addNode(frame_now, relative_tran, weight, false);
			}
		}
		else
		{
			transformation_matrix.push_back(relative_tran * transformation_matrix[frame_id - 1]);
		}
		last_transformation = relative_tran;
		last_cloud = cloud_for_registration;
	}
	std::cout << endl;
	frame_id++;
}

PointCloudPtr SlamEngine::GetScene()
{
	PointCloudPtr cloud(new PointCloudT);
	for (int i = 0; i < frame_id; i++)
	{
		PointCloudPtr tc(new PointCloudT);
		if (using_graph_optimizer)
			pcl::transformPointCloud(*point_clouds[i], *tc, graph_manager.getTransformation(i));
		else
			pcl::transformPointCloud(*point_clouds[i], *tc, transformation_matrix[i]);
		
		*cloud += *tc;
		if ((i + 1) % 20 == 0)
		{
			cloud = DownSamplingByVoxelGrid(cloud, downsample_rate, downsample_rate, downsample_rate);
		}
	}
	return cloud;
}

vector<pair<double, Eigen::Matrix4f>> SlamEngine::GetTransformations()
{
	vector<pair<double, Eigen::Matrix4f>> ret;

	for (int i = 0; i < frame_id; i++)
	{
		if (using_graph_optimizer)
			ret.push_back(pair<double, Eigen::Matrix4f>(timestamps[i], graph_manager.getTransformation(i)));
		else
			ret.push_back(pair<double, Eigen::Matrix4f>(timestamps[i], transformation_matrix[i]));
	}
	return ret;
}