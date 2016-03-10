#include "SlamEngine.h"
#include "PointCloud.h"
#include "Transformation.h"

SlamEngine::SlamEngine()
{
	frame_id = 0;

	using_downsampling = true;
	downsample_rate = 0.02;

	using_srba_optimizer = true;
	feature_type = SURF;

	using_gicp = true;
	gicp = new pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB>();
	gicp->setMaximumIterations(50);
	gicp->setMaxCorrespondenceDistance(0.1);
	gicp->setTransformationEpsilon(1e-4);

	using_icpcuda = false;
	icpcuda = nullptr;
	threads = Config::instance()->get<int>("icpcuda_threads");
	blocks = Config::instance()->get<int>("icpcuda_blocks");

	// statistics
	min_pt_count = numeric_limits<int>::max();
	max_pt_count = 0;
	min_icp_time = numeric_limits<float>::max();
	max_icp_time = 0;
	total_icp_time = 0;
	min_fit = numeric_limits<float>::max();
	max_fit = 0;
}

SlamEngine::~SlamEngine()
{
	if (gicp)
	{
		delete gicp;
	}
	if (icpcuda)
	{
		delete icpcuda;
	}
}

void SlamEngine::setUsingGicp(bool use)
{
	using_gicp = use;
	if (use)
	{
		if (gicp == nullptr)
			gicp = new pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB>();
	}
	else if (gicp)
	{
		delete gicp;
		gicp = nullptr;
	}
}

void SlamEngine::setUsingIcpcuda(bool use)
{
	using_icpcuda = use;
	if (use)
	{
		int width = Config::instance()->get<int>("image_width");
		int height = Config::instance()->get<int>("image_height");
		double cx = Config::instance()->get<double>("camera_cx");
		double cy = Config::instance()->get<double>("camera_cy");
		double fx = Config::instance()->get<double>("camera_fx");
		double fy = Config::instance()->get<double>("camera_fy");
		double depthFactor = Config::instance()->get<double>("depth_factor");
		if (icpcuda == nullptr)
			icpcuda = new ICPOdometry(width, height, cx, cy, fx, fy, depthFactor);
	}
	else if (icpcuda)
	{
		delete icpcuda;
		icpcuda = nullptr;
	}
}

void SlamEngine::RegisterNext(const cv::Mat &imgRGB, const cv::Mat &imgDepth, double timestamp)
{
	timestamps.push_back(timestamp);
	PointCloudPtr cloud_new = ConvertToPointCloudWithoutMissingData(imgDepth, imgRGB, timestamp, frame_id);

	std::cout << "Frame " << frame_id << ": ";

	if (frame_id == 0)
	{
		total_start = clock();
		last_cloud = cloud_new;
		if (using_downsampling)
		{
			last_cloud = DownSamplingByVoxelGrid(cloud_new, downsample_rate, downsample_rate, downsample_rate);
		}
		point_clouds.push_back(last_cloud);

		int m_size = last_cloud->size();
		if (m_size < min_pt_count) min_pt_count = m_size;
		if (m_size > max_pt_count) max_pt_count = m_size;
		std::cout << "size: " << m_size;

		if (using_icpcuda)
		{
			imgDepth.copyTo(last_depth);
		}

		if (using_hogman_optimizer)
		{
			Frame *frame;
			if (feature_type == SIFT)
			{
				frame = new Frame(imgRGB, imgDepth, "SIFT", Eigen::Matrix4f::Identity());
			}
			else if (feature_type == SURF)
			{
				frame = new Frame(imgRGB, imgDepth, "SURF", Eigen::Matrix4f::Identity());
			}
			frame->relative_tran = Eigen::Matrix4f::Identity();
			hogman_manager.active_window.build(0.0, 0.0, Config::instance()->get<float>("quadtree_size"), 4);
			// test
			string inliers, exists;
			bool isKeyframe = hogman_manager.addNode(frame, 1.0, true, &inliers, &exists);
			keyframe_candidates.push_back(pair<cv::Mat, cv::Mat>(imgRGB, imgDepth));
			if (isKeyframe)
			{
				keyframes.push_back(pair<cv::Mat, cv::Mat>(imgRGB, imgDepth));
				keyframes_inliers_sig.push_back(inliers);
				keyframes_exists_sig.push_back(exists);
			}
		}
		else if (using_srba_optimizer)
		{
			Frame *frame;
			if (feature_type == SIFT)
			{
				frame = new Frame(imgRGB, imgDepth, "SIFT", Eigen::Matrix4f::Identity());
			}
			else if (feature_type == SURF)
			{
				frame = new Frame(imgRGB, imgDepth, "SURF", Eigen::Matrix4f::Identity());
			}
			frame->f.buildFlannIndex();
			frame->relative_tran = Eigen::Matrix4f::Identity();
			srba_manager.active_window.build(0.0, 0.0, Config::instance()->get<float>("quadtree_size"), 4);
			string inliers, exists;
			bool isKeyframe = srba_manager.addNode(frame, 1.0, true, &inliers, &exists);
			keyframe_candidates.push_back(pair<cv::Mat, cv::Mat>(imgRGB, imgDepth));
			if (isKeyframe)
			{
				keyframes.push_back(pair<cv::Mat, cv::Mat>(imgRGB, imgDepth));
				keyframes_inliers_sig.push_back(inliers);
				keyframes_exists_sig.push_back(exists);
			}
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
		if (m_size < min_pt_count) min_pt_count = m_size;
		if (m_size > max_pt_count) max_pt_count = m_size;
		std::cout << "size: " << m_size;

		if (using_gicp)
		{
			step_start = clock();
			pcl::transformPointCloud(*cloud_for_registration, *cloud_transformed, last_transformation);

			gicp->setInputSource(cloud_transformed);
			gicp->setInputTarget(last_cloud);
			gicp->align(*cloud_transformed);

			Eigen::Matrix4f tran = gicp->getFinalTransformation();
			step_time = (clock() - step_start) / 1000.0;
			if (step_time < min_icp_time) min_icp_time = step_time;
			if (step_time > max_icp_time) max_icp_time = step_time;
			total_icp_time += step_time;
			std::cout << ", gicp time: " << fixed << setprecision(3) << step_time;

			weight = sqrt(1.0 / gicp->getFitnessScore());
			if (weight < min_fit) min_fit = weight;
			if (weight > max_fit) max_fit = weight;
			std::cout << ", Weight: " << fixed << setprecision(3) << weight;

			relative_tran = tran * last_transformation;	
		}
		if (using_icpcuda)
		{
			step_start = clock();
			icpcuda->initICPModel((unsigned short *)last_depth.data, 20.0f, Eigen::Matrix4f::Identity());
			icpcuda->initICP((unsigned short *)imgDepth.data, 20.0f);

			Eigen::Vector3f trans = relative_tran.topRightCorner(3, 1);
			Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rot = relative_tran.topLeftCorner(3, 3);

			icpcuda->getIncrementalTransformation(trans, rot, threads, blocks);

			step_time = (clock() - step_start) / 1000.0;
			if (step_time < min_icp_time) min_icp_time = step_time;
			if (step_time > max_icp_time) max_icp_time = step_time;
			total_icp_time += step_time;
			std::cout << ", icpcuda time: " << fixed << setprecision(3) << step_time;

			weight = icpcuda->lastICPError > 0 ? sqrt(1.0 / icpcuda->lastICPError) : sqrt(1000000);
			if (weight < min_fit) min_fit = weight;
			if (weight > max_fit) max_fit = weight;
			std::cout << ", Weight: " << fixed << setprecision(3) << weight;

			relative_tran.topLeftCorner(3, 3) = rot;
			relative_tran.topRightCorner(3, 1) = trans;
		}
		if (using_hogman_optimizer)
		{
			global_tran = relative_tran * hogman_manager.getLastTransformation();
			Frame *frame_now;
			if (IsTransformationBigEnough(hogman_manager.getLastKeyframeTransformation().inverse() * global_tran))
			{
				step_start = clock();
				if (feature_type == SIFT)
				{
					frame_now = new Frame(imgRGB, imgDepth, "SIFT", global_tran);
				}
				else if (feature_type == SURF)
				{
					frame_now = new Frame(imgRGB, imgDepth, "SURF", global_tran);
				}
				frame_now->relative_tran = relative_tran;
				step_time = (clock() - step_start) / 1000.0;
				std::cout << endl;
				std::cout << "Feature: " << fixed << setprecision(3) << step_time;

				// test
				string inliers, exists;
				bool isKeyframe = hogman_manager.addNode(frame_now, weight, true, &inliers, &exists);

				// record all keyframe
				
				keyframe_candidates.push_back(pair<cv::Mat, cv::Mat>(imgRGB, imgDepth));
				if (isKeyframe)
				{
					keyframes.push_back(pair<cv::Mat, cv::Mat>(imgRGB, imgDepth));
					keyframes_inliers_sig.push_back(inliers);
					keyframes_exists_sig.push_back(exists);
				}
			}
			else
			{
				frame_now = new Frame();
				frame_now->relative_tran = relative_tran;
				frame_now->tran = global_tran;
				hogman_manager.addNode(frame_now, weight, false);
			}
		}
		else if (using_srba_optimizer)
		{
			global_tran = relative_tran * srba_manager.getLastTransformation();
			Frame *frame_now;
			if (IsTransformationBigEnough(srba_manager.getLastKeyframeTransformation().inverse() * global_tran))
			{
				step_start = clock();
				if (feature_type == SIFT)
				{
					frame_now = new Frame(imgRGB, imgDepth, "SIFT", global_tran);
				}
				else if (feature_type == SURF)
				{
					frame_now = new Frame(imgRGB, imgDepth, "SURF", global_tran);
				}
				frame_now->f.buildFlannIndex();
				frame_now->relative_tran = relative_tran;
				step_time = (clock() - step_start) / 1000.0;
				std::cout << endl;
				std::cout << "Feature: " << fixed << setprecision(3) << step_time;

				// test
				string inliers, exists;
				bool isKeyframe = srba_manager.addNode(frame_now, weight, true, &inliers, &exists);

				// record all keyframe

				keyframe_candidates.push_back(pair<cv::Mat, cv::Mat>(imgRGB, imgDepth));
				if (isKeyframe)
				{
					keyframes.push_back(pair<cv::Mat, cv::Mat>(imgRGB, imgDepth));
					keyframes_inliers_sig.push_back(inliers);
					keyframes_exists_sig.push_back(exists);
				}
			}
			else
			{
				frame_now = new Frame();
				frame_now->relative_tran = relative_tran;
				frame_now->tran = global_tran;
				srba_manager.addNode(frame_now, weight, false);
			}
		}
		else
		{
			transformation_matrix.push_back(relative_tran * transformation_matrix[frame_id - 1]);
		}
		last_transformation = relative_tran;
		last_cloud = cloud_for_registration;
		if (using_icpcuda)
			imgDepth.copyTo(last_depth);
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
		if (using_hogman_optimizer)
			pcl::transformPointCloud(*point_clouds[i], *tc, hogman_manager.getTransformation(i));
		else if (using_srba_optimizer)
			pcl::transformPointCloud(*point_clouds[i], *tc, srba_manager.getTransformation(i));
		else
			pcl::transformPointCloud(*point_clouds[i], *tc, transformation_matrix[i]);
		
		*cloud += *tc;
		if ((i + 1) % 30 == 0)
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
		if (using_hogman_optimizer)
			ret.push_back(pair<double, Eigen::Matrix4f>(timestamps[i], hogman_manager.getTransformation(i)));
		else
			ret.push_back(pair<double, Eigen::Matrix4f>(timestamps[i], transformation_matrix[i]));
	}
	return ret;
}

void SlamEngine::ShowStatistics()
{
	cout << "-------------------------------------------------------------------------------" << endl;
	cout << "-------------------------------------------------------------------------------" << endl;
	cout << "Total runtime         : " << (clock() - total_start) / 1000.0 << endl;
	cout << "Total frames          : " << frame_id << endl;
	if (using_hogman_optimizer)
		cout << "Number of keyframes   : " << hogman_manager.keyframeCount << endl;
	else
		cout << "Number of keyframes   : " << srba_manager.keyframeInQuadTreeCount << endl;
	cout << "Min Cloud Size : " << min_pt_count << "\t\t Max Cloud Size: " << max_pt_count << endl;
	cout << "-------------------------------------------------------------------------------" << endl;
	cout << "Min Icp Time          : " << min_icp_time << "\t\tMax Gicp Time: " << max_icp_time << endl;
	cout << "Avg Icp Time          : " << total_icp_time / frame_id << endl;
	cout << "Min Fitness Score     : " << fixed << setprecision(7) << min_fit << "\tMax Fitness Score: " << max_fit << endl;
	cout << "-------------------------------------------------------------------------------" << endl;
	if (using_hogman_optimizer)
	{
		cout << "Min Closure Time      : " << fixed << setprecision(3) << srba_manager.min_closure_detect_time << ",\t\tMax Closure Time: " << srba_manager.max_closure_detect_time << endl;
		cout << "Avg Closure Time      : " << hogman_manager.total_closure_detect_time / hogman_manager.clousureCount << endl;
		cout << "Min Closure Candidate : " << hogman_manager.min_closure_candidate << "\t\tMax Closure Candidate: " << hogman_manager.max_closure_candidate << endl;
		cout << "-------------------------------------------------------------------------------" << endl;
		cout << "Min Graph Time        : " << hogman_manager.min_graph_opt_time << "\t\tMax Graph Time: " << hogman_manager.max_graph_opt_time << endl;
		cout << "Avg Graph Time        : " << hogman_manager.total_graph_opt_time / frame_id << endl;
		cout << "Min Edge Weight       : " << hogman_manager.min_edge_weight << "\t\tMax Edge Weight: " << hogman_manager.max_edge_weight << endl;
	}
	else
	{
		cout << "Min Closure Time      : " << fixed << setprecision(3) << srba_manager.min_closure_detect_time << ",\t\tMax Closure Time: " << srba_manager.max_closure_detect_time << endl;
		cout << "Avg Closure Time      : " << srba_manager.total_closure_detect_time / srba_manager.clousureCount << endl;
		cout << "Min Closure Candidate : " << srba_manager.min_closure_candidate << "\t\tMax Closure Candidate: " << srba_manager.max_closure_candidate << endl;
		cout << "-------------------------------------------------------------------------------" << endl;
		cout << "Min Graph Time        : " << srba_manager.min_graph_opt_time << "\t\tMax Graph Time: " << srba_manager.max_graph_opt_time << endl;
		cout << "Avg Graph Time        : " << srba_manager.total_graph_opt_time / frame_id << endl;
		cout << "Min Edge Weight       : " << srba_manager.min_edge_weight << "\t\tMax Edge Weight: " << srba_manager.max_edge_weight << endl;
	}
	cout << endl;
}