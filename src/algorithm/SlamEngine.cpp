#include "SlamEngine.h"
#include "PointCloud.h"
#include "Transformation.h"

#ifdef SHOW_Z_INDEX
extern float now_min_z;
extern float now_max_z;
#endif

SlamEngine::SlamEngine()
{
	frame_id = 0;

	using_downsampling = true;
	downsample_rate = 0.01;

	using_optimizer = true;
	using_hogman_optimizer = true;
	using_robust_optimizer = false;
	using_srba_optimizer = false;
	feature_type = SURF;

	using_gicp = false;
	gicp = new pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB>();
	gicp->setMaximumIterations(50);
	gicp->setMaxCorrespondenceDistance(0.1);
	gicp->setTransformationEpsilon(1e-4);

	using_icpcuda = true;
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

	accumulated_frame_count = 0;
	accumulated_transformation = Eigen::Matrix4f::Identity();
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
		float cx = Config::instance()->get<float>("camera_cx");
		float cy = Config::instance()->get<float>("camera_cy");
		float fx = Config::instance()->get<float>("camera_fx");
		float fy = Config::instance()->get<float>("camera_fy");
		float depthFactor = Config::instance()->get<float>("depth_factor");
		float distThresh = Config::instance()->get<float>("dist_threshold");
		float angleThresh = Config::instance()->get<float>("angle_threshold");
		if (icpcuda == nullptr)
			icpcuda = new ICPOdometry(width, height, cx, cy, fx, fy, depthFactor, distThresh, angleThresh);
	}
}

void SlamEngine::setUsingFeature(bool use)
{
	using_feature = use;
	if (using_feature)
	{
		int width = Config::instance()->get<int>("image_width");
		int height = Config::instance()->get<int>("image_height");
		float cx = Config::instance()->get<float>("camera_cx");
		float cy = Config::instance()->get<float>("camera_cy");
		float fx = Config::instance()->get<float>("camera_fx");
		float fy = Config::instance()->get<float>("camera_fy");
		float depthFactor = Config::instance()->get<float>("depth_factor");
		float distThresh = Config::instance()->get<float>("dist_threshold");
		float angleThresh = Config::instance()->get<float>("angle_threshold");
		if (icpcuda == nullptr)
			icpcuda = new ICPOdometry(width, height, cx, cy, fx, fy, depthFactor, distThresh, angleThresh);
	}
}

void SlamEngine::RegisterNext(const cv::Mat &imgRGB, const cv::Mat &imgDepth, double timestamp)
{
	timestamps.push_back(timestamp);
	PointCloudPtr cloud_new = ConvertToPointCloudWithoutMissingData(imgDepth, imgRGB, timestamp, frame_id);

#ifdef SHOW_Z_INDEX
	cout << now_min_z << ' ' << now_max_z << endl;
#endif

	PointCloudPtr cloud_downsampled = DownSamplingByVoxelGrid(cloud_new, downsample_rate, downsample_rate, downsample_rate);

	std::cout << "Frame " << frame_id << ": ";

	if (frame_id == 0)
	{
		total_start = clock();
		last_cloud = cloud_new;
		if (using_downsampling)
		{
			last_cloud = cloud_downsampled;
		}
		point_clouds.push_back(cloud_downsampled);

		int m_size = last_cloud->size();
		if (m_size < min_pt_count) min_pt_count = m_size;
		if (m_size > max_pt_count) max_pt_count = m_size;
		std::cout << "size: " << m_size;

		if (using_icpcuda)
		{
			imgDepth.copyTo(last_depth);
		}

		Frame *frame;
		if (using_feature || using_optimizer)
		{
			if (feature_type == SIFT)
			{
				frame = new Frame(imgRGB, imgDepth, "SIFT", Eigen::Matrix4f::Identity());
			}
			else if (feature_type == SURF)
			{
				frame = new Frame(imgRGB, imgDepth, "SURF", Eigen::Matrix4f::Identity());
			}
			else if (feature_type == ORB)
			{
				frame = new Frame(imgRGB, imgDepth, "ORB", Eigen::Matrix4f::Identity());
			}
			frame->relative_tran = Eigen::Matrix4f::Identity();
			frame->tran = frame->relative_tran;
		}

		if (using_feature)
		{
			frame->f->buildFlannIndex();
			last_frame = frame;
			imgDepth.copyTo(last_depth);
		}

		if (using_optimizer)
		{
			string inliers, exists;
			bool isKeyframe;

			if (using_hogman_optimizer)
			{
				hogman_manager.active_window.build(0.0, 0.0, Config::instance()->get<float>("quadtree_size"), 4);
				isKeyframe = hogman_manager.addNode(frame, 1.0, true, &inliers, &exists);
			}
			else if (using_srba_optimizer)
			{
				srba_manager.active_window.build(0.0, 0.0, Config::instance()->get<float>("quadtree_size"), 4);
				isKeyframe = srba_manager.addNode(frame, 1.0, true, &inliers, &exists);
			}
			else if (using_robust_optimizer)
			{
				robust_manager.active_window.build(0.0, 0.0, Config::instance()->get<float>("quadtree_size"), 4);
				isKeyframe = robust_manager.addNode(frame, 1.0, true, &inliers, &exists);
			}

			last_is_keyframe = isKeyframe;
			if (!isKeyframe && !using_feature)
			{
				delete frame->f;
			}

#ifdef SAVE_TEST_INFOS
			keyframe_candidates_id.push_back(frame_id);
			keyframe_candidates.push_back(pair<cv::Mat, cv::Mat>(imgRGB, imgDepth));

			if (isKeyframe)
			{
				keyframes_id.push_back(frame_id);
				keyframes.push_back(pair<cv::Mat, cv::Mat>(imgRGB, imgDepth));
				keyframes_inliers_sig.push_back(inliers);
				keyframes_exists_sig.push_back(exists);
			}
#endif
		}
		else
		{
			transformation_matrix.push_back(Eigen::Matrix4f::Identity());
		}
		accumulated_frame_count = 0;
		accumulated_transformation = Eigen::Matrix4f::Identity();
		last_transformation = Eigen::Matrix4f::Identity();
	}
	else
	{
		PointCloudPtr cloud_for_registration = cloud_new;
		PointCloudPtr cloud_transformed(new PointCloudT);
		Eigen::Matrix4f relative_tran = Eigen::Matrix4f::Identity();
		Eigen::Matrix4f global_tran = Eigen::Matrix4f::Identity();
		float weight = 1.0;
		bool registration_failed = false;

		clock_t step_start = 0;
		float step_time;

		if (using_downsampling)
		{
			cloud_for_registration = cloud_downsampled;
		}
		point_clouds.push_back(cloud_downsampled);
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

			relative_tran = last_transformation * tran;
		}
		if (using_icpcuda)
		{
			step_start = clock();
			icpcuda->initICPModel((unsigned short *)last_depth.data, 20.0f, Eigen::Matrix4f::Identity());
			icpcuda->initICP((unsigned short *)imgDepth.data, 20.0f);

			Eigen::Vector3f t = relative_tran.topRightCorner(3, 1);
			Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rot = relative_tran.topLeftCorner(3, 3);

			Eigen::Matrix4f estimated_tran = Eigen::Matrix4f::Identity()/*last_transformation*/;
			Eigen::Vector3f estimated_t = estimated_tran.topRightCorner(3, 1);
			Eigen::Matrix<float, 3, 3, Eigen::RowMajor> estimated_rot = estimated_tran.topLeftCorner(3, 3);

			icpcuda->getIncrementalTransformation(t, rot, estimated_t, estimated_rot, threads, blocks);

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
			relative_tran.topRightCorner(3, 1) = t;
		}
		
		Frame *frame_now;
		if (using_feature)
		{
			if (feature_type == SIFT)
			{
				frame_now = new Frame(imgRGB, imgDepth, "SIFT", global_tran);
			}
			else if (feature_type == SURF)
			{
				frame_now = new Frame(imgRGB, imgDepth, "SURF", global_tran);
			}
			else if (feature_type == ORB)
			{
				frame_now = new Frame(imgRGB, imgDepth, "ORB", global_tran);
			}

			vector<cv::DMatch> matches, inliers;
			Eigen::Matrix4f tran;
			Eigen::Matrix<double, 6, 6> information;
			float rmse;
			float coresp;

			last_frame->f->findMatchedPairs(matches, frame_now->f);
			if (Feature::getTransformationByRANSAC(tran, information, coresp, rmse, &inliers, last_frame->f, frame_now->f, nullptr, matches))
			{
				relative_tran = tran;
				cout << ", " << matches.size() << ", " << inliers.size();
			}
			else
			{
				cout << ", " << matches.size() << ", " << inliers.size() << ", RANSAC Failed";
				registration_failed = true;
				relative_tran = last_transformation;

// 				icpcuda->initICPModel((unsigned short *)last_depth.data, 20.0f, Eigen::Matrix4f::Identity());
// 				icpcuda->initICP((unsigned short *)imgDepth.data, 20.0f);
// 
// 				Eigen::Vector3f t = relative_tran.topRightCorner(3, 1);
// 				Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rot = relative_tran.topLeftCorner(3, 3);
// 
// 				Eigen::Matrix4f estimated_tran = Eigen::Matrix4f::Identity();
// 				Eigen::Vector3f estimated_t = estimated_tran.topRightCorner(3, 1);
// 				Eigen::Matrix<float, 3, 3, Eigen::RowMajor> estimated_rot = estimated_tran.topLeftCorner(3, 3);
// 
// 				icpcuda->getIncrementalTransformation(t, rot, estimated_t, estimated_rot, threads, blocks);
// 
// 				relative_tran.topLeftCorner(3, 3) = rot;
// 				relative_tran.topRightCorner(3, 1) = t;
			}

			if (!last_is_keyframe)
			{
				delete last_frame->f;
			}
		}

		accumulated_frame_count++;
		accumulated_transformation = accumulated_transformation * relative_tran;
		bool isBigEnough = IsTransformationBigEnough();

		if (!using_feature && using_optimizer)
		{
/*			step_start = clock();*/
			
			if (isBigEnough || registration_failed)
			{
				if (feature_type == SIFT)
				{
					frame_now = new Frame(imgRGB, imgDepth, "SIFT", global_tran);
				}
				else if (feature_type == SURF)
				{
					frame_now = new Frame(imgRGB, imgDepth, "SURF", global_tran);
				}
				else if (feature_type == ORB)
				{
					frame_now = new Frame(imgRGB, imgDepth, "ORB", global_tran);
				}
			}
			else
			{
				frame_now = new Frame();
			}
// 			step_time = (clock() - step_start) / 1000.0;
// 			std::cout << endl;
// 			std::cout << "Feature: " << fixed << setprecision(3) << step_time;
		}

		if (using_optimizer)
		{
			if (using_hogman_optimizer)
			{
				global_tran = hogman_manager.getLastTransformation() * relative_tran;
			}
			else if (using_srba_optimizer)
			{
				global_tran = srba_manager.getLastTransformation() * relative_tran;
			}
			else if (using_robust_optimizer)
			{
				global_tran = robust_manager.getLastTransformation() * relative_tran;
			}
			frame_now->relative_tran = relative_tran;
			frame_now->tran = global_tran;

			if (isBigEnough || registration_failed)
			{
				frame_now->f->updateFeaturePoints3D(global_tran);

				string inliers, exists;
				bool isKeyframe = false;

				if (using_hogman_optimizer)
				{
					isKeyframe = hogman_manager.addNode(frame_now, weight, true, &inliers, &exists);
				}
				else if (using_srba_optimizer)
				{
					isKeyframe = srba_manager.addNode(frame_now, weight, true, &inliers, &exists);
				}
				else if (using_robust_optimizer)
				{
					isKeyframe = robust_manager.addNode(frame_now, weight, true, &inliers, &exists);
				}
				
				if (using_feature)
				{
					last_is_keyframe = isKeyframe;
				}
				
				if (!isKeyframe && !using_feature)
				{
					delete frame_now->f;
				}

#ifdef SAVE_TEST_INFOS
				keyframe_candidates_id.push_back(frame_id);
				keyframe_candidates.push_back(pair<cv::Mat, cv::Mat>(imgRGB, imgDepth));

				if (isKeyframe)
				{
					keyframes_id.push_back(frame_id);
					keyframes.push_back(pair<cv::Mat, cv::Mat>(imgRGB, imgDepth));
					keyframes_inliers_sig.push_back(inliers);
					keyframes_exists_sig.push_back(exists);
				}
#endif

				accumulated_frame_count = 0;
				accumulated_transformation = Eigen::Matrix4f::Identity();
			}
			else
			{
				if (using_hogman_optimizer)
				{
					hogman_manager.addNode(frame_now, weight, false);
				}
				else if (using_srba_optimizer)
				{
					srba_manager.addNode(frame_now, weight, false);
				}
				else if (using_robust_optimizer)
				{
					robust_manager.addNode(frame_now, weight, false);
				}
				if (using_feature)
				{
					last_is_keyframe = false;
				}
			}
		}
		else
		{
			transformation_matrix.push_back(transformation_matrix[frame_id - 1] * relative_tran);
			if (using_feature)
			{
				last_is_keyframe = false;
			}
		}
		last_transformation = relative_tran;
		last_cloud = cloud_for_registration;
		if (using_icpcuda)
			imgDepth.copyTo(last_depth);
		if (using_feature)
		{
			frame_now->f->buildFlannIndex();
			last_frame = frame_now;
			imgDepth.copyTo(last_depth);
		}
	}
	std::cout << endl;
	frame_id++;
}

void SlamEngine::AddNext(const cv::Mat &imgRGB, const cv::Mat &imgDepth, double timestamp, Eigen::Matrix4f trajectory)
{
	timestamps.push_back(timestamp);
	PointCloudPtr cloud_new = ConvertToPointCloudWithoutMissingData(imgDepth, imgRGB, timestamp, frame_id);
	PointCloudPtr cloud_downsampled = DownSamplingByVoxelGrid(cloud_new, downsample_rate, downsample_rate, downsample_rate);

	point_clouds.push_back(cloud_downsampled);
	transformation_matrix.push_back(trajectory);
	frame_id++;
}

void SlamEngine::Refine()
{
	if (using_robust_optimizer)
	{
		robust_manager.refine();
	}
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
		else if (using_robust_optimizer)
			pcl::transformPointCloud(*point_clouds[i], *tc, robust_manager.getTransformation(i));
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
		else if (using_srba_optimizer)
			ret.push_back(pair<double, Eigen::Matrix4f>(timestamps[i], srba_manager.getTransformation(i)));
		else if (using_robust_optimizer)
			ret.push_back(pair<double, Eigen::Matrix4f>(timestamps[i], robust_manager.getTransformation(i)));
		else
			ret.push_back(pair<double, Eigen::Matrix4f>(timestamps[i], transformation_matrix[i]));
	}
	return ret;
}

void SlamEngine::SaveLogs(ofstream &outfile)
{
#ifdef SAVE_TEST_INFOS
	if (using_srba_optimizer)
	{
		//outfile << "base\ttarget\trmse\tmatches\tinliers\ttransformation" << endl;
		outfile << srba_manager.baseid.size() << endl;
		for (int i = 0; i < srba_manager.baseid.size(); i++)
		{
			outfile << srba_manager.baseid[i] << "\t"
				<< srba_manager.targetid[i] << "\t"
				<< srba_manager.rmses[i] << "\t"
				<< srba_manager.matchescount[i] << "\t"
				<< srba_manager.inlierscount[i] << endl;
			outfile << srba_manager.ransactrans[i] << endl;
		}
	}
	else if (using_hogman_optimizer)
	{
		//outfile << "base\ttarget\trmse\tmatches\tinliers\ttransformation" << endl;
		outfile << hogman_manager.baseid.size() << endl;
		for (int i = 0; i < hogman_manager.baseid.size(); i++)
		{
			outfile << hogman_manager.baseid[i] << "\t"
				<< hogman_manager.targetid[i] << "\t"
				<< hogman_manager.rmses[i] << "\t"
				<< hogman_manager.matchescount[i] << "\t"
				<< hogman_manager.inlierscount[i] << endl;
			outfile << hogman_manager.ransactrans[i] << endl;
		}
	}
	else if (using_robust_optimizer)
	{
		//outfile << "base\ttarget\trmse\tmatches\tinliers\ttransformation" << endl;
		outfile << robust_manager.baseid.size() << endl;
		for (int i = 0; i < robust_manager.baseid.size(); i++)
		{
			outfile << robust_manager.baseid[i] << "\t"
				<< robust_manager.targetid[i] << "\t"
				<< robust_manager.rmses[i] << "\t"
				<< robust_manager.matchescount[i] << "\t"
				<< robust_manager.inlierscount[i] << endl;
			outfile << robust_manager.ransactrans[i] << endl;
		}

		outfile << endl << "line process" << endl;
		vector<int> id0s, id1s;
		vector<float> lineps;
		robust_manager.getLineProcessResult(id0s, id1s, lineps);
		for (int i = 0; i < id0s.size(); i++)
		{
			outfile << id0s[i] << "\t" << id1s[i] << "\t" << lineps[i] << endl;
		}
	}
#endif
}

void SlamEngine::ShowStatistics()
{
	cout << "-------------------------------------------------------------------------------" << endl;
	cout << "-------------------------------------------------------------------------------" << endl;
	cout << "Total runtime         : " << (clock() - total_start) / 1000.0 << endl;
	cout << "Total frames          : " << frame_id << endl;
	cout << "Number of keyframes   : ";
	if (using_hogman_optimizer)
		cout << hogman_manager.keyframeInQuadTreeCount << endl;
	else if (using_srba_optimizer)
		cout << srba_manager.keyframeInQuadTreeCount << endl;
	else if (using_robust_optimizer)
		cout << robust_manager.keyframeInQuadTreeCount << endl;
	else
		cout << 0 << endl;
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
	else if (using_srba_optimizer)
	{
		cout << "Min Closure Time      : " << fixed << setprecision(3) << srba_manager.min_closure_detect_time << ",\t\tMax Closure Time: " << srba_manager.max_closure_detect_time << endl;
		cout << "Avg Closure Time      : " << srba_manager.total_closure_detect_time / srba_manager.clousureCount << endl;
		cout << "Min Closure Candidate : " << srba_manager.min_closure_candidate << "\t\tMax Closure Candidate: " << srba_manager.max_closure_candidate << endl;
		cout << "-------------------------------------------------------------------------------" << endl;
		cout << "Min Graph Time        : " << srba_manager.min_graph_opt_time << "\t\tMax Graph Time: " << srba_manager.max_graph_opt_time << endl;
		cout << "Avg Graph Time        : " << srba_manager.total_graph_opt_time / frame_id << endl;
		cout << "Min Edge Weight       : " << srba_manager.min_edge_weight << "\t\tMax Edge Weight: " << srba_manager.max_edge_weight << endl;
	}
	else if (using_robust_optimizer)
	{
		cout << "Min Closure Time      : " << fixed << setprecision(3) << robust_manager.min_closure_detect_time << ",\t\tMax Closure Time: " << robust_manager.max_closure_detect_time << endl;
		cout << "Avg Closure Time      : " << robust_manager.total_closure_detect_time / robust_manager.clousureCount << endl;
		cout << "Min Closure Candidate : " << robust_manager.min_closure_candidate << "\t\tMax Closure Candidate: " << robust_manager.max_closure_candidate << endl;
		cout << "-------------------------------------------------------------------------------" << endl;
		cout << "Min Graph Time        : " << robust_manager.min_graph_opt_time << "\t\tMax Graph Time: " << robust_manager.max_graph_opt_time << endl;
		cout << "Avg Graph Time        : " << robust_manager.total_graph_opt_time / frame_id << endl;
		cout << "Min Edge Weight       : " << robust_manager.min_edge_weight << "\t\tMax Edge Weight: " << robust_manager.max_edge_weight << endl;
	}
	cout << endl;

	if (using_robust_optimizer)
	{
		if (robust_manager.insertion_failure.size() > 0)
		{
			cout << endl << "QuadTree is not big enough" << endl;
			for (int i = 0; i < robust_manager.insertion_failure.size(); i++)
			{
				cout << robust_manager.insertion_failure[i].first << ", " << robust_manager.insertion_failure[i].second << endl;
			}
		}
	}
}



bool SlamEngine::IsTransformationBigEnough()
{
	if (!using_optimizer)
	{
		return false;
	}
	if (accumulated_frame_count >= Config::instance()->get<int>("max_keyframe_interval"))
	{
		return true;
	}

	Eigen::Vector3f t = TranslationFromMatrix4f(accumulated_transformation);
	float tnorm = t.norm();
	Eigen::Vector3f e = EulerAngleFromQuaternion(QuaternionFromMatrix4f(accumulated_transformation));
	e *= 180.0 / M_PI;
	float max_angle = std::max(fabs(e(0)), std::max(fabs(e(1)), fabs(e(2))));

	cout << ", " << tnorm << ", " << max_angle;

	if (tnorm > Config::instance()->get<float>("min_translation_meter")
		|| max_angle > Config::instance()->get<float>("min_rotation_degree"))
	{
		return true;
	}

	return false;
}