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
	feature_type = "ORB";
	graph_feature_type = "SURF";

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

	last_rational = 1.0;
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

		if (using_icpcuda || using_feature)
		{
			imgDepth.copyTo(last_depth);
		}

		Frame *frame;
		if (using_feature)
		{
			frame = new Frame(imgRGB, imgDepth, feature_type, Eigen::Matrix4f::Identity());
			frame->relative_tran = Eigen::Matrix4f::Identity();
			if (feature_type != "ORB")
				frame->f->buildFlannIndex();
			last_feature_frame = frame;
			last_feature_keyframe = frame;
			last_feature_frame_is_keyframe = true;
		}

		if (using_optimizer)
		{
			frame = new Frame(imgRGB, imgDepth, graph_feature_type, Eigen::Matrix4f::Identity());
			frame->relative_tran = Eigen::Matrix4f::Identity();
			frame->tran = frame->relative_tran;
	
			string inliers, exists;
			bool is_in_quadtree = false;
			if (using_hogman_optimizer)
			{
				is_in_quadtree = hogman_manager.addNode(frame, 1.0, true, &inliers, &exists);
			}
			else if (using_srba_optimizer)
			{
//				is_in_quadtree = srba_manager.addNode(frame, 1.0, true, &inliers, &exists);
			}
			else if (using_robust_optimizer)
			{
				is_in_quadtree = robust_manager.addNode(frame, true);
			}
			if (!is_in_quadtree)
			{
				delete frame->f;
				frame->f = nullptr;
			}

#ifdef SAVE_TEST_INFOS
			keyframe_candidates_id.push_back(frame_id);
			keyframe_candidates.push_back(pair<cv::Mat, cv::Mat>(imgRGB, imgDepth));

			if (last_keyframe_detect_lc)
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
		last_keyframe_transformation = Eigen::Matrix4f::Identity();
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

			Eigen::Matrix4f estimated_tran = Eigen::Matrix4f::Identity();
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

		bool isKeyframe = false;
		bool ransac_failed = false;

		Frame *f_frame;
		if (using_feature)
		{
			f_frame = new Frame(imgRGB, imgDepth, feature_type, Eigen::Matrix4f::Identity());
			
			vector<cv::DMatch> matches, inliers;
			Eigen::Matrix4f tran;
			Eigen::Matrix<double, 6, 6> information;
			float rmse;
			int pc, pcorrc;

			if (feature_type == "ORB")
				last_feature_frame->f->findMatchedPairsBruteForce(matches, f_frame->f);
			else
				last_feature_frame->f->findMatchedPairs(matches, f_frame->f);
			if (Feature::getTransformationByRANSAC(tran, information, pc, pcorrc, rmse, &inliers,
				last_feature_frame->f, f_frame->f, nullptr, matches))
			{
				relative_tran = tran;
				cout << ", " << matches.size() << ", " << inliers.size();

				matches.clear();
				inliers.clear();
				if (feature_type == "ORB")
					last_feature_keyframe->f->findMatchedPairsBruteForce(matches, f_frame->f);
				else
					last_feature_keyframe->f->findMatchedPairs(matches, f_frame->f);

				Feature::computeInliersAndError(inliers, rmse, nullptr, matches,
					accumulated_transformation * relative_tran,
					last_feature_keyframe->f, f_frame->f);

// 				if (Feature::getTransformationByRANSAC(tran, information, coresp, rmse, &inliers,
// 					last_feature_keyframe->f, f_frame->f, nullptr, matches))
// 				{
					float rrr = (float)inliers.size() / matches.size();
					if (last_feature_frame_is_keyframe)
					{
						last_rational = rrr;
					}
					rrr /= last_rational;
					if (rrr < Config::instance()->get<float>("keyframe_rational"))
					{
						isKeyframe = true;
					}
					cout << ", " << rrr;
// 				}
// 				else
// 				{
// 					cout << ", failed";
// 					isKeyframe = true;
// 				}
			}
			else
			{
				ransac_failed = true;
				ransac_failed_frames.push_back(frame_id);
				isKeyframe = true;

				icpcuda->initICPModel((unsigned short *)last_depth.data, 20.0f, Eigen::Matrix4f::Identity());
				icpcuda->initICP((unsigned short *)imgDepth.data, 20.0f);

				Eigen::Matrix4f tran2;
				Eigen::Vector3f t = tran2.topRightCorner(3, 1);
				Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rot = tran2.topLeftCorner(3, 3);

				Eigen::Matrix4f estimated_tran = Eigen::Matrix4f::Identity();
				Eigen::Vector3f estimated_t = estimated_tran.topRightCorner(3, 1);
				Eigen::Matrix<float, 3, 3, Eigen::RowMajor> estimated_rot = estimated_tran.topLeftCorner(3, 3);

				icpcuda->getIncrementalTransformation(t, rot, estimated_t, estimated_rot, threads, blocks);

				tran2.topLeftCorner(3, 3) = rot;
				tran2.topRightCorner(3, 1) = t;

				relative_tran = tran2;
			}
		}

		last_transformation = relative_tran;
		accumulated_frame_count++;
		accumulated_transformation = accumulated_transformation * relative_tran;
		relative_tran = accumulated_transformation;
		if (accumulated_frame_count >= Config::instance()->get<int>("max_keyframe_interval"))
		{
			isKeyframe = true;
		}

		if (isKeyframe)
		{
			accumulated_frame_count = 0;
			accumulated_transformation = Eigen::Matrix4f::Identity();
		}

		Frame *g_frame;
		if (using_optimizer)
		{
/*			step_start = clock();*/
			if (using_hogman_optimizer)
			{
				global_tran = hogman_manager.getLastKeyframeTransformation() * relative_tran;
			}
			else if (using_srba_optimizer)
			{
//				global_tran = srba_manager.getLastKeyframeTransformation() * relative_tran;
			}
			else if (using_robust_optimizer)
			{
				global_tran = robust_manager.getLastKeyframeTransformation() * relative_tran;
			}
			if (isKeyframe)
			{
				g_frame = new Frame(imgRGB, imgDepth, graph_feature_type, global_tran);
				g_frame->relative_tran = relative_tran;
				g_frame->tran = global_tran;
				g_frame->ransac_failed = ransac_failed;

				string inliers, exists;
				bool is_in_quadtree = false;
				if (using_hogman_optimizer)
				{
					is_in_quadtree = hogman_manager.addNode(g_frame, weight, true, &inliers, &exists);
				}
				else if (using_srba_optimizer)
				{
//					is_in_quadtree = srba_manager.addNode(g_frame, weight, true, &inliers, &exists);
				}
				else if (using_robust_optimizer)
				{
					is_in_quadtree = robust_manager.addNode(g_frame, true);
				}

				last_keyframe_detect_lc = is_in_quadtree;
				if (!is_in_quadtree)
				{
					delete g_frame->f;
					g_frame->f = nullptr;
				}
					

// #ifdef SAVE_TEST_INFOS
// 				keyframe_candidates_id.push_back(frame_id);
// 				keyframe_candidates.push_back(pair<cv::Mat, cv::Mat>(imgRGB, imgDepth));
// 
// 				if (isKeyframe)
// 				{
// 					keyframes_id.push_back(frame_id);
// 					keyframes.push_back(pair<cv::Mat, cv::Mat>(imgRGB, imgDepth));
// 					keyframes_inliers_sig.push_back(inliers);
// 					keyframes_exists_sig.push_back(exists);
// 				}
// #endif
			}
			else
			{
				g_frame = new Frame();
				g_frame->relative_tran = relative_tran;
				g_frame->tran = global_tran;

				if (using_hogman_optimizer)
				{
					hogman_manager.addNode(g_frame, weight, false);
				}
				else if (using_srba_optimizer)
				{
//					srba_manager.addNode(g_frame, weight, false);
				}
				else if (using_robust_optimizer)
				{
					robust_manager.addNode(g_frame, false);
				}
			}
			
// 			step_time = (clock() - step_start) / 1000.0;
// 			std::cout << endl;
// 			std::cout << "Feature: " << fixed << setprecision(3) << step_time;
		}
		else
		{
			transformation_matrix.push_back(last_keyframe_transformation * relative_tran);
			if (isKeyframe)
				last_keyframe_transformation = last_keyframe_transformation * relative_tran;
		}

		last_cloud = cloud_for_registration;
		if (using_icpcuda || using_feature)
			imgDepth.copyTo(last_depth);
		if (using_feature)
		{
			if (!last_feature_frame_is_keyframe)
			{
				delete last_feature_frame->f;
				last_feature_frame->f = nullptr;
			}
				
			if (feature_type != "ORB")
				f_frame->f->buildFlannIndex();
			last_feature_frame = f_frame;

			if (isKeyframe)
			{
				delete last_feature_keyframe->f;
				last_feature_keyframe->f = nullptr;
				last_feature_keyframe = f_frame;
				last_feature_frame_is_keyframe = true;
			}
			else
			{
				last_feature_frame_is_keyframe = false;
			}
		}
	}
	std::cout << endl;
	frame_id++;
}

void SlamEngine::AddGraph(Frame *frame, PointCloudPtr cloud, bool keyframe, double timestamp)
{
	timestamps.push_back(timestamp);
	point_clouds.push_back(cloud);
	
	if (frame_id == 0)
	{
		cout << "Frame " << frame_id;
		frame->tran = frame->relative_tran;

// 		// test
// 		vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> d;
// 		positions.push_back(d);
// 		// test end
		last_keyframe_detect_lc = robust_manager.addNode(frame, true);

		if (!last_keyframe_detect_lc)
		{
			delete frame->f;
			frame->f = nullptr;
		}
		else
		{
			id_detect_lc.push_back(0);
		}
		std::cout << endl;
	}
	else
	{
		frame->tran = robust_manager.getLastKeyframeTransformation() * frame->relative_tran;
		if (keyframe)
		{
			cout << "Frame " << frame_id;
			//frame->f->updateFeaturePoints3DReal(frame->tran);

			// test
// 			vector<Frame *> graph = robust_manager.getGraph();
// 			vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>  d;
// 			for (int i = 0; i < robust_manager.keyframe_indices.size(); i++)
// 			{
// 				Eigen::Vector3f translation = TranslationFromMatrix4f(graph[robust_manager.keyframe_indices[i]]->tran);
// 				d.push_back(translation);
// 			}
// 			current_positions.push_back(TranslationFromMatrix4f(frame->tran));
// 			positions.push_back(d);

			// test-end

			last_keyframe_detect_lc = robust_manager.addNode(frame, true);
			if (!last_keyframe_detect_lc)
			{
				delete frame->f;
				frame->f = nullptr;
			}
// 			// test
// 			else
// 			{
// 				id_detect_lc.push_back(frame_id);
// 			}
// 			// test end

			std::cout << endl;
		}
		else
		{
			robust_manager.addNode(frame, false);
		}
	}
	frame_id++;
}

void SlamEngine::AddGraph(Frame *frame, PointCloudPtr cloud, bool keyframe, bool quad, vector<int> &loop, double timestamp)
{
	timestamps.push_back(timestamp);
	point_clouds.push_back(cloud);
	cout << "Frame " << frame_id;
	if (frame_id == 0)
	{
//		robust_manager.active_window.build(0.0, 0.0, Config::instance()->get<float>("quadtree_size"), 4);
		frame->tran = frame->relative_tran;
		string inliers, exists;
//		last_keyframe_in_quadtree = robust_manager.addNode(frame, quad, &loop, true);

		if (!last_keyframe_detect_lc)
		{
			delete frame->f;
			frame->f = nullptr;
		}

		// #ifdef SAVE_TEST_INFOS
		// 			keyframe_candidates_id.push_back(frame_id);
		// 			keyframe_candidates.push_back(pair<cv::Mat, cv::Mat>(imgRGB, imgDepth));
		// 
		// 			if (isKeyframe)
		// 			{
		// 				keyframes_id.push_back(frame_id);
		// 				keyframes.push_back(pair<cv::Mat, cv::Mat>(imgRGB, imgDepth));
		// 				keyframes_inliers_sig.push_back(inliers);
		// 				keyframes_exists_sig.push_back(exists);
		// 			}
		// #endif
	}
	else
	{
		frame->tran = robust_manager.getLastKeyframeTransformation() * frame->relative_tran;
		if (keyframe)
		{
			frame->f->updateFeaturePoints3DReal(frame->tran);
//			last_keyframe_in_quadtree = robust_manager.addNode(frame, quad, &loop, true);
			if (!last_keyframe_detect_lc)
			{
				delete frame->f;
				frame->f = nullptr;
			}
		}
		else
		{
//			robust_manager.addNode(frame, quad, &loop, false);
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
//		else if (using_srba_optimizer)
//			pcl::transformPointCloud(*point_clouds[i], *tc, srba_manager.getTransformation(i));
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
//		else if (using_srba_optimizer)
//			ret.push_back(pair<double, Eigen::Matrix4f>(timestamps[i], srba_manager.getTransformation(i)));
		else if (using_robust_optimizer)
			ret.push_back(pair<double, Eigen::Matrix4f>(timestamps[i], robust_manager.getTransformation(i)));
		else
			ret.push_back(pair<double, Eigen::Matrix4f>(timestamps[i], transformation_matrix[i]));
	}
	return ret;
}

vector<pair<int, int>> SlamEngine::GetLoop()
{
	vector<pair<int, int>> ret;
	if (using_robust_optimizer)
	{
		for (int i = 0; i < robust_manager.loop_edges.size(); i++)
		{
			ret.push_back(pair<int, int>(robust_manager.keyframe_indices[robust_manager.loop_edges[i].id0],
				robust_manager.keyframe_indices[robust_manager.loop_edges[i].id1]));
		}
	}
	return ret;
}

void SlamEngine::SaveLogs(ofstream &outfile)
{
#ifdef SAVE_TEST_INFOS
	if (using_srba_optimizer)
	{
		//outfile << "base\ttarget\trmse\tmatches\tinliers\ttransformation" << endl;
// 		outfile << srba_manager.baseid.size() << endl;
// 		for (int i = 0; i < srba_manager.baseid.size(); i++)
// 		{
// 			outfile << srba_manager.baseid[i] << "\t"
// 				<< srba_manager.targetid[i] << "\t"
// 				<< srba_manager.rmses[i] << "\t"
// 				<< srba_manager.matchescount[i] << "\t"
// 				<< srba_manager.inlierscount[i] << endl;
// 			outfile << srba_manager.ransactrans[i] << endl;
// 		}
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
//	else if (using_srba_optimizer)
//		cout << srba_manager.keyframeInQuadTreeCount << endl;
	else if (using_robust_optimizer)
		cout << robust_manager.keyframe_for_lc.size() << endl;
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
		cout << "Min Closure Time      : " << fixed << setprecision(3) << hogman_manager.min_closure_detect_time << ",\t\tMax Closure Time: " << hogman_manager.max_closure_detect_time << endl;
		cout << "Avg Closure Time      : " << hogman_manager.total_closure_detect_time / hogman_manager.clousureCount << endl;
		cout << "Min Closure Candidate : " << hogman_manager.min_closure_candidate << "\t\tMax Closure Candidate: " << hogman_manager.max_closure_candidate << endl;
		cout << "-------------------------------------------------------------------------------" << endl;
		cout << "Min Graph Time        : " << hogman_manager.min_graph_opt_time << "\t\tMax Graph Time: " << hogman_manager.max_graph_opt_time << endl;
		cout << "Avg Graph Time        : " << hogman_manager.total_graph_opt_time / frame_id << endl;
		cout << "Min Edge Weight       : " << hogman_manager.min_edge_weight << "\t\tMax Edge Weight: " << hogman_manager.max_edge_weight << endl;
	}
	else if (using_srba_optimizer)
	{
// 		cout << "Min Closure Time      : " << fixed << setprecision(3) << srba_manager.min_closure_detect_time << ",\t\tMax Closure Time: " << srba_manager.max_closure_detect_time << endl;
// 		cout << "Avg Closure Time      : " << srba_manager.total_closure_detect_time / srba_manager.clousureCount << endl;
// 		cout << "Min Closure Candidate : " << srba_manager.min_closure_candidate << "\t\tMax Closure Candidate: " << srba_manager.max_closure_candidate << endl;
// 		cout << "-------------------------------------------------------------------------------" << endl;
// 		cout << "Min Graph Time        : " << srba_manager.min_graph_opt_time << "\t\tMax Graph Time: " << srba_manager.max_graph_opt_time << endl;
// 		cout << "Avg Graph Time        : " << srba_manager.total_graph_opt_time / frame_id << endl;
// 		cout << "Min Edge Weight       : " << srba_manager.min_edge_weight << "\t\tMax Edge Weight: " << srba_manager.max_edge_weight << endl;
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

	if (ransac_failed_frames.size() > 0)
	{
		cout << "ransac failed: " << endl;
		for (int i = 0; i < ransac_failed_frames.size(); i++)
		{
			cout << ransac_failed_frames[i] << endl;
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