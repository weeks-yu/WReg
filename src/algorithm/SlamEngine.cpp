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

	downsample_rate = Config::instance()->get<float>("downsample_rate");

	using_optimizer = true;
	using_robust_optimizer = true;
	feature_type = Config::instance()->get<std::string>("feature_type");
	min_matches = Config::instance()->get<int>("min_matches");
	inlier_percentage = Config::instance()->get<float>("min_inlier_p");
	inlier_dist = Config::instance()->get<float>("max_inlier_dist");

	graph_feature_type = Config::instance()->get<std::string>("graph_feature_type");

	using_icpcuda = true;
	icpcuda = nullptr;
	threads = Config::instance()->get<int>("icpcuda_threads");
	blocks = Config::instance()->get<int>("icpcuda_blocks");

	// statistics
	min_ftof_time = numeric_limits<float>::max();
	max_ftof_time = 0;
	total_ftof_time = 0;

	accumulated_frame_count = 0;
	accumulated_transformation = Eigen::Matrix4f::Identity();

	last_rational = 1.0;
}

SlamEngine::~SlamEngine()
{
	if (icpcuda)
	{
		delete icpcuda;
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
		point_clouds.push_back(cloud_downsampled);

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
	
			bool is_in_quadtree = false;
			if (using_robust_optimizer)
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
		clock_t step_start = 0;
		float step_time;

		point_clouds.push_back(cloud_downsampled);

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
			if (step_time < min_ftof_time) min_ftof_time = step_time;
			if (step_time > max_ftof_time) max_ftof_time = step_time;
			total_ftof_time += step_time;
			std::cout << ", icpcuda time: " << fixed << setprecision(3) << step_time;

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
			if (Feature::getTransformationByRANSAC(tran, rmse, &inliers,
				last_feature_frame->f, f_frame->f, matches, min_matches, inlier_percentage, inlier_dist))
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
					last_feature_keyframe->f, f_frame->f, inlier_dist * inlier_dist);

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
			}
			else
			{
				ransac_failed = true;
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
			if (using_robust_optimizer)
			{
				global_tran = robust_manager.getLastKeyframeTransformation() * relative_tran;
			}
			if (isKeyframe)
			{
				g_frame = new Frame(imgRGB, imgDepth, graph_feature_type, global_tran);
				g_frame->relative_tran = relative_tran;
				g_frame->tran = global_tran;
				g_frame->ransac_failed = ransac_failed;

				bool is_in_quadtree = false;
				if (using_robust_optimizer)
				{
					is_in_quadtree = robust_manager.addNode(g_frame, true);
				}

				last_keyframe_detect_lc = is_in_quadtree;
				if (!is_in_quadtree)
				{
					delete g_frame->f;
					g_frame->f = nullptr;
				}
					

#ifdef SAVE_TEST_INFOS
				keyframe_candidates_id.push_back(frame_id);
				keyframe_candidates.push_back(pair<cv::Mat, cv::Mat>(imgRGB, imgDepth));

				if (isKeyframe)
				{
					keyframes_id.push_back(frame_id);
					keyframes.push_back(pair<cv::Mat, cv::Mat>(imgRGB, imgDepth));
				}
#endif
			}
			else
			{
				g_frame = new Frame();
				g_frame->relative_tran = relative_tran;
				g_frame->tran = global_tran;

				if (using_robust_optimizer)
				{
					robust_manager.addNode(g_frame, false);
				}
			}
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
		last_keyframe_detect_lc = robust_manager.addNode(frame, true);
		if (!last_keyframe_detect_lc)
		{
			delete frame->f;
			frame->f = nullptr;
		}
		std::cout << endl;
	}
	else
	{
		frame->tran = robust_manager.getLastKeyframeTransformation() * frame->relative_tran;
		if (keyframe)
		{
			cout << "Frame " << frame_id;
			last_keyframe_detect_lc = robust_manager.addNode(frame, true);
			if (!last_keyframe_detect_lc)
			{
				delete frame->f;
				frame->f = nullptr;
			}
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
		frame->tran = frame->relative_tran;
		if (!last_keyframe_detect_lc)
		{
			delete frame->f;
			frame->f = nullptr;
		}
	}
	else
	{
		frame->tran = robust_manager.getLastKeyframeTransformation() * frame->relative_tran;
		if (keyframe)
		{
			if (!last_keyframe_detect_lc)
			{
				delete frame->f;
				frame->f = nullptr;
			}
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

PointCloudPtr SlamEngine::GetScene()
{
	PointCloudPtr cloud(new PointCloudT);
	cloud_temp.clear();
	for (int i = 0; i < frame_id; i++)
	{
		PointCloudPtr tc(new PointCloudT);
		if (using_robust_optimizer)
			pcl::transformPointCloud(*point_clouds[i], *tc, robust_manager.getTransformation(i));
		else
			pcl::transformPointCloud(*point_clouds[i], *tc, transformation_matrix[i]);
		
		*cloud += *tc;
		if ((i + 1) % 30 == 0 || i == frame_id - 1)
		{
			cloud = DownSamplingByVoxelGrid(cloud, downsample_rate, downsample_rate, downsample_rate);
			cloud_temp.push_back(cloud);
		}
	}
	cloud = PointCloudPtr(new PointCloudT);
	for (int i = 0; i < frame_id / 30 + 1; i++)
	{
		*cloud += *(cloud_temp[i]);
	}
	cloud = DownSamplingByVoxelGrid(cloud, downsample_rate, downsample_rate, downsample_rate);
	return cloud;
}

vector<pair<double, Eigen::Matrix4f>> SlamEngine::GetTransformations()
{
	vector<pair<double, Eigen::Matrix4f>> ret;

	for (int i = 0; i < frame_id; i++)
	{
		if (using_robust_optimizer)
			ret.push_back(pair<double, Eigen::Matrix4f>(timestamps[i], robust_manager.getTransformation(i)));
		else
			ret.push_back(pair<double, Eigen::Matrix4f>(timestamps[i], transformation_matrix[i]));
	}
	return ret;
}

void SlamEngine::SaveLogs(ofstream &outfile)
{
#ifdef SAVE_TEST_INFOS
	if (using_robust_optimizer)
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
	if (using_robust_optimizer)
		cout << robust_manager.keyframe_for_lc.size() << endl;
	else
		cout << 0 << endl;
	cout << "-------------------------------------------------------------------------------" << endl;
	cout << "Min ftof Time          : " << min_ftof_time << "\t\tMax ftof Time: " << max_ftof_time << endl;
	cout << "Avg ftof Time          : " << total_ftof_time / frame_id << endl;
	cout << "-------------------------------------------------------------------------------" << endl;
	if (using_robust_optimizer)
	{
		cout << "Min Closure Time      : " << fixed << setprecision(3) << robust_manager.min_lc_detect_time << ",\t\tMax Closure Time: " << robust_manager.max_lc_detect_time << endl;
		cout << "Avg Closure Time      : " << robust_manager.total_lc_detect_time / robust_manager.clousureCount << endl;
		cout << "-------------------------------------------------------------------------------" << endl;
		cout << "Min Graph Time        : " << robust_manager.min_graph_opt_time << "\t\tMax Graph Time: " << robust_manager.max_graph_opt_time << endl;
		cout << "Avg Graph Time        : " << robust_manager.total_graph_opt_time / frame_id << endl;
	}
	cout << endl;
}
