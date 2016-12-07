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

	int min_matches = Config::instance()->get<int>("min_matches");
	float inlier_percentage = Config::instance()->get<float>("min_inlier_p");
	float inlier_dist = Config::instance()->get<float>("max_inlier_dist");

	pairwise_register_type = Config::instance()->get<std::string>("feature_type");
	if (pairwise_register_type == "sift")
		pairwise_register = new SiftRegister(min_matches, inlier_percentage, inlier_dist);
	else if (pairwise_register_type == "surf")
		pairwise_register = new SurfRegister(min_matches, inlier_percentage, inlier_dist);
	else if (pairwise_register_type == "orb")
		pairwise_register = new OrbRegister(min_matches, inlier_percentage, inlier_dist);

	using_second_register = false;
	pairwise_register_2 = nullptr;

	graph_register_type = Config::instance()->get<std::string>("graph_feature_type");
	graph_manager_type = "robust";
	if (graph_manager_type == "robust")
	{
		graph_manager = new RobustManager(true);
		void *params[3];
		params[0] = static_cast<void *>(&min_matches);
		params[1] = static_cast<void *>(&inlier_percentage);
		params[2] = static_cast<void *>(&inlier_dist);
		graph_manager->setParameters(params);
	}

	icpcuda = nullptr;

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

void SlamEngine::setPairwiseRegister(string type)
{
	if (pairwise_register)
		delete pairwise_register;
	pairwise_register = nullptr;

	transform(type.begin(), type.end(), type.begin(), tolower);
	pairwise_register_type = type;
	if (type == "sift")
	{
		pairwise_register = new SiftRegister();
	}
	else if (type == "surf")
	{
		pairwise_register = new SurfRegister();
	}
	else if (type == "orb")
	{
		pairwise_register = new OrbRegister();
	}
	else if (type == "icpcuda")
	{
		pairwise_register = new IcpcudaRegister();
	}
}

void SlamEngine::setSecondPairwiseRegister(string type)
{
	if (pairwise_register_2)
		delete pairwise_register_2;
	pairwise_register_2 = nullptr;

	transform(type.begin(), type.end(), type.begin(), tolower);
	pairwise_register_type_2 = type;
	if (type == "sift")
	{
		pairwise_register_2 = new SiftRegister();
	}
	else if (type == "surf")
	{
		pairwise_register_2 = new SurfRegister();
	}
	else if (type == "orb")
	{
		pairwise_register_2 = new OrbRegister();
	}
	else if (type == "icpcuda")
	{
		pairwise_register_2 = new IcpcudaRegister();
	}
	using_second_register = true;
}

void SlamEngine::setGraphRegister(string type)
{
	transform(type.begin(), type.end(), type.begin(), tolower);
	graph_register_type = type;
}

void SlamEngine::setPairwiseParametersFeature(int min_matches, float inlier_percentage, float inlier_dist)
{
	if (pairwise_register_type != "sift" && pairwise_register_type != "surf" && pairwise_register_type != "orb")
	{
		return;
	}

	void *params[3];
	params[0] = static_cast<void *>(&min_matches);
	params[1] = static_cast<void *>(&inlier_percentage);
	params[2] = static_cast<void *>(&inlier_dist);
	
	pairwise_register->setParameters(params);
}

void SlamEngine::setPairwiseParametersIcpcuda(float dist, float angle, int threads, int blocks)
{
	if (pairwise_register_type != "icpcuda")
	{
		return;
	}

	if (icpcuda)
		delete icpcuda;
	icpcuda = nullptr;

	float depthCutOff = 20.f;

	int width = Config::instance()->get<int>("image_width");
	int height = Config::instance()->get<int>("image_height");
	float cx = Config::instance()->get<float>("camera_cx");
	float cy = Config::instance()->get<float>("camera_cy");
	float fx = Config::instance()->get<float>("camera_fx");
	float fy = Config::instance()->get<float>("camera_fy");
	float depthFactor = Config::instance()->get<float>("depth_factor");
	icpcuda = new ICPOdometry(width, height, cx, cy, fx, fy, depthFactor, dist, angle);

	void *params[4];
	params[0] = static_cast<void *>(icpcuda);
	params[1] = static_cast<void *>(&threads);
	params[2] = static_cast<void *>(&blocks);
	params[3] = static_cast<void *>(&depthCutOff);
	pairwise_register->setParameters(params);
}

void SlamEngine::setSecondPairwiseParametersFeature(int min_matches, float inlier_percentage, float inlier_dist)
{
	if (pairwise_register_type_2 != "sift" && pairwise_register_type_2 != "surf" && pairwise_register_type_2 != "orb")
	{
		return;
	}

	void *params[3];
	params[0] = static_cast<void *>(&min_matches);
	params[1] = static_cast<void *>(&inlier_percentage);
	params[2] = static_cast<void *>(&inlier_dist);

	pairwise_register_2->setParameters(params);
}

void SlamEngine::setSecondPairwiseParametersIcpcuda(float dist, float angle, int threads, int blocks)
{
	if (pairwise_register_type_2 != "icpcuda")
	{
		return;
	}

	if (icpcuda)
		delete icpcuda;
	icpcuda = nullptr;

	float depthCutOff = 20.f;

	int width = Config::instance()->get<int>("image_width");
	int height = Config::instance()->get<int>("image_height");
	float cx = Config::instance()->get<float>("camera_cx");
	float cy = Config::instance()->get<float>("camera_cy");
	float fx = Config::instance()->get<float>("camera_fx");
	float fy = Config::instance()->get<float>("camera_fy");
	float depthFactor = Config::instance()->get<float>("depth_factor");
	icpcuda = new ICPOdometry(width, height, cx, cy, fx, fy, depthFactor, dist, angle);

	void *params[4];
	params[0] = static_cast<void *>(icpcuda);
	params[1] = static_cast<void *>(&threads);
	params[2] = static_cast<void *>(&blocks);
	params[3] = static_cast<void *>(&depthCutOff);
	pairwise_register_2->setParameters(params);
}

void SlamEngine::setGraphManager(string type)
{
	transform(type.begin(), type.end(), type.begin(), tolower);
	graph_manager_type = type;
	if (type == "robust")
	{
		graph_manager = new RobustManager(true);
		using_optimizer = true;
	}
	else
	{
		graph_manager = NULL;
		using_optimizer = false;
	}
}

void SlamEngine::setGraphParametersFeature(int min_matches, float inlier_percentage, float inlier_dist)
{
	if (graph_register_type != "sift" && graph_register_type != "surf" && graph_register_type != "orb")
	{
		return;
	}

	void *params[3];
	params[0] = static_cast<void *>(&min_matches);
	params[1] = static_cast<void *>(&inlier_percentage);
	params[2] = static_cast<void *>(&inlier_dist);

	graph_manager->setParameters(params);
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
		point_clouds.push_back(cloud_downsampled);

		if (pairwise_register_type == "icpcuda" || pairwise_register_type_2 == "icpcuda")
		{
			imgDepth.copyTo(last_depth);
		}

		Frame *f_frame, *g_frame;
		if (pairwise_register_type == "sift" || pairwise_register_type == "surf" || pairwise_register_type == "orb")
		{
			f_frame = new Frame(imgRGB, imgDepth, pairwise_register_type, Eigen::Matrix4f::Identity());
			f_frame->relative_tran = Eigen::Matrix4f::Identity();
			if (pairwise_register_type != "orb")
				f_frame->f->buildFlannIndex();
			last_frame = f_frame;
			last_keyframe = f_frame;
			is_last_frame_keyframe = true;
		}

		if (using_optimizer)
		{
			if (graph_register_type == "sift" || graph_register_type == "surf" || graph_register_type == "orb")
			{
				if (graph_register_type != pairwise_register_type)
				{
					g_frame = new Frame(imgRGB, imgDepth, graph_register_type, Eigen::Matrix4f::Identity());
					g_frame->relative_tran = Eigen::Matrix4f::Identity();
				}
				else
				{
					g_frame = f_frame;
				}
				g_frame->tran = f_frame->relative_tran;
				bool is_candidate = graph_manager->addNode(g_frame, true);

				is_last_frame_candidate = is_candidate;
				is_last_keyframe_candidate = is_candidate;

				if (pairwise_register_type != graph_register_type && !is_candidate)
				{
					delete g_frame->f;
					g_frame->f = nullptr;
				}
			}

#ifdef SAVE_TEST_INFOS
			keyframe_candidates_id.push_back(frame_id);
			keyframe_candidates.push_back(pair<cv::Mat, cv::Mat>(imgRGB, imgDepth));

			if (is_last_frame_candidate)
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
		last_tran = Eigen::Matrix4f::Identity();
		last_keyframe_tran = Eigen::Matrix4f::Identity();
	}
	else
	{
		PointCloudPtr cloud_for_registration = cloud_new;
		PointCloudPtr cloud_transformed(new PointCloudT);
		Eigen::Matrix4f relative_tran = Eigen::Matrix4f::Identity();
		Eigen::Matrix4f global_tran = Eigen::Matrix4f::Identity();
		bool isKeyframe = false;
		bool is_candidate = false;
		bool ransac_failed = false;
		Frame *f_frame;

		point_clouds.push_back(cloud_downsampled);

		if (pairwise_register_type == "icpcuda")
		{
			pairwise_register->getTransformation(last_depth.data, imgDepth.data, relative_tran);
		}

		if (pairwise_register_type == "sift" || pairwise_register_type == "surf" || pairwise_register_type == "orb")
		{
			f_frame = new Frame(imgRGB, imgDepth, pairwise_register_type, Eigen::Matrix4f::Identity());
			Eigen::Matrix4f tran;
			if (pairwise_register->getTransformation(last_frame, f_frame, tran))
			{
				relative_tran = tran;
				Eigen::Matrix4f estimated_tran = accumulated_transformation * relative_tran;
				float rrr = pairwise_register->getCorrespondencePercent(last_keyframe, f_frame, estimated_tran);
				if (is_last_frame_keyframe)
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

				if (using_second_register)
				{
					Eigen::Matrix4f tran2 = Eigen::Matrix4f::Identity();
					if (pairwise_register_type_2 == "icpcuda")
					{
						pairwise_register_2->getTransformation(last_depth.data, imgDepth.data, tran2);
					}
					relative_tran = tran2;
				}
				else
				{
					relative_tran = last_tran;
				}
			}
		}

		last_tran = relative_tran;
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
			global_tran = graph_manager->getLastKeyframeTransformation() * relative_tran;
			if (isKeyframe)
			{
				if (pairwise_register_type != graph_register_type)
				{
					g_frame = new Frame(imgRGB, imgDepth, graph_register_type, global_tran);
				}
				else
				{
					g_frame = f_frame;
				}
				g_frame->relative_tran = relative_tran;
				g_frame->tran = global_tran;
				g_frame->ransac_failed = ransac_failed;

				is_candidate = graph_manager->addNode(g_frame, true);

				if (graph_register_type != pairwise_register_type && !is_candidate)
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

				graph_manager->addNode(g_frame, false);
			}
		}
		else
		{
			transformation_matrix.push_back(last_keyframe_tran * relative_tran);
			if (isKeyframe)
				last_keyframe_tran = last_keyframe_tran * relative_tran;
		}

		if (pairwise_register_type == "icpcuda" || pairwise_register_type_2 == "icpcuda")
			imgDepth.copyTo(last_depth);

		if (pairwise_register_type == "sift" || pairwise_register_type == "surf" || pairwise_register_type == "orb")
		{
			if (!is_last_frame_keyframe)
			{
				if (pairwise_register_type != graph_register_type)
				{
					delete last_frame;
					last_frame = nullptr;
				}
				else
				{
					delete last_frame->f;
					last_frame->f = nullptr;
				}
			}
				
			if (pairwise_register_type != "orb")
				f_frame->f->buildFlannIndex();
			last_frame = f_frame;

			if (isKeyframe)
			{
				if (pairwise_register_type != graph_register_type)
				{
					delete last_keyframe;
					last_keyframe = nullptr;
				}
				else if (!is_last_keyframe_candidate)
				{
					delete last_keyframe->f;
					last_keyframe->f = nullptr;
				}
				last_keyframe = f_frame;
				is_last_frame_keyframe = true;
				is_last_keyframe_candidate = is_candidate;
			}
			else
			{
				is_last_frame_keyframe = false;
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
		bool is_candidate = graph_manager->addNode(frame, true);
// 		if (!is_candidate)
// 		{
// 			delete frame->f;
// 			frame->f = nullptr;
// 		}
		std::cout << endl;
	}
	else
	{
		frame->tran = graph_manager->getLastKeyframeTransformation() * frame->relative_tran;
		if (keyframe)
		{
			cout << "Frame " << frame_id;
			bool is_candidate = graph_manager->addNode(frame, true);
// 			if (!is_candidate)
// 			{
// 				delete frame->f;
// 				frame->f = nullptr;
// 			}
			std::cout << endl;
		}
		else
		{
			graph_manager->addNode(frame, false);
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
	}
	else
	{
		frame->tran = graph_manager->getLastKeyframeTransformation() * frame->relative_tran;
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
		if (graph_manager_type == "robust")
			pcl::transformPointCloud(*point_clouds[i], *tc, graph_manager->getTransformation(i));
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
		if (using_optimizer)
			ret.push_back(pair<double, Eigen::Matrix4f>(timestamps[i], graph_manager->getTransformation(i)));
		else
			ret.push_back(pair<double, Eigen::Matrix4f>(timestamps[i], transformation_matrix[i]));
	}
	return ret;
}

void SlamEngine::SaveLogs(ofstream &outfile)
{
#ifdef SAVE_TEST_INFOS
	if (graph_manager_type == "robust")
	{
		RobustManager *robust_manager = static_cast<RobustManager *>(graph_manager);
		//outfile << "base\ttarget\trmse\tmatches\tinliers\ttransformation" << endl;
		outfile << robust_manager->baseid.size() << endl;
		for (int i = 0; i < robust_manager->baseid.size(); i++)
		{
			outfile << robust_manager->baseid[i] << "\t"
				<< robust_manager->targetid[i] << "\t"
				<< robust_manager->rmses[i] << "\t"
				<< robust_manager->matchescount[i] << "\t"
				<< robust_manager->inlierscount[i] << endl;
			outfile << robust_manager->ransactrans[i] << endl;
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
	if (graph_manager_type == "robust")
		cout << ((RobustManager *)graph_manager)->keyframe_for_lc.size() << endl;
	else
		cout << 0 << endl;
	cout << "-------------------------------------------------------------------------------" << endl;
	cout << "Min ftof Time          : " << min_ftof_time << "\t\tMax ftof Time: " << max_ftof_time << endl;
	cout << "Avg ftof Time          : " << total_ftof_time / frame_id << endl;
	cout << "-------------------------------------------------------------------------------" << endl;
	if (graph_manager_type == "robust")
	{
		cout << "Min Closure Time      : " << fixed << setprecision(3) << ((RobustManager *)graph_manager)->min_lc_detect_time << ",\t\tMax Closure Time: " << ((RobustManager *)graph_manager)->max_lc_detect_time << endl;
		cout << "Avg Closure Time      : " << ((RobustManager *)graph_manager)->total_lc_detect_time / ((RobustManager *)graph_manager)->clousureCount << endl;
		cout << "-------------------------------------------------------------------------------" << endl;
		cout << "Min Graph Time        : " << ((RobustManager *)graph_manager)->min_graph_opt_time << "\t\tMax Graph Time: " << ((RobustManager *)graph_manager)->max_graph_opt_time << endl;
		cout << "Avg Graph Time        : " << ((RobustManager *)graph_manager)->total_graph_opt_time / frame_id << endl;
	}
	cout << endl;
}
