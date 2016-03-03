#include "SrbaManager.h"
#include "Transformation.h"
#include <iomanip>

SrbaManager::SrbaManager()
{
	/*int numLevels = Config::instance()->get<int>("graph_levels");*/
	rba.setVerbosityLevel(1);   // 0: None; 1:Important only; 2:Verbose

	rba.parameters.srba.use_robust_kernel = true;
	rba.parameters.obs_noise.std_noise_observations = 0.1;

	// =========== Topology parameters ===========
	rba.parameters.srba.max_tree_depth = 3;
	rba.parameters.srba.max_optimize_depth = 3;

	min_graph_opt_time = 1e8;
	max_graph_opt_time = 0;
	total_graph_opt_time = 0;

	min_edge_weight = 1e10;
	max_edge_weight = 0;

	min_closure_detect_time = 1e8;
	max_closure_detect_time = 0;
	total_closure_detect_time = 0;
	
	min_closure_candidate = 1000;
	max_closure_candidate = 0;

	keyframeCount = 0;
	clousureCount = 0;
}

bool SrbaManager::addNode(Frame* frame, float weight, bool is_keyframe_candidate/* = false*/, string *inliers_sig, string *exists_sig)
{
	graph.push_back(frame);
	if (!is_keyframe_candidate)
		return false;

	keyframe_indices.push_back(graph.size() - 1);

	clock_t start = 0;
	double time = 0;
	int count = 0;
	bool isNewKeyframe = false;

	if (graph.size() == 1)
	{
		SrbaT::new_kf_observations_t list_obs;
		SrbaT::new_kf_observation_t obs_field;
		obs_field.is_fixed = false;
		obs_field.is_unknown_with_init_val = false;

		for (int i = 0; i < frame->f.size(); i++)
		{
			frame->f.feature_ids[i] = Feature::NewID();
			obs_field.obs.feat_id = frame->f.feature_ids[i];
			obs_field.obs.obs_data.pt.x = frame->f.feature_pts_3d[i](0);
			obs_field.obs.obs_data.pt.y = frame->f.feature_pts_3d[i](1);
			obs_field.obs.obs_data.pt.z = frame->f.feature_pts_3d[i](2);
			list_obs.push_back(obs_field);
		}

		SrbaT::TNewKeyFrameInfo new_kf_info;
		rba.define_new_keyframe(
			list_obs,      // Input observations for the new KF
			new_kf_info,   // Output info
			true           // Also run local optimization?
			);

// 		cout << "Created KF #" << new_kf_info.kf_id
// 			<< " | # kf-to-kf edges created:" << new_kf_info.created_edge_ids.size() << endl
// 			<< "Optimization error: " << new_kf_info.optimize_results.total_sqr_error_init << " -> " << new_kf_info.optimize_results.total_sqr_error_final << endl
// 			<< "-------------------------------------------------------" << endl;

		last_kc = 0;
		last_kc_tran = frame->tran;
		keyframeCount++;
		Eigen::Vector3f translation = TranslationFromMatrix4f(this->graph[0]->tran);
		active_window.insert(translation(0), translation(1), 0);
		isNewKeyframe = true;
	}
	else
	{
		int k = graph.size() - 1;

		vector<cv::DMatch> matches;

		start = clock();
		// get translation of current tran
		// move active window
		Eigen::Vector3f translation = TranslationFromMatrix4f(graph[k]->tran);
		float active_window_size = Config::instance()->get<float>("active_window_size");
		active_window.move(graph, RectangularRegion(translation.x(), translation.y(), active_window_size, active_window_size));
		active_window.feature_pool->findMatchedPairs(matches, graph[k]->f,
			Config::instance()->get<int>("kdtree_max_leaf"));

		int N = Config::instance()->get<int>("keyframe_check_N");
		int M = Config::instance()->get<int>("keyframe_check_M");
		int F = Config::instance()->get<int>("keyframe_check_F");
		int width = Config::instance()->get<int>("image_width");
		int height = Config::instance()->get<int>("image_height");
		int *keyframeTest = new int[N * M];
		bool *keyframeExists = new bool[N * M];
		int keyframeTestCount = 0;
		int keyframeExistsCount = 0;

		for (int i = 0; i < M; i++)
		{
			for (int j = 0; j < N; j++)
			{
				keyframeTest[i * N + j] = 0;
				keyframeExists[i * N + j] = false;
			}
		}

		for (int i = 0; i < this->graph[k]->f.size(); i++)
		{
			cv::KeyPoint keypoint = this->graph[k]->f.feature_pts[i];
			int tN = N * keypoint.pt.x / width;
			int tM = M * keypoint.pt.y / height;
			tN = tN < 0 ? 0 : (tN >= N ? N - 1 : tN);
			tM = tM < 0 ? 0 : (tM >= M ? M - 1 : tM);
			if (!keyframeExists[tM * N + tN])
			{
				keyframeExistsCount++;
				keyframeExists[tM * N + tN] = true;
			}
		}

		for (int i = 0; i < matches.size(); i++)
		{
			cv::KeyPoint keypoint = this->graph[k]->f.feature_pts[matches[i].queryIdx];
			int tN = N * keypoint.pt.x / width;
			int tM = M * keypoint.pt.y / height;
			tN = tN < 0 ? 0 : (tN >= N ? N - 1 : tN);
			tM = tM < 0 ? 0 : (tM >= M ? M - 1 : tM);

			keyframeTest[tM * N + tN]++;
		}

		for (int i = 0; i < M; i++)
		{
			for (int j = 0; j < N; j++)
			{
				if (keyframeTest[i * N + j] >= F)
					keyframeTestCount++;
			}
		}

		// for test
		if (inliers_sig != nullptr)
		{
			*inliers_sig = "";
			for (int i = 0; i < M; i++)
			{
				for (int j = 0; j < N; j++)
				{
					*inliers_sig += keyframeTest[i * N + j] >= F ? "1" : "0";
				}
			}
		}
		if (exists_sig != nullptr)
		{
			*exists_sig = "";
			for (int i = 0; i < M; i++)
			{
				for (int j = 0; j < N; j++)
				{
					*exists_sig += keyframeExists[i * N + j] ? "1" : "0";
				}
			}
		}
		delete keyframeTest;
		delete keyframeExists;

		count = 1;
		if (count < min_closure_candidate) min_closure_candidate = count;
		if (count > max_closure_candidate) max_closure_candidate = count;
		std::cout << ", Closure Candidate: " << count;

		for (int i = 0; i < matches.size(); i++)
		{
			graph[k]->f.feature_ids[matches[i].queryIdx] = active_window.feature_pool->feature_ids[matches[i].trainIdx];
		}

		SrbaT::new_kf_observations_t list_obs;
		SrbaT::new_kf_observation_t obs_field;
		obs_field.is_fixed = false;
		obs_field.is_unknown_with_init_val = false;

		for (int i = 0; i < graph[k]->f.size(); i++)
		{
			if (graph[k]->f.feature_ids[i] == 0)
				graph[k]->f.feature_ids[i] = Feature::NewID();
			obs_field.obs.feat_id = frame->f.feature_ids[i];
			obs_field.obs.obs_data.pt.x = frame->f.feature_pts_3d[i](0);
			obs_field.obs.obs_data.pt.y = frame->f.feature_pts_3d[i](1);
			obs_field.obs.obs_data.pt.z = frame->f.feature_pts_3d[i](2);
			list_obs.push_back(obs_field);
		}

		SrbaT::TNewKeyFrameInfo new_kf_info;
		rba.define_new_keyframe(
			list_obs,      // Input observations for the new KF
			new_kf_info,   // Output info
			true           // Also run local optimization?
			);

// 		cout << "Created KF #" << new_kf_info.kf_id
// 			<< " | # kf-to-kf edges created:" << new_kf_info.created_edge_ids.size() << endl
// 			<< "Optimization error: " << new_kf_info.optimize_results.total_sqr_error_init << " -> " << new_kf_info.optimize_results.total_sqr_error_final << endl
// 			<< "-------------------------------------------------------" << endl;

		if (keyframeTestCount + N * M - keyframeExistsCount < N * M * Config::instance()->get<double>("keyframe_check_P"))
		{
			frame_in_quadtree_indices.insert(k);
			Eigen::Vector3f translation = TranslationFromMatrix4f(this->graph[k]->tran);
			active_window.insert(translation(0), translation(1), k);
			keyframeCount++;
			std::cout << ", Keyframe";
			isNewKeyframe = true;
		}

		time = (clock() - start) / 1000.0;
		if (time < min_graph_opt_time) min_graph_opt_time = time;
		if (time > max_graph_opt_time) max_graph_opt_time = time;
		total_graph_opt_time += time;
		std::cout << ", Graph: " << fixed << std::setprecision(3) << time;

		deque<SrbaT::keyframe_info> kfs = rba.get_rba_state().keyframes;
		int ki = 0;
		for (int i = 1; i < this->graph.size(); i++)
		{
			Eigen::Matrix4f tran = Eigen::Matrix4f::Identity();
			if (ki < keyframe_indices.size() && keyframe_indices[ki] == i)
			{
				std::deque<SrbaT::k2k_edge_t *> edges = kfs[ki].adjacent_k2k_edges;
				for (int j = 0; j < edges.size(); j++)
				{
					if (edges[j]->from < ki &&
						edges[j]->to == ki)
					{
						SrbaT::pose_t pose = -edges[j]->inv_pose;
						Eigen::Vector3f translation(pose.x(), pose.y(), pose.z());
						Eigen::Quaternionf quaternion = QuaternionFromEulerAngle(pose.yaw(), pose.pitch(), pose.roll());
						Eigen::Matrix4f rt = transformationFromQuaternionsAndTranslation(quaternion, translation);
						tran = rt * graph[edges[j]->from]->tran;
						break;
					}
				}
			}
			else
			{
				tran = graph[i]->relative_tran * graph[i - 1]->tran;
			}
			if (this->graph[i]->tran != tran)
			{
				Eigen::Vector3f old_translation = TranslationFromMatrix4f(this->graph[i]->tran);
				Eigen::Vector3f new_translation = TranslationFromMatrix4f(tran);
				this->graph[i]->tran = tran;

				if (frame_in_quadtree_indices.find(i) != frame_in_quadtree_indices.end())
				{
					// keyframe pose changed
					// update 3d feature points
					this->graph[i]->f.updateFeaturePoints3D(tran);

					// update quadtree
					if (old_translation(0) != new_translation(0) || old_translation(1) != new_translation(1))
					{
						this->active_window.key_frames->update(old_translation(0), old_translation(1), i, new_translation(0), new_translation(1));
					}
				}
			}
		}

		last_kc = k;
		last_kc_tran = this->graph[k]->tran;
	}
	return isNewKeyframe;
}

Eigen::Matrix4f SrbaManager::getTransformation(int k)
{
	if (k < 0 || k >= graph.size())
	{
		return Eigen::Matrix4f::Identity();
	}
	return graph[k]->tran;
}

Eigen::Matrix4f SrbaManager::getLastTransformation()
{
	if (graph.size() == 0)
	{
		return Eigen::Matrix4f::Identity();
	}
	return graph[graph.size() - 1]->tran;
}

Eigen::Matrix4f SrbaManager::getLastKeyframeTransformation()
{
	return last_kc_tran;
// 	if (last_keyframe < 0 || last_keyframe >= graph.size())
// 	{
// 		return Eigen::Matrix4f::Identity();
// 	}
// 	return graph[last_keyframe]->tran;
}

int SrbaManager::size()
{
	return graph.size();
}