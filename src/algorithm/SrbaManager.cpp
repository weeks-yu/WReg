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


	// --------------------------------------------------------------------------------
	// Set parameters
	// --------------------------------------------------------------------------------
	rba_graph.setVerbosityLevel(1);   // 0: None; 1:Important only; 2:Verbose

	rba_graph.parameters.srba.use_robust_kernel = false;
	//rba.parameters.srba.optimize_new_edges_alone  = false;  // skip optimizing new edges one by one? Relative graph-slam without landmarks should be robust enough, but just to make sure we can leave this to "true" (default)

	// Information matrix for relative pose observations:
	{
		const double STD_NOISE_XYZ = 0.01;
		const double STD_NOISE_ANGLES = mrpt::utils::DEG2RAD(0.5);
		Eigen::Matrix<double, 6, 6> ObsL;
		ObsL.setZero();
		// X,Y,Z:
		for (int i = 0; i < 3; i++) ObsL(i, i) = 1 / mrpt::utils::square(STD_NOISE_XYZ);
		// Yaw,pitch,roll:
		for (int i = 0; i < 3; i++) ObsL(3 + i, 3 + i) = 1 / mrpt::utils::square(STD_NOISE_ANGLES);

		// Set:
		rba_graph.parameters.obs_noise.lambda = ObsL;
	}

	// =========== Topology parameters ===========
	rba_graph.parameters.srba.max_tree_depth = 3;
	rba_graph.parameters.srba.max_optimize_depth = 3;
	rba_graph.parameters.ecp.submap_size = 5;
	rba_graph.parameters.ecp.min_obs_to_loop_closure = 1;
	// ===========================================

	using_graph_slam = true;

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

	keyframeInQuadTreeCount = 0;
	clousureCount = 0;
}

bool SrbaManager::addNode(Frame* frame, float weight, bool is_keyframe_candidate/* = false*/, string *inliers_sig, string *exists_sig)
{
	graph.push_back(frame);
	if (!is_keyframe_candidate)
		return false;

	keyframe_indices.push_back(graph.size() - 1);
	keyframe_poses.push_back(Eigen::Matrix4f::Identity());
	is_keyfram_pose_set.push_back(false);

	clock_t start = 0;
	double time = 0;
	int count = 0;
	bool isNewKeyframe = false;

	if (graph.size() == 1)
	{
		graph[0]->tran = graph[0]->relative_tran;
		if (using_graph_slam)
		{
			SrbaGraphT::new_kf_observations_t list_obs;
			SrbaGraphT::new_kf_observation_t obs_field;

			// fake landmark
			obs_field.is_fixed = true;
			obs_field.obs.feat_id = 0; // Feature ID == keyframe ID
			obs_field.obs.obs_data.x = 0;   // Landmark values are actually ignored.
			obs_field.obs.obs_data.y = 0;
			obs_field.obs.obs_data.z = 0;
			obs_field.obs.obs_data.yaw = 0;
			obs_field.obs.obs_data.pitch = 0;
			obs_field.obs.obs_data.roll = 0;
			list_obs.push_back(obs_field);

			SrbaGraphT::TNewKeyFrameInfo new_kf_info;
			rba_graph.define_new_keyframe(
				list_obs,      // Input observations for the new KF
				new_kf_info,   // Output info
				true           // Also run local optimization?
				);
			keyframe_id.insert(pair<int, int>(0, 0));

// 			cout << "Created KF #" << new_kf_info.kf_id
// 				<< " | # kf-to-kf edges created:" << new_kf_info.created_edge_ids.size() << endl
// 				<< "Optimization error: " << new_kf_info.optimize_results.total_sqr_error_init << " -> " << new_kf_info.optimize_results.total_sqr_error_final << endl
// 				<< "-------------------------------------------------------" << endl;
		}
		else
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
			keyframe_id.insert(pair<int, int>(0, 0));

			// 		cout << "Created KF #" << new_kf_info.kf_id
			// 			<< " | # kf-to-kf edges created:" << new_kf_info.created_edge_ids.size() << endl
			// 			<< "Optimization error: " << new_kf_info.optimize_results.total_sqr_error_init << " -> " << new_kf_info.optimize_results.total_sqr_error_final << endl
			// 			<< "-------------------------------------------------------" << endl;
		}
		
		last_kc = 0;
		last_kc_tran = graph[0]->tran;
		keyframeInQuadTreeCount++;
		Eigen::Vector3f translation = TranslationFromMatrix4f(graph[0]->tran);
		active_window.insert(translation(0), translation(1), 0);
		isNewKeyframe = true;
	}
	else
	{
		int k = graph.size() - 1;

		if (using_graph_slam)
		{
			start = clock();
			// get translation of current tran
			// move active window
			Eigen::Vector3f translation = TranslationFromMatrix4f(this->graph[k]->tran);
			float active_window_size = Config::instance()->get<float>("active_window_size");
			active_window.move(this->graph, RectangularRegion(translation.x(), translation.y(), active_window_size, active_window_size));

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

			vector<int> frames;
			vector<vector<cv::DMatch>> matches;
			this->active_window.feature_pool->findMatchedPairsMultiple(frames, matches, this->graph[k]->f,
				Config::instance()->get<int>("kdtree_k_mult"), Config::instance()->get<int>("kdtree_max_leaf_mult"));

			count = matches.size();
			if (count < min_closure_candidate) min_closure_candidate = count;
			if (count > max_closure_candidate) max_closure_candidate = count;
			std::cout << ", Closure Candidate: " << count;

			count = 0;
			SrbaGraphT::new_kf_observations_t list_obs;
			for (int i = 0; i < frames.size(); i++)
			{
				Eigen::Matrix4f tran;
				float rmse;
				vector<cv::DMatch> inliers;
				// find edges
				if (Feature::getTransformationByRANSAC(tran, rmse, &inliers,
					this->active_window.feature_pool, &(this->graph[k]->f), matches[i]))
				{
					SrbaGraphT::new_kf_observation_t obs_field;
					obs_field.is_fixed = false;   // "Landmarks" (relative poses) have unknown relative positions (i.e. treat them as unknowns to be estimated)
					obs_field.is_unknown_with_init_val = false; // Ignored, since all observed "fake landmarks" already have an initialized value.

					obs_field.obs.feat_id = keyframe_id[this->active_window.active_frames[frames[i]]];

					Eigen::Vector3f translation = TranslationFromMatrix4f(tran);
					Eigen::Vector3f eulerAngle = EulerAngleFromMatrix4f(tran);
					obs_field.obs.obs_data.x = translation(0);
					obs_field.obs.obs_data.y = translation(1);
					obs_field.obs.obs_data.z = translation(2);
					obs_field.obs.obs_data.yaw = eulerAngle(0);
					obs_field.obs.obs_data.pitch = eulerAngle(1);
					obs_field.obs.obs_data.roll = eulerAngle(2);

					list_obs.push_back(obs_field);
					count++;

					for (int j = 0; j < inliers.size(); j++)
					{
						cv::KeyPoint keypoint = this->graph[k]->f.feature_pts[inliers[j].queryIdx];
						int tN = N * keypoint.pt.x / width;
						int tM = M * keypoint.pt.y / height;
						tN = tN < 0 ? 0 : (tN >= N ? N - 1 : tN);
						tM = tM < 0 ? 0 : (tM >= M ? M - 1 : tM);

						keyframeTest[tM * N + tN]++;
					}
				}
			}

			start = clock();
			SrbaGraphT::TNewKeyFrameInfo new_kf_info;
			rba_graph.define_new_keyframe(
				list_obs,      // Input observations for the new KF
				new_kf_info,   // Output info
				true           // Also run local optimization?
				);
			keyframe_id.insert(pair<int, int>(k, keyframe_indices.size() - 1));

			time = (clock() - start) / 1000.0;
			if (time < min_graph_opt_time) min_graph_opt_time = time;
			if (time > max_graph_opt_time) max_graph_opt_time = time;
			total_graph_opt_time += time;
			std::cout << ", Graph: " << fixed << setprecision(3) << time;

// 			cout << "Created KF #" << new_kf_info.kf_id
// 				<< " | # kf-to-kf edges created:" << new_kf_info.created_edge_ids.size() << endl
// 				<< "Optimization error: " << new_kf_info.optimize_results.total_sqr_error_init << " -> " << new_kf_info.optimize_results.total_sqr_error_final << endl
// 				<< "-------------------------------------------------------" << endl;

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

			if (keyframeTestCount + N * M - keyframeExistsCount < N * M * Config::instance()->get<double>("keyframe_check_P"))
			{
				frame_in_quadtree_indices.insert(k);
				Eigen::Vector3f translation = TranslationFromMatrix4f(this->graph[k]->tran);
				active_window.insert(translation(0), translation(1), k);
				keyframeInQuadTreeCount++;
				std::cout << ", Keyframe";
				isNewKeyframe = true;
			}

// 			deque<SrbaGraphT::keyframe_info> kfs = rba_graph.get_rba_state().keyframes;
// 			int ki = 0;
// 			for (int i = 1; i < this->graph.size(); i++)
// 			{
// 				while (ki < keyframe_indices.size() && keyframe_indices[ki] < i)
// 					ki++;
// 
// 				Eigen::Matrix4f tran = Eigen::Matrix4f::Identity();
// 				if (ki < keyframe_indices.size() && keyframe_indices[ki] == i)
// 				{
// 					std::deque<SrbaGraphT::k2k_edge_t *> edges = kfs[ki].adjacent_k2k_edges;
// 					for (int j = 0; j < edges.size(); j++)
// 					{
// 						if (edges[j]->from < ki &&
// 							edges[j]->to == ki)
// 						{
// 							SrbaGraphT::pose_t pose = -edges[j]->inv_pose;
// 							Eigen::Vector3f translation(pose.x(), pose.y(), pose.z());
// 							Eigen::Quaternionf quaternion = QuaternionFromEulerAngle(pose.yaw(), pose.pitch(), pose.roll());
// 							Eigen::Matrix4f rt = transformationFromQuaternionsAndTranslation(quaternion, translation);
// 							tran = rt * graph[edges[j]->from]->tran;
// 							break;
// 						}
// 					}
// 				}
// 				else
// 				{
// 					tran = graph[i]->relative_tran * graph[i - 1]->tran;
// 				}
// 				if (this->graph[i]->tran != tran)
// 				{
// 					Eigen::Vector3f old_translation = TranslationFromMatrix4f(this->graph[i]->tran);
// 					Eigen::Vector3f new_translation = TranslationFromMatrix4f(tran);
// 					this->graph[i]->tran = tran;
// 
// 					if (frame_in_quadtree_indices.find(i) != frame_in_quadtree_indices.end())
// 					{
// 						// keyframe pose changed
// 						// update 3d feature points
// 						this->graph[i]->f.updateFeaturePoints3D(tran);
// 
// 						// update quadtree
// 						if (old_translation(0) != new_translation(0) || old_translation(1) != new_translation(1))
// 						{
// 							this->active_window.key_frames->update(old_translation(0), old_translation(1), i, new_translation(0), new_translation(1));
// 						}
// 					}
// 				}
// 			}
			for (int i = 0; i < is_keyfram_pose_set.size(); i++)
			{
				is_keyfram_pose_set[i] = false;
			}
			rba_graph.bfs_visitor(0, 1000000, false, *this, *this, *this, *this);

			int ki = 0;
			for (int i = 0; i < this->graph.size(); i++)
			{
				while (ki < keyframe_indices.size() && keyframe_indices[ki] < i)
					ki++;

				Eigen::Matrix4f tran = Eigen::Matrix4f::Identity();
				if (ki < keyframe_indices.size() && keyframe_indices[ki] == i)
				{
					tran = keyframe_poses[ki];
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
		}
		else
		{
			vector<cv::DMatch> temp_matches, matches;
			int maxleaf = Config::instance()->get<int>("kdtree_max_leaf");

			start = clock();
			// get translation of current tran
			// move active window
			Eigen::Vector3f translation = TranslationFromMatrix4f(graph[k]->tran);
			float active_window_size = Config::instance()->get<float>("active_window_size");
			active_window.move(graph, RectangularRegion(translation.x(), translation.y(), active_window_size, active_window_size));
			active_window.feature_pool->findMatchedPairs(temp_matches, graph[k]->f, maxleaf, 1);

			for (int i = 0; i < temp_matches.size(); i++)
			{
				vector<cv::DMatch> tm;
				int queryid = temp_matches[i].queryIdx;
				int trainid = temp_matches[i].trainIdx;
				int frame_index = active_window.feature_pool->feature_frame_index[trainid];
				int c = graph[active_window.active_frames[frame_index]]->f.findMatched(tm,
					graph[k]->f.feature_descriptors.row(queryid), maxleaf);

				for (int j = 0; j < tm.size(); j++)
				{
					tm[j].queryIdx = queryid;
					matches.push_back(tm[j]);
				}
			}

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
			keyframe_id.insert(pair<int, int>(k, keyframe_indices.size() - 1));

			// 		cout << "Created KF #" << new_kf_info.kf_id
			// 			<< " | # kf-to-kf edges created:" << new_kf_info.created_edge_ids.size() << endl
			// 			<< "Optimization error: " << new_kf_info.optimize_results.total_sqr_error_init << " -> " << new_kf_info.optimize_results.total_sqr_error_final << endl
			// 			<< "-------------------------------------------------------" << endl;

			if (keyframeTestCount + N * M - keyframeExistsCount < N * M * Config::instance()->get<double>("keyframe_check_P"))
			{
				frame_in_quadtree_indices.insert(k);
				Eigen::Vector3f translation = TranslationFromMatrix4f(this->graph[k]->tran);
				active_window.insert(translation(0), translation(1), k);
				keyframeInQuadTreeCount++;
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
				while (ki < keyframe_indices.size() && keyframe_indices[ki] < i)
					ki++;

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

bool SrbaManager::visit_filter_feat(const TLandmarkID lm_ID, const topo_dist_t cur_dist)
{
	return false;
}
void SrbaManager::visit_feat(const TLandmarkID lm_ID, const topo_dist_t cur_dist)
{

}

bool SrbaManager::visit_filter_kf(const TKeyFrameID kf_ID, const topo_dist_t cur_dist)
{
	return true;
}
void SrbaManager::visit_kf(const TKeyFrameID kf_ID, const topo_dist_t cur_dist)
{

}

bool SrbaManager::visit_filter_k2k(
	const TKeyFrameID current_kf,
	const TKeyFrameID next_kf,
	const SrbaGraphT::k2k_edge_t* edge,
	const topo_dist_t cur_dist)
{
	return true;
}
void SrbaManager::visit_k2k(
	const TKeyFrameID current_kf,
	const TKeyFrameID next_kf,
	const SrbaGraphT::k2k_edge_t* edge,
	const topo_dist_t cur_dist)
{
	if (!is_keyfram_pose_set[next_kf])
	{
		SrbaGraphT::pose_t pose = edge->inv_pose;
		Eigen::Vector3f translation(pose.x(), pose.y(), pose.z());
		Eigen::Quaternionf quaternion = QuaternionFromEulerAngle(pose.yaw(), pose.pitch(), pose.roll());
		Eigen::Matrix4f rt = transformationFromQuaternionsAndTranslation(quaternion, translation);
		keyframe_poses[next_kf] = rt.inverse() * keyframe_poses[current_kf];
		is_keyfram_pose_set[next_kf] = true;
	}
}

bool SrbaManager::visit_filter_k2f(
	const TKeyFrameID current_kf,
	const SrbaGraphT::k2f_edge_t* edge,
	const topo_dist_t cur_dist)
{
	return false;
}
void SrbaManager::visit_k2f(
	const TKeyFrameID current_kf,
	const SrbaGraphT::k2f_edge_t *edge,
	const topo_dist_t cur_dist)
{

}