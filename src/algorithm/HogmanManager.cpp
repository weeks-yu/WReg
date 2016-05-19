#include "HogmanManager.h"
#include "Transformation.h"

HogmanManager::HogmanManager(bool keyframe_only)
{
	int numLevels = Config::instance()->get<int>("graph_levels");
	int nodeDistance = Config::instance()->get<int>("node_distance");
	this->optimizer = new AISNavigation::HCholOptimizer3D(numLevels, nodeDistance);
	iteration_count = Config::instance()->get<int>("hogman_iterations");

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

	threads = Config::instance()->get<int>("icpcuda_threads");
	blocks = Config::instance()->get<int>("icpcuda_blocks");
	int width = Config::instance()->get<int>("image_width");
	int height = Config::instance()->get<int>("image_height");
	float cx = Config::instance()->get<float>("camera_cx");
	float cy = Config::instance()->get<float>("camera_cy");
	float fx = Config::instance()->get<float>("camera_fx");
	float fy = Config::instance()->get<float>("camera_fy");
	float depthFactor = Config::instance()->get<float>("depth_factor");
	float distThresh = Config::instance()->get<float>("dist_threshold");
	float angleThresh = Config::instance()->get<float>("angle_threshold");
	pcc = new PointCloudCuda(width, height, cx, cy, fx, fy, depthFactor, distThresh, angleThresh);
}

bool HogmanManager::addNode(Frame* frame, float weight, bool is_keyframe_candidate/* = false*/, string *inliers_sig, string *exists_sig)
{
	graph.push_back(frame);
	temp_poses.push_back(frame->tran);

	if (!is_keyframe_candidate)
		return false;

	keyframe_indices.push_back(graph.size() - 1);

	clock_t start = 0;
	double time = 0;
	int count = 0;
	bool isNewKeyframe = false;


	if (this->graph.size() == 1)
	{
		this->optimizer->addVertex(0, Transformation3(), 1e9 * Matrix6::eye(1.0));
		keyframe_id.insert(pair<int, int>(0, 0));

		last_kc = 0;
		last_kc_tran = graph[0]->tran;
		keyframeInQuadTreeCount++;
		frame_in_quadtree_indices.insert(0);
		Eigen::Vector3f translation = TranslationFromMatrix4f(graph[0]->tran);
		active_window.insert(translation(0), translation(1), 0);
		isNewKeyframe = true;
	}
	else
	{
		int k = graph.size() - 1;

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

		for (int i = 0; i < this->graph[k]->f->size(); i++)
		{
			cv::KeyPoint keypoint = this->graph[k]->f->feature_pts[i];
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
		std::cout << ", aw: " << active_window.active_frames.size();
		vector<int> frames;
		vector<vector<cv::DMatch>> matches;
		this->active_window.feature_pool->findMatchedPairsMultiple(frames, matches, this->graph[k]->f,
			Config::instance()->get<int>("kdtree_k_mult"), Config::instance()->get<int>("kdtree_max_leaf_mult"));

		count = matches.size();
		if (count < min_closure_candidate) min_closure_candidate = count;
		if (count > max_closure_candidate) max_closure_candidate = count;
		std::cout << ", Closure Candidate: " << count;

		count = 0;
		AISNavigation::PoseGraph3D::Vertex *now_v = this->optimizer->addVertex(keyframe_indices.size() - 1,
			eigenToHogman(graph[k]->tran), Matrix6::eye(1.0));

		pcc->initCurr((unsigned short *)this->graph[k]->depth->data, 20.0f);
		Eigen::Matrix<double, 6, 6> information;
		for (int i = 0; i < frames.size(); i++)
		{
			if (keyframe_id[this->active_window.active_frames[frames[i]]] == keyframe_indices.size() - 2)
			{
				//continue; // do not run ransac between this and last keyframe
			}
			Eigen::Matrix4f tran;
			float rmse;
			vector<cv::DMatch> inliers;
			pcc->initPrev((unsigned short *)this->graph[this->active_window.active_frames[frames[i]]]->depth->data, 20.0f);
			// find edges
			if (Feature::getTransformationByRANSAC(tran, information, rmse, &inliers,
				this->active_window.feature_pool, this->graph[k]->f,
				pcc, matches[i]))
			{
				AISNavigation::PoseGraph3D::Vertex* other_v = this->optimizer->vertex(
					keyframe_id[this->active_window.active_frames[frames[i]]]);

				double w = 1.0 / rmse;
				if (w < min_edge_weight) min_edge_weight = w;
				if (w > max_edge_weight) max_edge_weight = w;
				this->optimizer->addEdge(other_v, now_v, eigenToHogman(tran), Matrix6::eye(1.0)/*eigenToHogman(information)*/);

#ifdef SAVE_TEST_INFOS
				baseid.push_back(k);
				targetid.push_back(this->active_window.active_frames[frames[i]]);
				rmses.push_back(rmse);
				matchescount.push_back(matches[i].size());
				inlierscount.push_back(inliers.size());
				ransactrans.push_back(tran);
#endif

				count++;
				for (int j = 0; j < inliers.size(); j++)
				{
					cv::KeyPoint keypoint = this->graph[k]->f->feature_pts[inliers[j].queryIdx];
					int tN = N * keypoint.pt.x / width;
					int tM = M * keypoint.pt.y / height;
					tN = tN < 0 ? 0 : (tN >= N ? N - 1 : tN);
					tM = tM < 0 ? 0 : (tM >= M ? M - 1 : tM);

					keyframeTest[tM * N + tN]++;
				}
			}
		}

		if (keyframe_indices.size() > 1)
		{
			pcc->initPrev((unsigned short *)this->graph[keyframe_indices[keyframe_indices.size() - 2]]->depth->data, 20.0f);
			Eigen::Matrix4f tran = Eigen::Matrix4f::Identity();
			for (int i = keyframe_indices[keyframe_indices.size() - 2] + 1; i < k; i++)
			{
				tran = tran * graph[i]->relative_tran;
			}
			Eigen::Vector3f t = tran.topRightCorner(3, 1);
			Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rot = tran.topLeftCorner(3, 3);
			int point_count, point_corr_count;
			pcc->getCoresp(t, rot, information, point_count, point_corr_count, threads, blocks);

			AISNavigation::PoseGraph3D::Vertex *prev_v = this->optimizer->vertex(keyframe_indices.size() - 2);
			this->optimizer->addEdge(prev_v, now_v, eigenToHogman(tran), Matrix6::eye(1.0)/*eigenToHogman(information)*/);

#ifdef SAVE_TEST_INFOS
			baseid.push_back(k);
			targetid.push_back(keyframe_indices[keyframe_indices.size() - 2]);
			rmses.push_back(0.0);
			matchescount.push_back(0);
			inlierscount.push_back(0);
			ransactrans.push_back(tran);
#endif
			if (frame_in_quadtree_indices.find(keyframe_indices[keyframe_indices.size() - 2]) == frame_in_quadtree_indices.end())
				delete this->graph[keyframe_indices[keyframe_indices.size() - 2]]->depth;
			count++;
		}

		cout << ", oberservation: " << count << endl;

		start = clock();
		this->optimizer->optimize(iteration_count, true);

		keyframe_id.insert(pair<int, int>(k, keyframe_indices.size() - 1));
		time = (clock() - start) / 1000.0;
		if (time < min_graph_opt_time) min_graph_opt_time = time;
		if (time > max_graph_opt_time) max_graph_opt_time = time;
		total_graph_opt_time += time;
		std::cout << ", Graph: " << fixed << setprecision(3) << time;

		cout << endl;
		for (int i = 0; i < M; i++)
		{
			for (int j = 0; j < N; j++)
			{
				cout << keyframeTest[i * N + j] << " ";
				if (keyframeTest[i * N + j] >= F)
					keyframeTestCount++;
			}
		}
		cout << endl;

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

		if (keyframeTestCount + N * M - keyframeExistsCount < N * M * Config::instance()->get<float>("keyframe_check_P"))
		{
			frame_in_quadtree_indices.insert(k);
			Eigen::Vector3f translation = TranslationFromMatrix4f(this->graph[k]->tran);
			active_window.insert(translation(0), translation(1), k);
			keyframeInQuadTreeCount++;
			std::cout << ", Keyframe";
			isNewKeyframe = true;
		}

		int ki = 0;
		for (int i = 0; i < this->graph.size(); i++)
		{
			if (this->graph[i]->tran != temp_poses[i])
			{
				Eigen::Vector3f old_translation = TranslationFromMatrix4f(this->graph[i]->tran);
				Eigen::Vector3f new_translation = TranslationFromMatrix4f(temp_poses[i]);
				this->graph[i]->tran = temp_poses[i];

				if (frame_in_quadtree_indices.find(i) != frame_in_quadtree_indices.end())
				{
					// keyframe pose changed
					// update 3d feature points
					this->graph[i]->f->updateFeaturePoints3D(temp_poses[i]);

					// update quadtree
					if (old_translation(0) != new_translation(0) || old_translation(1) != new_translation(1))
					{
						this->active_window.key_frames->update(old_translation(0), old_translation(1), i, new_translation(0), new_translation(1));
					}
				}
			}
		}

		for (int i = 0; i < this->graph.size(); i++)
		{
			if (keyframe_id.find(i) != keyframe_id.end())
			{
				AISNavigation::PoseGraph3D::Vertex* v = this->optimizer->vertex(keyframe_id[i]);
				temp_poses[i] = hogmanToEigen(v->transformation);
			}
			else
			{
				temp_poses[i] = temp_poses[i - 1] * graph[i]->relative_tran;
			}
		}

		for (int i = 0; i < this->graph.size(); i++)
		{
			if (this->graph[i]->tran != temp_poses[i])
			{
				Eigen::Vector3f old_translation = TranslationFromMatrix4f(this->graph[i]->tran);
				Eigen::Vector3f new_translation = TranslationFromMatrix4f(temp_poses[i]);
				this->graph[i]->tran = temp_poses[i];

				if (frame_in_quadtree_indices.find(i) != frame_in_quadtree_indices.end())
				{
					// keyframe pose changed
					// update 3d feature points
					this->graph[i]->f->updateFeaturePoints3D(temp_poses[i]);

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

Eigen::Matrix4f HogmanManager::getTransformation(int k)
{
	if (k < 0 || k >= graph.size())
	{
		return Eigen::Matrix4f::Identity();
	}
	return graph[k]->tran;
}

Eigen::Matrix4f HogmanManager::getLastTransformation()
{
	if (graph.size() == 0)
	{
		return Eigen::Matrix4f::Identity();
	}
	return graph[graph.size() - 1]->tran;
}

Eigen::Matrix4f HogmanManager::getLastKeyframeTransformation()
{
	return last_kc_tran;
// 	if (last_keyframe < 0 || last_keyframe >= graph.size())
// 	{
// 		return Eigen::Matrix4f::Identity();
// 	}
// 	return graph[last_keyframe]->tran;
}

int HogmanManager::size()
{
	return graph.size();
}