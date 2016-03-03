#include "HogmanManager.h"
#include "Transformation.h"

HogmanManager::HogmanManager(bool keyframe_only)
{
	int numLevels = Config::instance()->get<int>("graph_levels");
	int nodeDistance = Config::instance()->get<int>("node_distance");
	this->optimizer = new AISNavigation::HCholOptimizer3D(numLevels, nodeDistance);

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

	keyframeOnly = keyframe_only;

	int width = Config::instance()->get<int>("image_width");
	int height = Config::instance()->get<int>("image_height");
	double cx = Config::instance()->get<double>("camera_cx");
	double cy = Config::instance()->get<double>("camera_cy");
	double fx = Config::instance()->get<double>("camera_fx");
	double fy = Config::instance()->get<double>("camera_fy");
	double depthFactor = Config::instance()->get<double>("depth_factor");
	icpcuda = new ICPOdometry(width, height, cx, cy, fx, fy, depthFactor);
}

bool HogmanManager::addNode(Frame* frame, float weight, bool is_keyframe_candidate/* = false*/, string *inliers_sig, string *exists_sig)
{
	clock_t start = 0;
	double time = 0;
	int count = 0;
	bool isNewKeyframe = false;

	this->graph.push_back(frame);

	if (keyframeOnly && !is_keyframe_candidate)
	{
		return false;
	}

	if (this->graph.size() == 1)
	{
		this->optimizer->addVertex(0, Transformation3(), 1e9 * Matrix6::eye(1.0));
		if (is_keyframe_candidate)
		{
			last_kc = 0;
			last_kc_tran = frame->tran;
			keyframeCount++;
			Eigen::Vector3f translation = TranslationFromMatrix4f(this->graph[0]->tran);
			active_window.insert(translation(0), translation(1), 0);
			isNewKeyframe = true;
		}
	}
	else
	{
		if (keyframeOnly)
		{

		}
		else
		{
			int k = this->graph.size() - 1;
			AISNavigation::PoseGraph3D::Vertex *now_v = this->optimizer->addVertex(k, Transformation3(), Matrix6::eye(1.0));
			AISNavigation::PoseGraph3D::Vertex *prev_v = this->optimizer->vertex(k - 1);
			this->optimizer->addEdge(prev_v, now_v, eigenToHogman(frame->relative_tran), Matrix6::eye(weight));

			vector<int> frames;
			vector<vector<cv::DMatch>> matches;

			if (is_keyframe_candidate)
			{
				start = clock();
				// get translation of current tran
				// move active window
				Eigen::Vector3f translation = TranslationFromMatrix4f(this->graph[k]->tran);
				float active_window_size = Config::instance()->get<float>("active_window_size");
				active_window.move(this->graph, RectangularRegion(translation.x(), translation.y(), active_window_size, active_window_size));

				this->active_window.feature_pool->findMatchedPairsMultiple(frames, matches, this->graph[k]->f,
					Config::instance()->get<int>("kdtree_k_mult"), Config::instance()->get<int>("kdtree_max_leaf_mult"));

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

				for (int i = 0; i < frames.size(); i++)
				{
					for (int j = 0; j < matches[i].size(); j++)
					{
						cv::KeyPoint keypoint = this->graph[k]->f.feature_pts[matches[i][j].queryIdx];
						int tN = N * keypoint.pt.x / width;
						int tM = M * keypoint.pt.y / height;
						tN = tN < 0 ? 0 : (tN >= N ? N - 1 : tN);
						tM = tM < 0 ? 0 : (tM >= M ? M - 1 : tM);

						keyframeTest[tM * N + tN]++;
					}
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

				count = matches.size();
				if (count < min_closure_candidate) min_closure_candidate = count;
				if (count > max_closure_candidate) max_closure_candidate = count;
				std::cout << ", Closure Candidate: " << count;

				count = 0;
				for (int i = 0; i < frames.size(); i++)
				{
					Eigen::Matrix4f tran;
					float rmse;
					vector<cv::DMatch> inliers;
					// find edges
					if (Feature::getTransformationByRANSAC(tran, rmse, &inliers,
						this->active_window.feature_pool, &(this->graph[k]->f), matches[i]))
					{
						// icp
						icpcuda->initICPModel((unsigned short *)this->graph[this->active_window.active_frames[frames[i]]]->depth.data, 20.0f, Eigen::Matrix4f::Identity());
						icpcuda->initICP((unsigned short *)this->graph[k]->depth.data, 20.0f);
						Eigen::Matrix4f tran2 = Eigen::Matrix4f::Identity();
						Eigen::Vector3f tra = tran2.topRightCorner(3, 1);
						Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rot = tran2.topLeftCorner(3, 3);
						int threads = Config::instance()->get<int>("icpcuda_threads");
						int blocks = Config::instance()->get<int>("icpcuda_blocks");
						icpcuda->getIncrementalTransformation(tra, rot, threads, blocks);
						tran2.topLeftCorner(3, 3) = rot;
						tran2.topRightCorner(3, 1) = tra;

						baseid.push_back(this->active_window.active_frames[frames[i]]);
						targetid.push_back(k);
						ransac_tran.push_back(tran);
						icp_tran.push_back(tran2);
						//double w = icpcuda->lastICPError > 0 ? sqrt(1.0 / icpcuda->lastICPError) : sqrt(1000000);
						// icp end
						AISNavigation::PoseGraph3D::Vertex* other_v = this->optimizer->vertex(this->active_window.active_frames[frames[i]]);

						double w = 1.0 / rmse;
						if (w < min_edge_weight) min_edge_weight = w;
						if (w > max_edge_weight) max_edge_weight = w;
						this->optimizer->addEdge(other_v, now_v, eigenToHogman(tran), Matrix6::eye(w));
						count++;
					}
				}

				if (keyframeTestCount + N * M - keyframeExistsCount < N * M * Config::instance()->get<double>("keyframe_check_P"))
				{
					key_frame_indices.insert(k);
					Eigen::Vector3f translation = TranslationFromMatrix4f(this->graph[k]->tran);
					active_window.insert(translation(0), translation(1), k);
					keyframeCount++;
					std::cout << ", Keyframe";
					isNewKeyframe = true;
				}

				std::cout << ", Edges: " << count;
				time = (clock() - start) / 1000.0;
				if (time < min_closure_detect_time) min_closure_detect_time = time;
				if (time > max_closure_detect_time) max_closure_detect_time = time;
				total_closure_detect_time += time;
				clousureCount++;
				std::cout << ", Closure: " << time;
			}

			int iteration = Config::instance()->get<int>("hogman_iterations");
			start = clock();
			this->optimizer->optimize(iteration, true);
			time = (clock() - start) / 1000.0;
			if (time < min_graph_opt_time) min_graph_opt_time = time;
			if (time > max_graph_opt_time) max_graph_opt_time = time;
			total_graph_opt_time += time;
			std::cout << ", Graph: " << fixed << setprecision(3) << time;

			for (int i = 0; i < this->graph.size(); i++)
			{
				AISNavigation::PoseGraph3D::Vertex* v = this->optimizer->vertex(i);
				Eigen::Matrix4f tran = hogmanToEigen(v->transformation);

				if (this->graph[i]->tran != tran)
				{
					Eigen::Vector3f old_translation = TranslationFromMatrix4f(this->graph[i]->tran);
					Eigen::Vector3f new_translation = TranslationFromMatrix4f(tran);
					this->graph[i]->tran = tran;

					if (key_frame_indices.find(i) != key_frame_indices.end())
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

			if (is_keyframe_candidate)
			{
				last_kc = k;
				last_kc_tran = this->graph[k]->tran;
			}
		}
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