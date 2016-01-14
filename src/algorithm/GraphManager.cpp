#include "GraphManager.h"
#include "Transformation.h"

GraphManager::GraphManager()
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
}

void GraphManager::buildQuadTree(float center_x, float center_y, float size, int capacity)
{
	if (this->active_window.key_frames != nullptr) delete this->active_window.key_frames;
	this->active_window.key_frames = new QuadTree<int>(center_x, center_y, size, nullptr, capacity);
}

bool GraphManager::insertKeyframe(float x, float y, int frame_index)
{
	if (!this->active_window.key_frames->insert(x, y, frame_index)) return false;
	return true;
}

void GraphManager::moveActiveWindow(const RectangularRegion &region)
{
	this->active_window.region = region;

	this->active_window.active_frames = this->active_window.key_frames->queryRange(this->active_window.region);

	if (this->active_window.feature_pool != nullptr) delete this->active_window.feature_pool;
	this->active_window.feature_pool = new Feature(true);
	for (int i = 0; i < this->active_window.active_frames.size(); i++)
	{
		Frame *now_f = this->graph[this->active_window.active_frames[i]];
		for (int j = 0; j < now_f->f.feature_pts_3d_real.size(); j++)
		{
			if (this->active_window.region.containsPoint(now_f->f.feature_pts_3d_real[j](0), now_f->f.feature_pts_3d_real[j](1)))
			{
				this->active_window.feature_pool->feature_pts.push_back(now_f->f.feature_pts[j]);
				this->active_window.feature_pool->feature_pts_3d.push_back(now_f->f.feature_pts_3d[j]);
				this->active_window.feature_pool->feature_pts_3d_real.push_back(now_f->f.feature_pts_3d_real[j]);
				this->active_window.feature_pool->feature_descriptors.push_back(now_f->f.feature_descriptors.row(j));
				this->active_window.feature_pool->feature_frame_index.push_back(i);
			}
		}
	}

	this->active_window.feature_pool->buildFlannIndex();
}

bool GraphManager::addNode(Frame* frame, const Eigen::Matrix4f &relative_tran, float weight, bool is_keyframe_candidate/* = false*/, string *inliers_sig, string *exists_sig)
{
	clock_t start = 0;
	double time = 0;
	int count = 0;
	bool isNewKeyframe = false;

	this->graph.push_back(frame);
	if (this->graph.size() == 1)
	{
		this->optimizer->addVertex(0, Transformation3(), 1e9 * Matrix6::eye(1.0));
		if (is_keyframe_candidate)
		{
			last_keyframe = 0;
			last_keyframe_tran = frame->tran;
			keyframeCount++;
			Eigen::Vector3f translation = TranslationFromMatrix4f(this->graph[0]->tran);
			this->insertKeyframe(translation(0), translation(1), 0);
			isNewKeyframe = true;
		}
	}
	else
	{
		int k = this->graph.size() - 1;
		AISNavigation::PoseGraph3D::Vertex *now_v = this->optimizer->addVertex(k, Transformation3(), Matrix6::eye(1.0));
		AISNavigation::PoseGraph3D::Vertex *prev_v = this->optimizer->vertex(k - 1);
		this->optimizer->addEdge(prev_v, now_v, eigenToHogman(relative_tran), Matrix6::eye(weight));

		vector<int> frames;
		vector<vector<cv::DMatch>> matches;

		if (is_keyframe_candidate)
		{
			start = clock();
			// get translation of current tran
			// move active window
			Eigen::Vector3f translation = TranslationFromMatrix4f(this->graph[k]->tran);
			float active_window_size = Config::instance()->get<float>("active_window_size");
			this->moveActiveWindow(RectangularRegion(translation.x(), translation.y(), active_window_size, active_window_size));

			this->active_window.feature_pool->findMatchedPairsMultiple(frames, matches, this->graph[k]->f,
				Config::instance()->get<int>("kdtree_k_mult"), Config::instance()->get<int>("kdtree_max_leaf_mult"));

			count = matches.size();
			if (count < min_closure_candidate) min_closure_candidate = count;
			if (count > max_closure_candidate) max_closure_candidate = count;
			std::cout << ", Closure Candidate: " << count;
			
			int N = Config::instance()->get<int>("keyframe_check_N");
			int M = Config::instance()->get<int>("keyframe_check_M");
			int width = Config::instance()->get<int>("image_width");
			int height = Config::instance()->get<int>("image_height");
			bool *keyframeTest = new bool[N * M];
			bool *keyframeExists = new bool[N * M];
			int keyframeTestCount = 0;
			int keyframeExistsCount = 0;

			for (int i = 0; i < M; i++)
			{
				for (int j = 0; j < N; j++)
				{
					keyframeTest[i * N + j] = false;
					keyframeExists[i * N + j] = false;
				}
			}

			for (int i = 0; i < this->graph[k]->f.feature_pts.size(); i++)
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
					AISNavigation::PoseGraph3D::Vertex* other_v = this->optimizer->vertex(this->active_window.active_frames[frames[i]]);
					
					double w = 1.0 / rmse;
					if (w < min_edge_weight) min_edge_weight = w;
					if (w > max_edge_weight) max_edge_weight = w;
					this->optimizer->addEdge(other_v, now_v, eigenToHogman(tran), Matrix6::eye(w));
					count++;

					for (int i = 0; i < inliers.size(); i++)
					{
						cv::KeyPoint keypoint = this->graph[k]->f.feature_pts[inliers[i].queryIdx];
						int tN = N * keypoint.pt.x / width;
						int tM = M * keypoint.pt.y / height;
						tN = tN < 0 ? 0 : (tN >= N ? N - 1 : tN);
						tM = tM < 0 ? 0 : (tM >= M ? M - 1 : tM);
						if (!keyframeTest[tM * N + tN])
						{
							keyframeTestCount++;
							keyframeTest[tM * N + tN] = true;
						}
					}
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
						*inliers_sig += keyframeTest[i * N + j] ? "1" : "0";
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
				key_frame_indices.insert(k);
				Eigen::Vector3f translation = TranslationFromMatrix4f(this->graph[k]->tran);
				this->insertKeyframe(translation(0), translation(1), k);
				last_keyframe = k;
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
			last_keyframe_tran = this->graph[k]->tran;
		}
	}
	return isNewKeyframe;
}

Eigen::Matrix4f GraphManager::getTransformation(int k)
{
	if (k < 0 || k >= graph.size())
	{
		return Eigen::Matrix4f::Identity();
	}
	return graph[k]->tran;
}

Eigen::Matrix4f GraphManager::getLastTransformation()
{
	if (graph.size() == 0)
	{
		return Eigen::Matrix4f::Identity();
	}
	return graph[graph.size() - 1]->tran;
}

Eigen::Matrix4f GraphManager::getLastKeyframeTransformation()
{
	return last_keyframe_tran;
// 	if (last_keyframe < 0 || last_keyframe >= graph.size())
// 	{
// 		return Eigen::Matrix4f::Identity();
// 	}
// 	return graph[last_keyframe]->tran;
}

int GraphManager::size()
{
	return graph.size();
}