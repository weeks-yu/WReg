#include "RobustManager.h"
#include "Transformation.h"

typedef g2o::BlockSolver< g2o::BlockSolverTraits<6, 3> >  SlamBlockSolver;
typedef g2o::LinearSolverCSparse<SlamBlockSolver::PoseMatrixType> SlamLinearCSparseSolver;
typedef g2o::LinearSolverPCG<SlamBlockSolver::PoseMatrixType> SlamLinearPCGSolver;

RobustManager::RobustManager(bool use_lp)
{
	optimizer = new g2o::SparseOptimizer();
	optimizer->setVerbose(false);
	SlamBlockSolver * solver = NULL;
	SlamLinearCSparseSolver* linearSolver = new SlamLinearCSparseSolver();
	linearSolver->setBlockOrdering(false);
	solver = new SlamBlockSolver(linearSolver);
	g2o::OptimizationAlgorithmLevenberg* algo = new g2o::OptimizationAlgorithmLevenberg(solver);
	optimizer->setAlgorithm(algo);
	switchable_id = 1 << 16;
	iteration_count = Config::instance()->get<int>("robust_iterations");
	using_line_process = use_lp;

	min_graph_opt_time = 1e8;
	max_graph_opt_time = 0;
	total_graph_opt_time = 0;

	total_kdtree_build = 0.0;
	total_kdtree_match = 0.0;
	total_loop_ransac = 0.0;

	min_lc_detect_time = 1e8;
	max_lc_detect_time = 0;
	total_lc_detect_time = 0;

	clousureCount = 0;
	min_matches = Config::instance()->get<int>("graph_min_matches");
	inlier_percentage = Config::instance()->get<float>("graph_min_inlier_p");
	inlier_dist = Config::instance()->get<float>("graph_max_inlier_dist");
	knn_k = Config::instance()->get<int>("graph_knn_k");
	width = Config::instance()->get<int>("image_width");
	height = Config::instance()->get<int>("image_height");
	aw_N = Config::instance()->get<int>("keyframe_check_N");
	aw_M = Config::instance()->get<int>("keyframe_check_M");
	aw_F = Config::instance()->get<int>("keyframe_check_F");
	aw_P = Config::instance()->get<float>("keyframe_check_P");
	aw_Size = Config::instance()->get<float>("active_window_size");

	using_line_process = true;
}

RobustManager::~RobustManager()
{

}

bool RobustManager::addNode(Frame* frame, bool is_keyframe_candidate/* = false*/)
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
		g2o::VertexSE3 * v = new g2o::VertexSE3();
		v->setId(0);
		v->setEstimate(Eigen2G2O(graph[0]->tran));
		v->setFixed(true);
		optimizer->addVertex(v);

		keyframe_id.insert(pair<int, int>(0, 0));

		last_kc = 0;
		last_kc_tran = graph[0]->tran;
		keyframe_for_lc.push_back(0);
		isNewKeyframe = true;
	}
	else
	{
		int k = graph.size() - 1;

		int *keyframeTest = new int[aw_N * aw_M];
		bool *keyframeExists = new bool[aw_N * aw_M];
		int keyframeTestCount = 0;
		int keyframeExistsCount = 0;

		for (int i = 0; i < aw_M; i++)
		{
			for (int j = 0; j < aw_N; j++)
			{
				keyframeTest[i * aw_N + j] = 0;
				keyframeExists[i * aw_N + j] = false;
			}
		}

		for (int i = 0; i < this->graph[k]->f->size(); i++)
		{
			cv::KeyPoint keypoint = this->graph[k]->f->feature_pts[i];
			int tN = aw_N * keypoint.pt.x / width;
			int tM = aw_M * keypoint.pt.y / height;
			tN = tN < 0 ? 0 : (tN >= aw_N ? aw_N - 1 : tN);
			tM = tM < 0 ? 0 : (tM >= aw_M ? aw_M - 1 : tM);
			if (!keyframeExists[tM * aw_N + tN])
			{
				keyframeExistsCount++;
				keyframeExists[tM * aw_N + tN] = true;
			}
		}

		// one kd-tree multiple match
// 		start = clock();
// 		// get translation of current tran
// 		// move active window
// 		Eigen::Vector3f current_pose = TranslationFromMatrix4f(graph[k]->tran);
// 		float r = Config::instance()->get<float>("candidate_radius");
// 		r *= r;
// 		Feature* feature_pool = new Feature(true);
// 		feature_pool->type = graph[0]->f->type;
// 		count = 0;
// 		for (int i = 0; i < keyframe_for_lc.size(); i++)
// 		{
// 			Eigen::Vector3f pose = TranslationFromMatrix4f(graph[keyframe_for_lc[i]]->tran);
// 			if ((current_pose - pose).squaredNorm() <= r)
// 			{
// 				count++;
// 				Frame *now_f = graph[keyframe_for_lc[i]];
// 				for (int j = 0; j < now_f->f->feature_pts_3d.size(); j++)
// 				{
// 					feature_pool->feature_pts.push_back(now_f->f->feature_pts[j]);
// 					feature_pool->feature_pts_3d.push_back(now_f->f->feature_pts_3d[j]);
// 					feature_pool->feature_descriptors.push_back(now_f->f->feature_descriptors.row(j));
// 					feature_pool->feature_frame_index.push_back(i);
// 				}
// 			}
// 		}
// 		feature_pool->buildFlannIndex();
// 		time = clock() - start;
// 		total_kdtree_build += time;
// 
// 		std::cout << ", KD-tree: " << time << "ms";
// 		std::cout << ", RN: " << count;
// 
// 		start = clock();
// 		vector<int> frames;
// 		vector<vector<cv::DMatch>> matches;
// 		feature_pool->findMatchedPairsMultiple(frames, matches, this->graph[k]->f, knn_k, min_matches);
// 		time = clock() - start;
// 		total_kdtree_match += time;
// 		std::cout << ", MM: " << time << "ms";
// 
// 		count = 0;
// 		g2o::VertexSE3 * v = new g2o::VertexSE3();
// 		v->setId(keyframe_indices.size() - 1);
// 		v->setEstimate(Eigen2G2O(graph[k]->tran));
// 		optimizer->addVertex(v);
// 
// 		start = clock();
// 		for (int i = 0; i < frames.size(); i++)
// 		{
// 			Eigen::Matrix4f tran;
// 			float rmse;
// 			vector<cv::DMatch> inliers;
// 
// 			// find edges
// 			if (Feature::getTransformationByRANSAC(tran, rmse, &inliers,
// 				feature_pool, graph[k]->f, matches[i], min_matches, inlier_percentage, inlier_dist))
// 			{
// 				if (using_line_process)
// 				{
// 					SwitchableEdge edge;
// 					edge.id0 = keyframe_id[keyframe_for_lc[frames[i]]];
// 					edge.id1 = keyframe_indices.size() - 1;
// 
// 					edge.v_ = new VertexSwitchLinear();
// 					edge.v_->setId(switchable_id++);
// 					edge.v_->setEstimate(1.0);
// 					optimizer->addVertex(edge.v_);
// 
// 					edge.ep_ = new EdgeSwitchPrior();
// 					edge.ep_->vertices()[0] = edge.v_;
// 					edge.ep_->setMeasurement(1.0);
// 					edge.ep_->setInformation(Eigen::Matrix<double, 1, 1>::Identity() * 1.0);
// 					optimizer->addEdge(edge.ep_);
// 
// 					edge.e_ = new EdgeSE3Switchable();
// 					edge.e_->vertices()[0] = dynamic_cast<g2o::VertexSE3*>(optimizer->vertex(edge.id0));
// 					edge.e_->vertices()[1] = dynamic_cast<g2o::VertexSE3*>(optimizer->vertex(edge.id1));
// 					edge.e_->vertices()[2] = edge.v_;
// 					edge.e_->setMeasurement(g2o::internal::fromSE3Quat(Eigen2G2O(tran)));
// 					edge.e_->setInformation(InformationMatrix::Identity());
// 					optimizer->addEdge(edge.e_);
// 				}
// 				else
// 				{
// 					g2o::EdgeSE3* g2o_edge = new g2o::EdgeSE3();
// 					g2o_edge->vertices()[0] = dynamic_cast<g2o::VertexSE3*>(optimizer->vertex(keyframe_id[keyframe_for_lc[frames[i]]]));
// 					g2o_edge->vertices()[1] = dynamic_cast<g2o::VertexSE3*>(optimizer->vertex(keyframe_indices.size() - 1));
// 					g2o_edge->setMeasurement(g2o::internal::fromSE3Quat(Eigen2G2O(tran)));
// 					g2o_edge->setInformation(InformationMatrix::Identity());
// 					optimizer->addEdge(g2o_edge);
// 				}
// #ifdef SAVE_TEST_INFOS
// 				baseid.push_back(k);
// 				targetid.push_back(keyframe_for_lc[frames[i]]);
// 				rmses.push_back(rmse);
// 				matchescount.push_back(matches[i].size());
// 				inlierscount.push_back(inliers.size());
// 				ransactrans.push_back(tran);
// #endif
// 
// 
// 
// 				count++;
// 				
// // 				if (k - keyframe_for_lc[frames[i]] > 90)
// // 				{
// // 					continue;
// // 				}
// 				for (int j = 0; j < inliers.size(); j++)
// 				{
// 					cv::KeyPoint keypoint = this->graph[k]->f->feature_pts[inliers[j].queryIdx];
// 					int tN = aw_N * keypoint.pt.x / width;
// 					int tM = aw_M * keypoint.pt.y / height;
// 					tN = tN < 0 ? 0 : (tN >= aw_N ? aw_N - 1 : tN);
// 					tM = tM < 0 ? 0 : (tM >= aw_M ? aw_M - 1 : tM);
// 
// 					keyframeTest[tM * aw_N + tN]++;
// 				}
// 			}
// 		}
// 		
// 		delete feature_pool;
// 
// 		if (keyframe_indices.size() > 1)
// 		{
// 			Eigen::Matrix4f tran = graph[k]->relative_tran;
// 			g2o::EdgeSE3* g2o_edge = new g2o::EdgeSE3();
// 			g2o_edge->vertices()[0] = dynamic_cast<g2o::VertexSE3*>(optimizer->vertex(keyframe_indices.size() - 2));
// 			g2o_edge->vertices()[1] = dynamic_cast<g2o::VertexSE3*>(optimizer->vertex(keyframe_indices.size() - 1));
// 			g2o_edge->setMeasurement(g2o::internal::fromSE3Quat(Eigen2G2O(tran)));
// 			g2o_edge->setInformation(InformationMatrix::Identity());
// 			optimizer->addEdge(g2o_edge);
// 
// #ifdef SAVE_TEST_INFOS
// 			baseid.push_back(k);
// 			targetid.push_back(keyframe_indices[keyframe_indices.size() - 2]);
// 			rmses.push_back(0.0);
// 			matchescount.push_back(0);
// 			inlierscount.push_back(0);
// 			ransactrans.push_back(tran);
// #endif
// 
// 			count++;
// 		}
// 
// 		time = clock() - start;
// 		total_loop_ransac += time;
// 		cout << ", oberservation: " << count << endl;

		// multiple kdtree multiple matches
		// get translation of current tran
		// move active window
		Eigen::Vector3f current_pose = TranslationFromMatrix4f(graph[k]->tran);
		float r = Config::instance()->get<float>("candidate_radius");
		r *= r;
		count = 0;
		int o_count = 0;

		g2o::VertexSE3 * v = new g2o::VertexSE3();
		v->setId(keyframe_indices.size() - 1);
		v->setEstimate(Eigen2G2O(graph[k]->tran));
		optimizer->addVertex(v);

		double current_kdtree_match = 0, current_ransac = 0;

		for (int i = 0; i < keyframe_for_lc.size(); i++)
		{
			Eigen::Vector3f pose = TranslationFromMatrix4f(graph[keyframe_for_lc[i]]->tran);
			if ((current_pose - pose).squaredNorm() <= r)
			{
				count++;
				
				start = clock();
				vector<cv::DMatch> matches;
				graph[keyframe_for_lc[i]]->f->findMatchedPairs(matches, this->graph[k]->f);
				time = clock() - start;
				current_kdtree_match += time;

				Eigen::Matrix4f tran;
				float rmse;
				vector<cv::DMatch> inliers;

				start = clock();
				// find edges
				if (Feature::getTransformationByRANSAC(tran, rmse, &inliers,
					graph[keyframe_for_lc[i]]->f, graph[k]->f, matches, min_matches, inlier_percentage, inlier_dist))
				{
					if (using_line_process)
					{
						SwitchableEdge edge;
						edge.id0 = keyframe_id[keyframe_for_lc[i]];
						edge.id1 = keyframe_indices.size() - 1;

						edge.v_ = new VertexSwitchLinear();
						edge.v_->setId(switchable_id++);
						edge.v_->setEstimate(1.0);
						optimizer->addVertex(edge.v_);

						edge.ep_ = new EdgeSwitchPrior();
						edge.ep_->vertices()[0] = edge.v_;
						edge.ep_->setMeasurement(1.0);
						edge.ep_->setInformation(Eigen::Matrix<double, 1, 1>::Identity() * 1.0);
						optimizer->addEdge(edge.ep_);

						edge.e_ = new EdgeSE3Switchable();
						edge.e_->vertices()[0] = dynamic_cast<g2o::VertexSE3*>(optimizer->vertex(edge.id0));
						edge.e_->vertices()[1] = dynamic_cast<g2o::VertexSE3*>(optimizer->vertex(edge.id1));
						edge.e_->vertices()[2] = edge.v_;
						edge.e_->setMeasurement(g2o::internal::fromSE3Quat(Eigen2G2O(tran)));
						edge.e_->setInformation(InformationMatrix::Identity());
						optimizer->addEdge(edge.e_);
					}
					else
					{
						g2o::EdgeSE3* g2o_edge = new g2o::EdgeSE3();
						g2o_edge->vertices()[0] = dynamic_cast<g2o::VertexSE3*>(optimizer->vertex(keyframe_id[keyframe_for_lc[i]]));
						g2o_edge->vertices()[1] = dynamic_cast<g2o::VertexSE3*>(optimizer->vertex(keyframe_indices.size() - 1));
						g2o_edge->setMeasurement(g2o::internal::fromSE3Quat(Eigen2G2O(tran)));
						g2o_edge->setInformation(InformationMatrix::Identity());
						optimizer->addEdge(g2o_edge);
					}
#ifdef SAVE_TEST_INFOS
					baseid.push_back(k);
					targetid.push_back(keyframe_for_lc[i]);
					rmses.push_back(rmse);
					matchescount.push_back(matches.size());
					inlierscount.push_back(inliers.size());
					ransactrans.push_back(tran);
#endif

					o_count++;

					for (int j = 0; j < inliers.size(); j++)
					{
						cv::KeyPoint keypoint = this->graph[k]->f->feature_pts[inliers[j].queryIdx];
						int tN = aw_N * keypoint.pt.x / width;
						int tM = aw_M * keypoint.pt.y / height;
						tN = tN < 0 ? 0 : (tN >= aw_N ? aw_N - 1 : tN);
						tM = tM < 0 ? 0 : (tM >= aw_M ? aw_M - 1 : tM);

						keyframeTest[tM * aw_N + tN]++;
					}
				}
				time = clock() - start;
				current_ransac += time;
			}
		}

		if (keyframe_indices.size() > 1)
		{
			Eigen::Matrix4f tran = graph[k]->relative_tran;
			g2o::EdgeSE3* g2o_edge = new g2o::EdgeSE3();
			g2o_edge->vertices()[0] = dynamic_cast<g2o::VertexSE3*>(optimizer->vertex(keyframe_indices.size() - 2));
			g2o_edge->vertices()[1] = dynamic_cast<g2o::VertexSE3*>(optimizer->vertex(keyframe_indices.size() - 1));
			g2o_edge->setMeasurement(g2o::internal::fromSE3Quat(Eigen2G2O(tran)));
			g2o_edge->setInformation(InformationMatrix::Identity());
			optimizer->addEdge(g2o_edge);

#ifdef SAVE_TEST_INFOS
			baseid.push_back(k);
			targetid.push_back(keyframe_indices[keyframe_indices.size() - 2]);
			rmses.push_back(0.0);
			matchescount.push_back(0);
			inlierscount.push_back(0);
			ransactrans.push_back(tran);
#endif

			o_count++;
		}

		total_kdtree_match += current_kdtree_match;
		total_loop_ransac += current_ransac;

		std::cout << ", RN: " << count;
		std::cout << ", MM: " << current_kdtree_match;
		std::cout << ", oberservation: " << o_count << std::endl;

		start = clock();
		optimizer->initializeOptimization();
		optimizer->optimize(iteration_count);
		time = clock() - start;
		total_graph_opt_time += time;
		std::cout << ", Graph: " << time;

		keyframe_id.insert(pair<int, int>(k, keyframe_indices.size() - 1));

		cout << endl;
		for (int i = 0; i < aw_M; i++)
		{
			for (int j = 0; j < aw_N; j++)
			{
				cout << keyframeTest[i * aw_N + j] << " ";
				if (keyframeTest[i * aw_N + j] >= aw_F)
					keyframeTestCount++;
			}
		}
		cout << endl;

		delete keyframeTest;
		delete keyframeExists;

		if (keyframeTestCount + aw_N * aw_M - keyframeExistsCount < aw_N * aw_M * aw_P)
		{
			keyframe_for_lc.push_back(k);
			Eigen::Vector3f translation = TranslationFromMatrix4f(this->graph[k]->tran);
			std::cout << ", Keyframe";
			isNewKeyframe = true;
		}

		Eigen::Matrix4f last_kf_tran;
		for (int i = 0; i < this->graph.size(); i++)
		{
			if (keyframe_id.find(i) != keyframe_id.end())
			{
				g2o::VertexSE3 *v = dynamic_cast<g2o::VertexSE3*>(optimizer->vertex(keyframe_id[i]));
				temp_poses[i] = G2O2Matrix4f(v->estimateAsSE3Quat());
				last_kf_tran = temp_poses[i];
			}
			else
			{
				temp_poses[i] = last_kf_tran * graph[i]->relative_tran;
			}
		}

		for (int i = 0; i < this->graph.size(); i++)
		{
			this->graph[i]->tran = temp_poses[i];
		}
			
		last_kc = k;
		last_kc_tran = this->graph[k]->tran;
	}

	return isNewKeyframe;
}

Eigen::Matrix4f RobustManager::getTransformation(int k)
{
	if (k < 0 || k >= graph.size())
	{
		return Eigen::Matrix4f::Identity();
	}
	return graph[k]->tran;
}

Eigen::Matrix4f RobustManager::getLastTransformation()
{
	if (graph.size() == 0)
	{
		return Eigen::Matrix4f::Identity();
	}
	return graph[graph.size() - 1]->tran;
}

Eigen::Matrix4f RobustManager::getLastKeyframeTransformation()
{
	if (keyframe_indices.size() <= 0)
	{
		return Eigen::Matrix4f::Identity();
	}
	return graph[keyframe_indices[keyframe_indices.size() - 1]]->tran;
}

void RobustManager::setParameters(void **params)
{
	min_matches = *static_cast<int *>(params[0]);
	inlier_percentage = *static_cast<float *>(params[1]);
	inlier_dist = *static_cast<float *>(params[2]);
}
