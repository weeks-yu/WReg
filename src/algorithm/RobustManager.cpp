#include "RobustManager.h"
#include "Transformation.h"

typedef g2o::BlockSolver< g2o::BlockSolverTraits<6, 3> >  SlamBlockSolver;
typedef g2o::LinearSolverCSparse<SlamBlockSolver::PoseMatrixType> SlamLinearCSparseSolver;
typedef g2o::LinearSolverPCG<SlamBlockSolver::PoseMatrixType> SlamLinearPCGSolver;

RobustManager::RobustManager(bool use_lp)
{
	optimizer = new g2o::SparseOptimizer();
	optimizer->setVerbose(true);
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

	min_edge_weight = 1e10;
	max_edge_weight = 0;

	min_closure_detect_time = 1e8;
	max_closure_detect_time = 0;
	total_closure_detect_time = 0;
	
	min_closure_candidate = 1000;
	max_closure_candidate = 0;

//	keyframeLoopCandidateCount = 0;
	clousureCount = 0;

	threads = Config::instance()->get<int>("icpcuda_threads");
	blocks = Config::instance()->get<int>("icpcuda_blocks");
	width = Config::instance()->get<int>("image_width");
	height = Config::instance()->get<int>("image_height");
	float cx = Config::instance()->get<float>("camera_cx");
	float cy = Config::instance()->get<float>("camera_cy");
	float fx = Config::instance()->get<float>("camera_fx");
	float fy = Config::instance()->get<float>("camera_fy");
	float depthFactor = Config::instance()->get<float>("depth_factor");
	float distThresh = Config::instance()->get<float>("dist_threshold");
	squared_dist_threshold = distThresh * distThresh;
	float angleThresh = Config::instance()->get<float>("angle_threshold");
	pcc = nullptr;

	aw_N = Config::instance()->get<int>("keyframe_check_N");
	aw_M = Config::instance()->get<int>("keyframe_check_M");
	aw_F = Config::instance()->get<int>("keyframe_check_F");
	aw_P = Config::instance()->get<float>("keyframe_check_P");
	aw_Size = Config::instance()->get<float>("active_window_size");

	using_line_process = true;
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
		odometry_traj.push_back(Eigen::Matrix4f::Identity());
		odometry_info.push_back(Eigen::Matrix<double, 6, 6>::Identity());

		keyframe_id.insert(pair<int, int>(0, 0));

		last_kc = 0;
		last_kc_tran = graph[0]->tran;
		keyframe_for_lc.push_back(0);
//		keyframeLoopCandidateCount++;
//		frame_in_quadtree_indices.insert(0);
//		Eigen::Vector3f translation = TranslationFromMatrix4f(graph[0]->tran);
//		active_window.build(0.0, 0.0, Config::instance()->get<float>("quadtree_size"), 4);
//		if (!active_window.insert(translation(0), translation(1), 0))
//			insertion_failure.push_back(pair<float, float>(translation(0), translation(1)));
//		active_window.move(graph, RectangularRegion(translation(0), translation(1), aw_Size, aw_Size));
		isNewKeyframe = true;
		weight = 0.0;
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

		// individually
// 		Eigen::Vector3f current_pose = TranslationFromMatrix4f(graph[k]->tran);
// 		float r = Config::instance()->get<float>("candidate_radius");
// 		r *= r;
// 		vector<cv::DMatch> matches, inliers;
// 
// 		count = 0;
// 		float totaltime = 0;
// 		g2o::VertexSE3 * v = new g2o::VertexSE3();
// 		v->setId(keyframe_indices.size() - 1);
// 		v->setEstimate(Eigen2G2O(graph[k]->tran));
// 		optimizer->addVertex(v);
// 
// 		int max_leaf = Config::instance()->get<int>("kdtree_max_leaf");
// 		for (int i = 0; i < keyframe_for_lc.size(); i++)
// 		{
// 			Eigen::Vector3f pose = TranslationFromMatrix4f(graph[keyframe_for_lc[i]]->tran);
// 			if ((current_pose - pose).squaredNorm() <= r)
// 			{
// 				start = clock();
// 				matches.clear();
// 				inliers.clear();
// 				graph[keyframe_for_lc[i]]->f->findMatchedPairs(matches, graph[k]->f, max_leaf);
// 
// 				totaltime += clock() - start;
// 
// 				Eigen::Matrix4f tran;
// 				Eigen::Matrix<double, 6, 6> information;
// 				float rmse, coresp;
// //				pcc->initPrev((unsigned short *)this->graph[keyframe_for_lc[frames[i]]]->depth->data, 20.0f);
// 				// find edges
// 				if (Feature::getTransformationByRANSAC(tran, information, coresp, rmse, &inliers,
// 					graph[keyframe_for_lc[i]]->f, graph[k]->f, pcc, matches))
// 				{
// 					//				if (keyframe_id[keyframe_for_lc[frames[i]]] != keyframe_indices.size() - 2)
// 					{
// 						SwitchableEdge edge;
// 						/*				edge.t_ = &t;*/
// 						edge.id0 = keyframe_id[keyframe_for_lc[i]];
// 						edge.id1 = keyframe_indices.size() - 1;
// 
// 						edge.v_ = new VertexSwitchLinear();
// 						edge.v_->setId(switchable_id++);
// 						edge.v_->setEstimate(1.0);
// 						optimizer->addVertex(edge.v_);
// 
// 						edge.ep_ = new EdgeSwitchPrior();
// 						edge.ep_->vertices()[0] = edge.v_;
// 						edge.ep_->setMeasurement(1.0);
// 						edge.ep_->setInformation(Eigen::Matrix<double, 1, 1>::Identity() * 1.0);
// 						optimizer->addEdge(edge.ep_);
// 
// 						edge.e_ = new EdgeSE3Switchable();
// 						edge.e_->vertices()[0] = dynamic_cast<g2o::VertexSE3*>(optimizer->vertex(edge.id0));
// 						edge.e_->vertices()[1] = dynamic_cast<g2o::VertexSE3*>(optimizer->vertex(edge.id1));
// 						edge.e_->vertices()[2] = edge.v_;
// 						edge.e_->setMeasurement(g2o::internal::fromSE3Quat(Eigen2G2O(tran)));
// 						edge.e_->setInformation(information);
// 						optimizer->addEdge(edge.e_);
// 						loop_edges.push_back(edge);
// 						loop_trans.push_back(tran);
// 						loop_info.push_back(information);
// 
// #ifdef SAVE_TEST_INFOS
// 						baseid.push_back(k);
// 						targetid.push_back(keyframe_for_lc[i]);
// 						rmses.push_back(rmse);
// 						matchescount.push_back(matches.size());
// 						inlierscount.push_back(inliers.size());
// 						ransactrans.push_back(tran);
// #endif
// 
// 						count++;
// 					}
// 
// 					// 				if (k - keyframe_for_lc[frames[i]] > 200)
// 					// 				{
// 					// 					continue;
// 					// 				}
// 					for (int j = 0; j < inliers.size(); j++)
// 					{
// 						cv::KeyPoint keypoint = this->graph[k]->f->feature_pts[inliers[j].queryIdx];
// 						int tN = aw_N * keypoint.pt.x / width;
// 						int tM = aw_M * keypoint.pt.y / height;
// 						tN = tN < 0 ? 0 : (tN >= aw_N ? aw_N - 1 : tN);
// 						tM = tM < 0 ? 0 : (tM >= aw_M ? aw_M - 1 : tM);
// 
// 						keyframeTest[tM * aw_N + tN]++;
// 					}
// 				}
// 			}
// 		}
// 
// 		std::cout << ", IM: " << totaltime << "ms";
// 
// 		if (keyframe_indices.size() > 1)
// 		{
// 			//			pcc->initPrev((unsigned short *)this->graph[keyframe_indices[keyframe_indices.size() - 2]]->depth->data, 20.0f);
// 			Eigen::Matrix4f tran = graph[k]->relative_tran;
// 			InfomationMatrix information;
// 			//			Eigen::Vector3f t = tran.topRightCorner(3, 1);
// 			//			Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rot = tran.topLeftCorner(3, 3);
// 			//			int point_count, point_corr_count;
// 			//			pcc->getCoresp(t, rot, information, point_count, point_corr_count, threads, blocks);
// 
// 			g2o::EdgeSE3* g2o_edge = new g2o::EdgeSE3();
// 			g2o_edge->vertices()[0] = dynamic_cast<g2o::VertexSE3*>(optimizer->vertex(keyframe_indices.size() - 2));
// 			g2o_edge->vertices()[1] = dynamic_cast<g2o::VertexSE3*>(optimizer->vertex(keyframe_indices.size() - 1));
// 			g2o_edge->setMeasurement(g2o::internal::fromSE3Quat(Eigen2G2O(tran)));
// 			g2o_edge->setInformation(Eigen::Matrix< double, 6, 6 >::Identity());
// 			optimizer->addEdge(g2o_edge);
// 
// 			odometry_traj.push_back(tran);
// 			odometry_info.push_back(information);
// 
// #ifdef SAVE_TEST_INFOS
// 			baseid.push_back(k);
// 			targetid.push_back(keyframe_indices[keyframe_indices.size() - 2]);
// 			rmses.push_back(0.0);
// 			matchescount.push_back(0);
// 			inlierscount.push_back(0);
// 			ransactrans.push_back(tran);
// #endif
// 			// 			if (frame_in_quadtree_indices.find(keyframe_indices[keyframe_indices.size() - 2]) == frame_in_quadtree_indices.end())
// 			// 				delete this->graph[keyframe_indices[keyframe_indices.size() - 2]]->depth;
// 			count++;
// 		}



		// one kd-tree multiple match
		start = clock();
		// get translation of current tran
		// move active window
		Eigen::Vector3f current_pose = TranslationFromMatrix4f(graph[k]->tran);
		// 		float active_window_size = Config::instance()->get<float>("active_window_size");
		// 		active_window.move(this->graph, RectangularRegion(translation.x(), translation.y(), active_window_size, active_window_size));

		float r = Config::instance()->get<float>("candidate_radius");
		r *= r;
		Feature* feature_pool = new Feature(true);
		feature_pool->type = graph[0]->f->type;
		count = 0;
		for (int i = 0; i < keyframe_for_lc.size(); i++)
		{
			Eigen::Vector3f pose = TranslationFromMatrix4f(graph[keyframe_for_lc[i]]->tran);
			if ((current_pose - pose).squaredNorm() <= r)
			{
				count++;
				Frame *now_f = graph[keyframe_for_lc[i]];
				for (int j = 0; j < now_f->f->feature_pts_3d.size(); j++)
				{
//					feature_pool->feature_ids.push_back(now_f->f->feature_ids[j]);
					feature_pool->feature_pts.push_back(now_f->f->feature_pts[j]);
					feature_pool->feature_pts_3d.push_back(now_f->f->feature_pts_3d[j]);
//					feature_pool->feature_pts_3d_real.push_back(now_f->f->feature_pts_3d_real[j]);
					feature_pool->feature_descriptors.push_back(now_f->f->feature_descriptors.row(j));
					feature_pool->feature_frame_index.push_back(i);
				}
			}
		}
		feature_pool->buildFlannIndex();

		std::cout << ", KD-tree: " << clock() - start << "ms";
		std::cout << ", RN: " << count;

		start = clock();
		vector<int> frames;
		vector<vector<cv::DMatch>> matches;
		feature_pool->findMatchedPairsMultiple(frames, matches, this->graph[k]->f);
		std::cout << ", MM: " << clock() - start << "ms";

		count = matches.size();
		if (count < min_closure_candidate) min_closure_candidate = count;
		if (count > max_closure_candidate) max_closure_candidate = count;
		std::cout << ", LC Candidate: " << count;

		count = 0;
		g2o::VertexSE3 * v = new g2o::VertexSE3();
		v->setId(keyframe_indices.size() - 1);
		v->setEstimate(Eigen2G2O(graph[k]->tran));
		optimizer->addVertex(v);

		Eigen::Matrix<double, 6, 6> information;
		for (int i = 0; i < frames.size(); i++)
		{
			Eigen::Matrix4f tran;
			float rmse;
			int pc, pcorrc;
			vector<cv::DMatch> inliers;

			// find edges
			if (Feature::getTransformationByRANSAC(tran, information, pc, pcorrc, rmse, &inliers,
				feature_pool, graph[k]->f, nullptr, matches[i]))
			{
//				if (keyframe_id[keyframe_for_lc[frames[i]]] != keyframe_indices.size() - 2)
				if (using_line_process)
				{
					SwitchableEdge edge;
					/*				edge.t_ = &t;*/
					edge.id0 = keyframe_id[keyframe_for_lc[frames[i]]];
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
					edge.e_->setInformation(/*information*/InformationMatrix::Identity());
					optimizer->addEdge(edge.e_);
					loop_edges.push_back(edge);
					loop_trans.push_back(tran);
					loop_info.push_back(/*information*/InformationMatrix::Identity());
				}
				else
				{
					g2o::EdgeSE3* g2o_edge = new g2o::EdgeSE3();
					g2o_edge->vertices()[0] = dynamic_cast<g2o::VertexSE3*>(optimizer->vertex(keyframe_id[keyframe_for_lc[frames[i]]]));
					g2o_edge->vertices()[1] = dynamic_cast<g2o::VertexSE3*>(optimizer->vertex(keyframe_indices.size() - 1));
					g2o_edge->setMeasurement(g2o::internal::fromSE3Quat(Eigen2G2O(tran)));
					g2o_edge->setInformation(/*information*/InformationMatrix::Identity());
					optimizer->addEdge(g2o_edge);
				}
#ifdef SAVE_TEST_INFOS
				baseid.push_back(k);
				targetid.push_back(keyframe_for_lc[frames[i]]);
				rmses.push_back(rmse);
				matchescount.push_back(matches[i].size());
				inlierscount.push_back(inliers.size());
				ransactrans.push_back(tran);
#endif

				count++;
				
				if (k - keyframe_for_lc[frames[i]] > 90)
				{
					continue;
				}
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
		}
		
		delete feature_pool;

		if (keyframe_indices.size() > 1)
		{
			Eigen::Matrix4f tran = graph[k]->relative_tran;
		
// 			if (graph[k]->ransac_failed)
// 			{
// 				std::cout << ", failed edge";
// 				SwitchableEdge edge;
// 				/*				edge.t_ = &t;*/
// 				edge.id0 = keyframe_indices.size() - 2;
// 				edge.id1 = keyframe_indices.size() - 1;
// 
// 				edge.v_ = new VertexSwitchLinear();
// 				edge.v_->setId(switchable_id++);
// 				edge.v_->setEstimate(1.0);
// 				optimizer->addVertex(edge.v_);
// 
// 				edge.ep_ = new EdgeSwitchPrior();
// 				edge.ep_->vertices()[0] = edge.v_;
// 				edge.ep_->setMeasurement(1.0);
// 				edge.ep_->setInformation(Eigen::Matrix<double, 1, 1>::Identity() * 1.0);
// 				optimizer->addEdge(edge.ep_);
// 
// 				edge.e_ = new EdgeSE3Switchable();
// 				edge.e_->vertices()[0] = dynamic_cast<g2o::VertexSE3*>(optimizer->vertex(edge.id0));
// 				edge.e_->vertices()[1] = dynamic_cast<g2o::VertexSE3*>(optimizer->vertex(edge.id1));
// 				edge.e_->vertices()[2] = edge.v_;
// 				edge.e_->setMeasurement(g2o::internal::fromSE3Quat(Eigen2G2O(tran)));
// 				edge.e_->setInformation(/*information*/InformationMatrix::Identity());
// 				optimizer->addEdge(edge.e_);
// 				failed_edges.push_back(edge);
// 				failed_trans.push_back(tran);
// 				failed_info.push_back(/*information*/InformationMatrix::Identity());
// 			}
// 			else
			{
				g2o::EdgeSE3* g2o_edge = new g2o::EdgeSE3();
				g2o_edge->vertices()[0] = dynamic_cast<g2o::VertexSE3*>(optimizer->vertex(keyframe_indices.size() - 2));
				g2o_edge->vertices()[1] = dynamic_cast<g2o::VertexSE3*>(optimizer->vertex(keyframe_indices.size() - 1));
				g2o_edge->setMeasurement(g2o::internal::fromSE3Quat(Eigen2G2O(tran)));
				g2o_edge->setInformation(/*information*/InformationMatrix::Identity());
				optimizer->addEdge(g2o_edge);

				odometry_traj.push_back(tran);
				odometry_info.push_back(/*information*/InformationMatrix::Identity());
			}
			

#ifdef SAVE_TEST_INFOS
			baseid.push_back(k);
			targetid.push_back(keyframe_indices[keyframe_indices.size() - 2]);
			rmses.push_back(0.0);
			matchescount.push_back(0);
			inlierscount.push_back(0);
			ransactrans.push_back(tran);
#endif

			count++;
		}

		cout << ", oberservation: " << count << endl;

		start = clock();
		optimizer->initializeOptimization();
		optimizer->optimize(iteration_count);

		keyframe_id.insert(pair<int, int>(k, keyframe_indices.size() - 1));
		time = (clock() - start) / 1000.0;
		if (time < min_graph_opt_time) min_graph_opt_time = time;
		if (time > max_graph_opt_time) max_graph_opt_time = time;
		total_graph_opt_time += time;
		std::cout << ", Graph: " << fixed << setprecision(3) << time;

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
//			frame_in_quadtree_indices.insert(k);
			Eigen::Vector3f translation = TranslationFromMatrix4f(this->graph[k]->tran);
// 			if (!active_window.insert(translation(0), translation(1), k))
// 				insertion_failure.push_back(pair<float, float>(translation(0), translation(1)));
// 			active_window.move(graph, RectangularRegion(translation(0), translation(1), aw_Size, aw_Size));
// 			keyframeLoopCandidateCount++;
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

		int ww = 0;
		for (int i = 0; i < this->graph.size(); i++)
		{
			if (keyframe_for_lc[ww] == i)
			{
// 				if (this->graph[i]->tran != temp_poses[i])
// 				{
// 					this->graph[i]->f->updateFeaturePoints3DReal(temp_poses[i]);
// 				}
				ww++;
			}
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

int RobustManager::size()
{
	return graph.size();
}

void RobustManager::refine()
{
	cout << endl << "Refine Start" << endl;

	g2o::SparseOptimizer* opt;
	opt = new g2o::SparseOptimizer();
	opt->setVerbose(true);
	SlamBlockSolver * solver = NULL;
	SlamLinearCSparseSolver* linearSolver = new SlamLinearCSparseSolver();
	linearSolver->setBlockOrdering(false);
	solver = new SlamBlockSolver(linearSolver);
	g2o::OptimizationAlgorithmLevenberg* algo = new g2o::OptimizationAlgorithmLevenberg(solver);
	opt->setAlgorithm(algo);
	//iteration_count = Config::instance()->get<int>("robust_iterations");

	for (int i = 0; i < keyframe_indices.size(); i++)
	{
		g2o::VertexSE3 *v = new g2o::VertexSE3();
		v->setId(i);
		if (i == 0)
		{
			v->setFixed(true);
		}
		v->setEstimate(Eigen2G2O(graph[keyframe_indices[i]]->tran));
		opt->addVertex(v);

		if (i > 0)
		{
			g2o::EdgeSE3* g2o_edge = new g2o::EdgeSE3();
			g2o_edge->vertices()[0] = dynamic_cast<g2o::VertexSE3*>(opt->vertex(i - 1));
			g2o_edge->vertices()[1] = dynamic_cast<g2o::VertexSE3*>(opt->vertex(i));
			g2o_edge->setMeasurement(g2o::internal::fromSE3Quat(Eigen2G2O(odometry_traj[i])));
			g2o_edge->setInformation(odometry_info[i]);
			opt->addEdge(g2o_edge);
		}
	}

	for (int i = 0; i < loop_edges.size(); i++)
	{
		if (loop_edges[i].v_->estimate() > 0.25)
		{
			g2o::EdgeSE3* g2o_edge = new g2o::EdgeSE3();
			g2o_edge->vertices()[0] = dynamic_cast<g2o::VertexSE3*>(opt->vertex(loop_edges[i].id0));
			g2o_edge->vertices()[1] = dynamic_cast<g2o::VertexSE3*>(opt->vertex(loop_edges[i].id1));
			g2o_edge->setMeasurement(g2o::internal::fromSE3Quat(Eigen2G2O(loop_trans[i])));
			g2o_edge->setInformation(loop_info[i]);
			opt->addEdge(g2o_edge);
		}
	}

	opt->initializeOptimization();
	opt->optimize(iteration_count);

	cout << "optimization finished" << endl;

	for (int i = 0; i < this->graph.size(); i++)
	{
		if (keyframe_id.find(i) != keyframe_id.end())
		{
			g2o::VertexSE3 *v = dynamic_cast<g2o::VertexSE3*>(optimizer->vertex(keyframe_id[i]));
			graph[i]->tran = G2O2Matrix4f(v->estimateAsSE3Quat());
		}
		else
		{
			graph[i]->tran = graph[i - 1]->tran  * graph[i]->relative_tran;
		}
	}
	cout << "updated transformations" << endl;
}

void RobustManager::getLineProcessResult(vector<int> &id0s, vector<int> &id1s, vector<float> &linep)
{
	id0s.clear();
	id1s.clear();
	linep.clear();
	for (int i = 0; i < loop_edges.size(); i++)
	{
		id0s.push_back(keyframe_indices[loop_edges[i].id0]);
		id1s.push_back(keyframe_indices[loop_edges[i].id1]);
		linep.push_back(loop_edges[i].v_->estimate());
	}
}