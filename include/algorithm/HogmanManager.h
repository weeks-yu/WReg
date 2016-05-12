#pragma once
#include "QuadTree.h"
#include "Frame.h"
#include "ActiveWindow.h"
#include "graph_optimizer_hogman/graph_optimizer3d_hchol.h"

class HogmanManager
{
public:

	ActiveWindow active_window;

	vector<int> keyframe_indices;
	map<int, int> keyframe_id;
	set<int> frame_in_quadtree_indices;

	double min_graph_opt_time;
	double max_graph_opt_time;
	double total_graph_opt_time;

	double min_edge_weight;
	double max_edge_weight;

	double min_closure_detect_time;
	double max_closure_detect_time;
	double total_closure_detect_time;
	
	int min_closure_candidate;
	int max_closure_candidate;

	int keyframeInQuadTreeCount;
	int clousureCount;

#ifdef SAVE_TEST_INFOS
	vector<int> baseid;
	vector<int> targetid;
	vector<float> rmses;
	vector<int> matchescount;
	vector<int> inlierscount;
	vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> ransactrans;
#endif

private:

	AISNavigation::GraphOptimizer3D* optimizer;

	int iteration_count;

	vector<Frame*> graph;

	int last_kc;

	Eigen::Matrix4f last_kc_tran;

	vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> temp_poses;

	PointCloudCuda *pcc;
	int threads;
	int blocks;

public:

	HogmanManager(bool keyframe_only = false);

	bool addNode(Frame* frame, float weight, bool keyframe = false, string *inliers = nullptr, string *exists = nullptr);

	Eigen::Matrix4f getTransformation(int k);

	Eigen::Matrix4f getLastTransformation();

	Eigen::Matrix4f getLastKeyframeTransformation();

	int size();
};