#pragma once
#include "QuadTree.h"
#include "Frame.h"
#include "graph_optimizer_hogman/graph_optimizer3d_hchol.h"
#undef min
#include "ICPOdometry.h"

class GraphManager
{
public:
	struct ActiveWindow
	{
		QuadTree<int>* key_frames;
		vector<int> active_frames;
		RectangularRegion region;
		Feature* feature_pool;

		ActiveWindow()
		{
			key_frames = nullptr;
			feature_pool = nullptr;
			this->region.center_x = 0.0;
			this->region.center_y = 0.0;
			this->region.half_width = Config::instance()->get<float>("active_window_size");
			this->region.half_height = this->region.half_width;
		}
	};

public:

	ActiveWindow active_window;

	set<int> key_frame_indices;

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

	int keyframeCount;
	int clousureCount;

	vector<int> baseid;
	vector<int> targetid;
	vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> ransac_tran;
	vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> icp_tran;

private:

	AISNavigation::GraphOptimizer3D* optimizer;

	vector<Frame*> graph;

	int last_kc;

	Eigen::Matrix4f last_kc_tran;

	bool keyframeOnly;

	ICPOdometry *icpcuda;

public:

	GraphManager(bool keyframe_only = false);

	void buildQuadTree(float center_x, float center_y, float size, int capacity = 1);

	void moveActiveWindow(const RectangularRegion &region);

	bool addNode(Frame* frame, const Eigen::Matrix4f &relative_tran, float weight, bool keyframe = false, string *inliers = nullptr, string *exists = nullptr);

	Eigen::Matrix4f getTransformation(int k);

	Eigen::Matrix4f getLastTransformation();

	Eigen::Matrix4f getLastKeyframeTransformation();

	int size();

private:

	bool insertKeyframe(float x, float y, int frame_index);

};