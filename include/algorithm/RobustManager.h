#pragma once
//#include "QuadTree.h"
#include "Frame.h"

#include <g2o/types/slam3d/se3quat.h>
#include <g2o/types/slam3d/edge_se3.h>
#include <g2o/core/block_solver.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/solvers/pcg/linear_solver_pcg.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include "vertigo/vertex_switchLinear.h"
#include "vertigo/edge_switchPrior.h"
#include "vertigo/edge_se3Switchable.h"

class RobustManager
{
public:
	struct SwitchableEdge {
	public:
		VertexSwitchLinear * v_;
		EdgeSwitchPrior * ep_;
		EdgeSE3Switchable * e_;
		int id0, id1;
	};

public:

	vector<int> keyframe_for_lc;
	vector<int> keyframe_indices;
	map<int, int> keyframe_id;

	double total_kdtree_build;
	double total_kdtree_match;
	double total_loop_ransac;

	double min_graph_opt_time;
	double max_graph_opt_time;
	double total_graph_opt_time;

	double min_lc_detect_time;
	double max_lc_detect_time;
	double total_lc_detect_time;
	
	int clousureCount;

#ifdef SAVE_TEST_INFOS
	vector<int> baseid;
	vector<int> targetid;
	vector<float> rmses;
	vector<int> matchescount;
	vector<int> inlierscount;
	vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> ransactrans;
#endif

public:

	g2o::SparseOptimizer* optimizer;

	vector<Frame*> graph;

private:

	int iteration_count;

	int last_kc;

	Eigen::Matrix4f last_kc_tran;

	vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> temp_poses;

	int switchable_id;

	int aw_N, aw_M, aw_F, width, height;
	float aw_P, aw_Size;

	bool using_line_process;

	int min_matches;
	float inlier_percentage;
	float inlier_dist;
	int knn_k;

public:

	RobustManager(bool use_lp = true);

	void setUsingLineProcess(bool use) { using_line_process = use; }

	bool addNode(Frame* frame, bool keyframe = false);

	Eigen::Matrix4f getTransformation(int k);

	Eigen::Matrix4f getLastTransformation();

	Eigen::Matrix4f getLastKeyframeTransformation();

	int size();

	vector<Frame*> getGraph() { return graph; }

public:
	static Eigen::Matrix4f G2O2Matrix4f(const g2o::SE3Quat se3) {
		Eigen::Matrix4d m = se3.to_homogeneous_matrix(); //_Matrix< 4, 4, double >
		Eigen::Matrix4f mm;
		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				mm(i, j) = m(i, j);
			}
		}
		return mm;
	}

	g2o::SE3Quat Eigen2G2O(const Eigen::Matrix4f & eigen_mat) {
		Eigen::Matrix4d m;
		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				m(i, j) = eigen_mat(i, j);
			}
		}
		Eigen::Affine3d eigen_transform(m);
		Eigen::Quaterniond eigen_quat(eigen_transform.rotation());
		Eigen::Vector3d translation(m(0, 3), m(1, 3), m(2, 3));
		g2o::SE3Quat result(eigen_quat, translation);
		return result;
	}
};