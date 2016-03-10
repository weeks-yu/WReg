#pragma once
#include "srba.h"
#include "QuadTree.h"
#include "Frame.h"
#include "ActiveWindow.h"


using namespace srba;

class SrbaManager
{
public:
	struct RBA_OPTIONS : public RBA_OPTIONS_DEFAULT
	{
		//	typedef ecps::local_areas_fixed_size            edge_creation_policy_t;  //!< One of the most important choices: how to construct the relative coordinates graph problem
		//	typedef options::sensor_pose_on_robot_none      sensor_pose_on_robot_t;  //!< The sensor pose coincides with the robot pose
		//	typedef options::observation_noise_identity     obs_noise_matrix_t;      //!< The sensor noise matrix is the same for all observations and equal to \sigma * I(identity)
		//	typedef options::solver_LM_schur_dense_cholesky solver_t;                //!< Solver algorithm (Default: Lev-Marq, with Schur, with dense Cholesky)
	};

	typedef RbaEngine <
		kf2kf_poses::SE3,             // Parameterization  of KF-to-KF poses
		landmarks::Euclidean3D,       // Parameterization of landmark positions
		observations::Cartesian_3D,   // Type of observations
		RBA_OPTIONS
	> SrbaT;

	struct RBA_OPTIONS_GRAPH : public RBA_OPTIONS_DEFAULT
	{
		//	typedef ecps::local_areas_fixed_size            edge_creation_policy_t;  //!< One of the most important choices: how to construct the relative coordinates graph problem
		//	typedef options::sensor_pose_on_robot_none      sensor_pose_on_robot_t;  //!< The sensor pose coincides with the robot pose
		typedef options::observation_noise_constant_matrix<observations::RelativePoses_3D>   obs_noise_matrix_t;      // The sensor noise matrix is the same for all observations and equal to some given matrix
		//	typedef options::solver_LM_schur_dense_cholesky solver_t;                //!< Solver algorithm (Default: Lev-Marq, with Schur, with dense Cholesky)
	};

	typedef RbaEngine <
		kf2kf_poses::SE3,               // Parameterization  of KF-to-KF poses
		landmarks::RelativePoses3D,     // Parameterization of landmark positions
		observations::RelativePoses_3D, // Type of observations
		RBA_OPTIONS_GRAPH
	> SrbaGraphT;

public:
	SrbaT rba;
	SrbaGraphT rba_graph;

	bool using_graph_slam;

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

	vector<int> baseid;
	vector<int> targetid;
	vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> ransac_tran;
	vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> icp_tran;

private:

	vector<Frame*> graph;

	int last_kc;

	Eigen::Matrix4f last_kc_tran;

	vector<bool> is_keyfram_pose_set;
	vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> keyframe_poses;

public:

	SrbaManager();

	bool addNode(Frame* frame, float weight, bool keyframe = false, string *inliers = nullptr, string *exists = nullptr);

	Eigen::Matrix4f getTransformation(int k);

	Eigen::Matrix4f getLastTransformation();

	Eigen::Matrix4f getLastKeyframeTransformation();

	int size();

	bool visit_filter_feat(const TLandmarkID lm_ID, const topo_dist_t cur_dist);

	void visit_feat(const TLandmarkID lm_ID, const topo_dist_t cur_dist);

	bool visit_filter_kf(const TKeyFrameID kf_ID, const topo_dist_t cur_dist);

	void visit_kf(const TKeyFrameID kf_ID, const topo_dist_t cur_dist);

	bool visit_filter_k2k(
		const TKeyFrameID current_kf,
		const TKeyFrameID next_kf,
		const SrbaGraphT::k2k_edge_t* edge,
		const topo_dist_t cur_dist);

	void visit_k2k(
		const TKeyFrameID current_kf,
		const TKeyFrameID next_kf,
		const SrbaGraphT::k2k_edge_t* edge,
		const topo_dist_t cur_dist);

	bool visit_filter_k2f(
		const TKeyFrameID current_kf,
		const SrbaGraphT::k2f_edge_t* edge,
		const topo_dist_t cur_dist);

	void visit_k2f(
		const TKeyFrameID current_kf,
		const SrbaGraphT::k2f_edge_t *edge,
		const topo_dist_t cur_dist);
};