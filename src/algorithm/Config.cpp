#include "Config.h"

using namespace std;

Config* Config::_instance = nullptr;

Config::Config() {
	config["feature_type"]			= static_cast<std::string>("surf");
	config["graph_feature_type"]    = static_cast<std::string>("surf");
	config["downsample_rate"]		= static_cast<float>  (0.01);

	// RANSAC
	config["candidate_number"]		= static_cast<int>    (10);
	config["matches_criterion"]		= static_cast<float>  (0.8);
	config["ransac_max_iteration"]	= static_cast<int>    (1000);
	config["min_matches"]			= static_cast<int>    (40);
	config["min_inlier_p"]			= static_cast<float>  (0.3);
	config["max_inlier_dist"]		= static_cast<float>  (0.1);

	// Graph
	config["graph_min_matches"]		= static_cast<int>    (40);
	config["graph_min_inlier_p"]	= static_cast<float>  (0.3);
	config["graph_max_inlier_dist"]	= static_cast<float>  (0.1);
	config["graph_knn_k"]			= static_cast<int> (30);

	// Keyframe
	config["max_keyframe_interval"] = static_cast<int>    (30);
	config["keyframe_rational"]		= static_cast<float>  (0.5);
	config["min_translation_meter"] = static_cast<float>  (0.25);
	config["min_rotation_degree"]	= static_cast<float>  (15.0);

	// KDTree
	config["kdtree_trees"]			= static_cast<int>    (4);
	config["kdtree_max_leaf"]		= static_cast<int>    (64);
	config["kdtree_max_leaf_mult"]	= static_cast<int>    (256);
	config["kdtree_k_mult"]			= static_cast<int>    (30);

	// Quad Tree
	config["quadtree_size"]			= static_cast<float>  (20.0);
	config["active_window_size"]	= static_cast<float>  (2.5);
	config["candidate_radius"]		= static_cast<float>  (1.0);
	config["keyframe_check_N"]		= static_cast<int>    (4);
	config["keyframe_check_M"]		= static_cast<int>    (4);
	config["keyframe_check_F"]		= static_cast<int>    (1);
	config["keyframe_check_P"]		= static_cast<float>  (0.75);

 	// Robust
	config["robust_iterations"]		= static_cast<int> (10);

	// Camera parameters
	config["image_width"]			= static_cast<int>    (640);
	config["image_height"]			= static_cast<int>    (480);
	config["camera_fx"]				= static_cast<float>  (517.3);
	config["camera_fy"]				= static_cast<float>  (516.5);
	config["camera_cx"]				= static_cast<float>  (318.6);
	config["camera_cy"]				= static_cast<float>  (255.3);
// 	config["camera_fx"]				= static_cast<float>  (481.2);
// 	config["camera_fy"]				= static_cast<float>  (-480.0);
// 	config["camera_cx"]				= static_cast<float>  (319.5);
// 	config["camera_cy"]				= static_cast<float>  (239.5);
// 	config["camera_fx"]				= static_cast<float>  (525.0);
// 	config["camera_fy"]				= static_cast<float>  (525.0);
// 	config["camera_cx"]				= static_cast<float>  (319.5);
// 	config["camera_cy"]				= static_cast<float>  (239.5);

	config["depth_factor"]			= static_cast<float>  (5000.0);
//	config["depth_factor"]			= static_cast<float>  (1000.0);

	// cuda parameters
	config["icpcuda_threads"]		= static_cast<int>    (256);		// warpSize(32)的倍数
	config["icpcuda_blocks"]		= static_cast<int>    (80);			// 480 * 640 / 80 / 256 = 15 整数
	config["dist_threshold"]		= static_cast<float>  (0.1);
	config["angle_threshold"]		= static_cast<float>  (0.34202);
}

Config* Config::instance() {
	if (_instance == nullptr) {
		_instance = new Config();
	}
	return _instance;
}

