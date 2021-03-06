#include "Config.h"

using namespace std;

Config* Config::_instance = nullptr;

Config::Config() {
	config["feature_type"]			= std::string("ORB");
	config["graph_feature_type"]    = std::string("SIFT");

	// RANSAC
	config["candidate_number"]		= static_cast<int>    (10);
	config["min_matches"]			= static_cast<int>    (40);
	config["min_inliers_percent"]	= static_cast<float>  (0.3);
	config["max_dist_for_inliers"]	= static_cast<float>  (0.1);
	config["matches_criterion"]		= static_cast<float>  (0.8);
	config["coresp_percent"]		= static_cast<float>  (0.3);
	config["ransac_max_iteration"]	= static_cast<int>    (1000);

	// Keyframe
	config["max_keyframe_interval"] = static_cast<int>    (15);
	config["keyframe_rational"]		= static_cast<float>  (0.4);
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

	// HOG-man
	config["hogman_iterations"]		= static_cast<int>    (10);

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

	// hogman parameters
	config["graph_levels"]			= static_cast<int>    (3);
	config["node_distance"]			= static_cast<int>    (2);

	// cuda parameters
	config["icpcuda_threads"]		= static_cast<int>    (256);		// warpSize(32)�ı���
	config["icpcuda_blocks"]		= static_cast<int>    (80);			// 480 * 640 / 80 / 256 = 15 ����
	config["dist_threshold"]		= static_cast<float>  (0.1);
	config["angle_threshold"]		= static_cast<float>  (0.34202);

	// plane fitting
	config["plane_max_iteration"]	= static_cast<int>    (20);
	config["plane_dist_threshold"]	= static_cast<float>  (0.02);

// 	config["start_paused"]                 =  static_cast<bool>  (1);
// 	config["subscriber_queue_size"]        =  static_cast<int>   (20);
// 	config["publisher_queue_size"]         =  static_cast<int>   (1);
// 	config["adjuster_max_keypoints"]       =  static_cast<int>   (1800);
// 	config["adjuster_min_keypoints"]       =  static_cast<int>   (1000);
 	
// 	config["fast_adjuster_max_iterations"] =  static_cast<int>   (10);
// 	config["surf_adjuster_max_iterations"] =  static_cast<int>   (5);
// 	config["min_translation_meter"]        =  static_cast<double>(0.1);
// 	config["min_rotation_degree"]          =  static_cast<int>   (5);
// 	config["min_time_reported"]            =  static_cast<double>(0.01);
// 	config["squared_meshing_threshold"]    =  static_cast<double>(0.0009);
// 	config["use_glwidget"]                 =  static_cast<bool>  (1);
// 	config["preserve_raster_on_save"]      =  static_cast<bool>  (0);
// 	config["connectivity"]                 =  static_cast<int>   (10);

// 	config["drop_async_frames"]            =  static_cast<bool>  (1);
}

Config* Config::instance() {
	if (_instance == nullptr) {
		_instance = new Config();
	}
	return _instance;
}

