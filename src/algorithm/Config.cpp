#include "Config.h"

using namespace std;

Config* Config::_instance = nullptr;

Config::Config() {
	config["feature_type"]					= std::string("SURF");

	// RANSAC
	config["candidate_number"]		= static_cast<int>    (10);
	config["min_matches"]			= static_cast<int>    (100);
	config["min_inliers_percent"]	= static_cast<float>  (0.2);
	config["max_dist_for_inliers"]	= static_cast<double> (0.03);
	config["matches_criterion"]		= static_cast<float>  (0.75);

	// Keyframe
	config["min_translation_meter"] = static_cast<double> (0.1);
	config["min_rotation_degree"]	= static_cast<double> (10.0);
	config["keyframe_check_N"]		= static_cast<int>    (4);
	config["keyframe_check_M"]		= static_cast<int>    (4);
	config["keyframe_check_F"]		= static_cast<int>    (1);
	config["keyframe_check_P"]		= static_cast<double> (0.8);

	// KDTree
	config["kdtree_trees"]			= static_cast<int>    (4);
	config["kdtree_max_leaf"]		= static_cast<int>    (64);
	config["kdtree_max_leaf_mult"]	= static_cast<int>    (128);
	config["kdtree_k_mult"]			= static_cast<int>    (30);

	// Quad Tree
	config["quadtree_size"]			= static_cast<float>  (20.0);
	config["active_window_size"]	= static_cast<float>  (5.0);

	// HOG-man
	config["hogman_iterations"]		= static_cast<int>    (10);

	// Camera parameters
	config["image_width"]			= static_cast<int>    (640);
	config["image_height"]			= static_cast<int>    (480);
	config["camera_fx"]				= static_cast<double> (517.3);
	config["camera_fy"]				= static_cast<double> (516.5);
	config["camera_cx"]				= static_cast<double> (318.6);
	config["camera_cy"]				= static_cast<double> (255.3);
	config["depth_factor"]			= static_cast<double> (5000.0);

	// hogman parameters
	config["graph_levels"]			= static_cast<int>    (3);
	config["node_distance"]			= static_cast<int>    (2);

	// cuda parameters
	config["icpcuda_threads"]		= static_cast<int>    (240);
	config["icpcuda_blocks"]		= static_cast<int>    (80);

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

