#include <vector>
#include <pcl/common/common_headers.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>

using namespace std;

bool getPlanesByRANSACCuda(
	vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f>> &result_planes,
	vector<vector<pair<int, int>>> *matches,
	const cv::Mat &rgb, const cv::Mat &depth);