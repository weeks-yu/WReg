#ifndef COMMONFUNCTION_H_
#define COMMONFUNCTION_H_

#include "Cuda/internal.h"
#include "OdometryProvider.h"

#include <opencv2/opencv.hpp>
#include <vector>
// #include <vector_types.h>
// #include <boost/thread/mutex.hpp>
// #include <boost/thread/thread.hpp>
// #include <boost/lexical_cast.hpp>
// #include <boost/date_time/posix_time/posix_time.hpp>
// #include <boost/date_time/posix_time/posix_time_types.hpp>
// #include <boost/thread/condition_variable.hpp>

class PointCloudCuda
{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
		PointCloudCuda(int width,
                    int height,
                    float cx, float cy, float fx, float fy,
					float depthFactor);

		virtual ~PointCloudCuda();

        void init(unsigned short * depth);


		void getVMap(cv::Mat &mat, const float depthCutoff);
		void getNMap(cv::Mat &mat);

    private:
		// gpu
        DeviceArray2D<unsigned short> depth_tmp;
		DeviceArray2D<float> vmaps_;
		DeviceArray2D<float> nmaps_;

        Intr intr;
        const int width;
        const int height;
		const float depthFactor_;
};

#endif /* COMMONFUNCTION_H_ */
