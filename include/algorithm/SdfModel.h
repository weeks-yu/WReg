#ifndef SDF_MODEL_H
#define SDF_MODEL_H

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include "PointCloud.h"

class SdfModel
{
public:
	SdfModel()
	{
	}

	virtual ~SdfModel()
	{
	};

	virtual void dataFusion(PointCloudPtr cloud, PointCloudNormalPtr normals,
		Eigen::Matrix4f tran, int nHalfXres, int nHalfYres, double fCoeffX, double fCoeffY) = 0;

	virtual void rayCast(double ix, double iy, double ax, double ay, double devide,
		Eigen::Matrix4f tran,
		float* p_dists, float* p_nx, float* p_ny, float* p_nz) = 0;

	virtual void getSample(float sdevide, float sminX, float sminY, float sminZ, int sXlen, int sYlen, int sZlen, float* absrate) = 0;

	virtual void freeData() = 0;

};

#endif // SDF_MODEL_H
