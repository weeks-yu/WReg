#pragma once

#include "PointCloud.h"
#include "TriangleMesh.h"
#include "TsdfModel.h"
#include "SNPGenerator.h"

class MeshModel
{
public:
	MeshModel();
	MeshModel(const Eigen::Matrix4f &tran,
		double minx, double miny, double minz, double maxx, double maxy, double maxz,
		double devide,
		int width, int height, double fx, double fy);
	~MeshModel();

	void dataFusion(PointCloudWithNormalPtr cloud, const Eigen::Matrix4f &tran);
	GLMesh* getModel();

private:
	TsdfModel *tsdf;
	SNPGenerator *gen;

	double devide;
	int halfwidth;
	int halfheight;
	double fx;
	double fy;
};