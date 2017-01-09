#include "MeshModel.h"

MeshModel::MeshModel()
{
	this->devide = 0.01;
	this->halfwidth = 320;
	this->halfheight = 240;
	this->fx = 525;
	this->fy = 525;
	tsdf = new TsdfModel(Eigen::Matrix4f::Identity(),1.5, 1, 5.0, -1.5, -1.0, 0.0, devide, devide * 10, devide / 10, true);
	gen = new SNPGenerator(tsdf, devide);
}

MeshModel::MeshModel(const Eigen::Matrix4f &tran,
	double minx, double miny, double minz, double maxx, double maxy, double maxz,
	double devide,
	int width, int height, double fx, double fy)
{
	this->devide = devide;
	this->halfwidth = width / 2;
	this->halfheight = height / 2;
	this->fx = fx;
	this->fy = fy;
	tsdf = new TsdfModel(tran, maxx, maxy, maxz, minx, miny, minz, devide, devide * 10, devide / 10, true);
	gen = new SNPGenerator(tsdf, devide);
}

MeshModel::~MeshModel()
{
	
}

void MeshModel::dataFusion(PointCloudWithNormalPtr cloud, const Eigen::Matrix4f &tran)
{
	tsdf->dataFusion(cloud, tran, halfwidth, halfheight, fx, fy);
	gen->castNextMeshes(tran);
}

GLMesh* MeshModel::getModel()
{
	GLMesh *ret = new GLMesh(*(gen->mesh));
	return ret;
}