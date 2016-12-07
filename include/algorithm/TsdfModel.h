#ifndef TSDF_MODEL_H
#define TSDF_MODEL_H

#include "SdfModel.h"
#include "Plane3D.h"

class TsdfModel : public SdfModel
{
public:
	//Rotate or not
	bool doRotate;

	//boundBox
	double maxX;
	double maxY;
	double maxZ;
	double minX;
	double minY;
	double minZ;
	int Xlen;
	int Ylen;
	int Zlen;

	//parameters
	double devide;
    double thre;
    double epsi;
	Eigen::Matrix4f local_tran;

	//fit plane
	Plane3D current_coff;
		

public:
    TsdfModel(Eigen::Matrix4f ltran, 
			    double maxx,double maxy, double maxz,double minx,double miny,double minz,
				double dev,double th,double ep,
				bool dorotate);
	~TsdfModel();

    void changeLocal(Eigen::Matrix4f ltran, 
			            double maxx,double maxy, double maxz,double minx,double miny,double minz);

	virtual void dataFusion(PointCloud::Ptr cloud, PointNormal::Ptr normals, 
			        Eigen::Matrix4f tran, int nHalfXres, int nHalfYres, double fCoeffX, double fCoeffY);

	void Traversal(PointCloud::Ptr cloud, PointNormal::Ptr normals, 
			        Eigen::Matrix4f tran, int nHalfXres, int nHalfYres, double fCoeffX, double fCoeffY);

	virtual void rayCast(double ix, double iy, double ax, double ay, double devide,
			                Eigen::Matrix4f tran, 
			                float* p_dists, float* p_nx, float* p_ny, float* p_nz);

	virtual void freeData();

	virtual void getSample(float sdevide, float sminX, float sminY, float sminZ, int sXlen, int sYlen, int sZlen, float* absrate);

private:

	//volume data
	//float* dist;
	//float* n_x;
	//float* n_y;
	//float* n_z;
	//float* weight;

private:

		
};

#endif // TSDF_H
