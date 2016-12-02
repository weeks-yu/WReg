#ifndef MESH_GENERATOR_H
#define MESH_GENERATOR_H


#include "SdfModel.h"
#include <set>

namespace Rgbd
{
	class GLMesh;
	class GLTriangle;
	class GLVertex;

	struct ProjectPoint{
		int vertexID;
		GLVertex* v;
		double x; //Ͷx coord in project plane
		double y; //Ͷy coord in project plane
		double theta;  // polar angle in project plane
		double radius;  // polar radius in project plane
		bool deleted; // delete flag
	};

	//additional attribute for vertex in mesh
	struct VertexAttribute
	{
		double dist; //dist in ray-casting
		int tran;
		double upxyz[3];
		double upcount;
		double degree;

		inline VertexAttribute():degree(0),upcount(0)
		{
			upxyz[0] = 0;
			upxyz[1] = 0;
			upxyz[2] = 0;
		}
	};

	class SNPGenerator
	{
	public:
		//triangular parameters
		int mu;
		double surfaceAng;  
		double minAng;
		double maxAng;
		double devide;

		//Sdfmoel
		SdfModel* sdf;

		
		int buildId; //build step
		int **oldp; //store last frame plane

		//generate Meshes;
        GLMesh* mesh;
		//additional attribute
		std::vector<VertexAttribute> vertAttrs;
		std::vector<Eigen::Matrix4f> raycastTrans;
	public:

		SNPGenerator(SdfModel* model, double dv);
		~SNPGenerator();

		void castNextMeshes(Eigen::Matrix4f tran);

	private:

		//vertex edit
		void updateVertexes();

		void DoUpdate(int vertexId);

		void LaplacianUpdate(int vertexId, int nearId);

		int putPoint(double x,double y, double d,
					  double nx, double ny, double nz,
					   Eigen::Matrix4f tran);

		void addDegree(int vertexId, int near1, int near2);


		//mesh edit
		bool validTriangle(GLTriangle& m,Eigen::Matrix4f tran);

		void TryaddTriangle(int v1,int v2,int v3,Eigen::Matrix4f tran);

		void TryAddRightTriangle(int a,int b,int c,Eigen::Matrix4f tran);

		void buildTic(int px,int py,double ix,double iy,double xl,double yl,
			          int** pre,int** planePoints,
					  Eigen::Matrix4f mc, Eigen::Matrix4f tran,
					  int forbid,int* except,int m);

		void findNearest(int** planePoints,
		                 int xid,int yid,int xl,int yl,
						 int Rid,std::set<int>& nearest,int m,int forbid);

		bool buildMeshesAtR(int** points,int i,int j,Eigen::Matrix4f tran,int forbid);

		void buildMeshesGreedy(std::set<int> nearest,int Rid,Eigen::Matrix4f tran,int forbid, int* except);

		int** castPieceMeshes(int tranID,int** pre,int lastID);

		void coverPointinTriangle(int v[3],int** pts, int xl,int yl,Eigen::Matrix4f mc,double minx,double miny);

		void coverPointinMeshes(int** pre,int** pts,int i, int j, int xl,int yl,Eigen::Matrix4f mc,double minx,double miny);

		inline bool inTriangle(Eigen::Vector4f t[3],double x,double y)
		{
			double s1 = (t[0][0] - x)*(t[0][1]-t[1][1])-(t[0][0]-t[1][0])*(t[0][1] - y);
			double s2 = (t[1][0] - x)*(t[1][1]-t[2][1])-(t[1][0]-t[2][0])*(t[1][1] - y);
			double s3 = (t[2][0] - x)*(t[2][1]-t[0][1])-(t[2][0]-t[0][0])*(t[2][1] - y);
			if((s1*s2 > 0)&&(s2*s3 > 0)) return true;
			if(s1 == 0 && x >= std::min(t[0][0],t[1][0]) && x <= std::max(t[0][0],t[1][0])) return true;
			if(s2 == 0 && x >= std::min(t[1][0],t[2][0]) && x <= std::max(t[1][0],t[2][0])) return true;
			if(s3 == 0 && x >= std::min(t[2][0],t[0][0]) && x <= std::max(t[2][0],t[0][0])) return true;
			return false;
		}

	};
}

#endif // MESH_GENERATOR_H
