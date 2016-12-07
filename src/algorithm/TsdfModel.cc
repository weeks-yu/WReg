#include "TsdfModel.h"
#include "tsdf.cuh"
#include "Util.h"
#include <time.h>

using namespace std;

Eigen::Matrix4f CalTransport(double vectorBefore[4],double vectorAfter[4])
{
	double vb[3];
	double va[3];
	for(int i = 0;i < 3;i++){
		vb[i] = vectorBefore[i];
		va[i] = vectorAfter[i];
	}

	double rotateMatrix[4][4];
	Calculation4d(vb, va, rotateMatrix);

	Eigen::Matrix4f p_transformation_matrix;
	for(int ri = 0; ri < 4; ri++)
		for(int rj = 0; rj < 4; rj++)
		    p_transformation_matrix(ri, rj) = rotateMatrix[ri][rj];
		
	return p_transformation_matrix;

}

TsdfModel::TsdfModel(Eigen::Matrix4f ltran, 
		            double maxx,double maxy, double maxz,double minx,double miny,double minz,
					double dev,double th,double ep,
					bool dorotate)
	                :local_tran(ltran),
					maxX(maxx),maxY(maxy),maxZ(maxz),minX(minx),minY(miny),minZ(minz),
					devide(dev),thre(th),epsi(ep),doRotate(dorotate)
{

	current_coff.inliers = 0;
	Xlen = (int)((maxx-minx)/devide) + 1;
	Ylen = (int)((maxy-miny)/devide) + 1;
	Zlen = (int)((maxz-minz)/devide) + 1;

	int size = Xlen*Ylen*Zlen;
	float* dist = new float[size];
	float* n_x = new float[size];
	float* n_y = new float[size];
	float* n_z = new float[size];
	float* weight = new float[size];
	for(int i = 0;i < size;i++){
		dist[i] = 100;
		n_x[i] = 0;
		n_y[i] = 0;
		n_z[i] = 0;
		weight[i] = 0;
	}

	float* tran = new float[4*4];
	for(int i = 0;i < 4;i++)
	{
		for(int j = 0;j < 4;j++)
		{
			tran[i*4+j] = local_tran(i,j);
		}
	}
	int ret = initialSdf(minX,minY,minZ,Xlen,Ylen,Zlen,devide,epsi,thre,dist,weight,n_x,n_y,n_z,tran);

	delete[] dist;
	delete[] weight;
	delete[] n_x;
	delete[] n_y;
	delete[] n_z;
	delete[] tran;
}

TsdfModel::~TsdfModel()
{
}

void TsdfModel::changeLocal(Eigen::Matrix4f ltran, 
		                    double maxx,double maxy, double maxz,double minx,double miny,double minz)
{
	maxX = maxx;
	maxY = maxy;
	maxZ = maxz;
	minX = minx;
	minY = miny;
	minZ = minz;

	Xlen = (int)((maxx-minx)/devide) + 1;
	Ylen = (int)((maxy-miny)/devide) + 1;
	Zlen = (int)((maxz-minz)/devide) + 1;

	float* tran = new float[4*4];
	for(int i = 0;i < 4;i++)
	{
		for(int j = 0;j < 4;j++)
		{
			tran[i*4+j] = ltran(i,j);
		}
	}

	float* omc = new float[4*4];
	Eigen::Matrix4f olocalc = local_tran.inverse();
	for(int i = 0;i < 4;i++)
	{
		for(int j = 0;j < 4;j++)
		{
			omc[i*4+j] = olocalc(i,j);
		}
	}

	int ret = changeSdf(minX, minY, minZ, Xlen, Ylen, Zlen, tran, omc);
	delete[] omc;
	delete[] tran;
	local_tran = ltran;
}

void TsdfModel::dataFusion(PointCloud::Ptr cloud, PointNormal::Ptr normals, 
			                Eigen::Matrix4f tran, int nHalfXres, int nHalfYres, double fCoeffX, double fCoeffY)
{
	clock_t start, finish, duration;

	start = clock(); 
	if(doRotate)
	{
		Plane3D coff = coff_cloud(normals,cloud);
		double dis = (coff.a*current_coff.a)+(coff.b*current_coff.b)+(coff.c*current_coff.c);
		if(current_coff.inliers == 0 || dis < 0.7)
		{
			current_coff = coff;
		}

		double before[4];
		double after[4];
		before[0] = coff.a;
		before[1] = coff.b;
		before[2] = coff.c;
		before[3] = coff.d;

		after[0] = 0;
		after[1] = 0;
		after[2] = 1;
		after[3] = 0;

		Eigen::Matrix4f zero;  
		zero << 1, 0, 0, 0,  
				0, 1, 0, 0,  
				0, 0, 1, 0,  
				0, 0, 0, 1;

		//Eigen::Matrix4f local = CalTransport(before,after);
		Eigen::Matrix4f local = zero;
		Eigen::Matrix4f local_c = local.inverse();

		bool flag = false;
		double max_X,max_Y,max_Z,min_X,min_Y,min_Z;
		for(int i = 0;i < cloud->size();i++)
		{
			pcl::PointXYZRGB pp = cloud->at(i);
			Eigen::Vector4f v;
			v[0] = pp.x ;
			v[1] = pp.y;
			v[2] = pp.z;
			v[3] = 1;
			Eigen::Vector4f t = local_c * v;
			if(!flag)
			{
				max_X = t[0];
				max_Y = t[1];
				max_Z = t[2];
				min_X = t[0];
				min_Y = t[1];
				min_Z = t[2];
				flag = true;
			}else{
				if(t[0] > max_X) max_X = t[0];
				if(t[1] > max_Y) max_Y = t[1];
				if(t[2] > max_Z) max_Z = t[2];
				if(t[0] < min_X) min_X = t[0];
				if(t[1] < min_Y) min_Y = t[1];
				if(t[2] < min_Z) min_Z = t[2];
			}
		}
		
		
		double bound = 5 * devide;
		//cout<<max_X-min_X<<","<<max_Y-min_Y<<","<<max_Z-min_Z<<endl;
		double dx = (max_X - min_X);
		double dy = (max_Y - min_Y);
		double dz = (max_Z - min_Z);
		cout<<"size:"<<dx<<","<<dy<<","<<dz<<endl;

		changeLocal(local, max_X, max_Y, max_Z, min_X, min_Y, min_Z);
	}
	finish = clock(); 
	duration = (double)(finish - start) * 1000 / CLOCKS_PER_SEC ; 
	std::cout<<"change:"<<duration<<std::endl;

	start = clock();
	Traversal(cloud, normals, tran, nHalfXres, nHalfYres, fCoeffX, fCoeffY);
	finish = clock(); 
	duration = (double)(finish - start) * 1000 / CLOCKS_PER_SEC ; 
	std::cout<<"fusion:"<<duration<<std::endl;
}

void TsdfModel::Traversal(PointCloud::Ptr cloud, PointNormal::Ptr normals, 
			                Eigen::Matrix4f tran, int nHalfXres, int nHalfYres, double fCoeffX, double fCoeffY)
{
	int size = cloud->points.size();
	float* ptsx = new float[size];
	float* ptsy = new float[size];
	float* ptsz = new float[size];
	float* norx = new float[size];
	float* nory = new float[size];
	float* norz = new float[size];
	for(int i = 0;i < size;i++)
	{
		ptsx[i] = cloud->points[i].x;
		ptsy[i] = cloud->points[i].y;
		ptsz[i] = cloud->points[i].z;
		norx[i] = normals->points[i].normal_x;
		nory[i] = normals->points[i].normal_y;
		norz[i] = normals->points[i].normal_z;
	}
	Eigen::Matrix4f mcT = tran.inverse();
	float* mc = new float[4*4];
	for(int i = 0;i < 4;i++)
	{
		for(int j = 0;j < 4;j++)
		{
			mc[i*4+j] = mcT(i,j);
		}
	}
	int width = cloud->width;

	int ret = updateSdf(ptsx, ptsy, ptsz, norx, nory, norz,size, mc,width,nHalfXres,nHalfYres,fCoeffX,fCoeffY);
	delete[] ptsx;
	delete[] ptsy;
	delete[] ptsz;
	delete[] norx;
	delete[] nory;
	delete[] norz;
	delete[] mc;
}

	
void TsdfModel::rayCast(double ix, double iy, double ax, double ay, double devide,
			                Eigen::Matrix4f tran, 
			                float* p_dists, float* p_nx, float* p_ny, float* p_nz)
{
	Eigen::Matrix4f localc = local_tran.inverse();
	Eigen::Matrix4f t_in_local = localc*tran;;

	int xl = (ax-ix) / devide;
	int yl = (ay-iy) / devide;

	float* pmc = new float[4*4];
	for(int i = 0;i < 4;i++)
	{
		for(int j = 0;j < 4;j++)
		{
			pmc[i*4+j] = t_in_local(i,j);
		}
	}
	getAllDepths(ix, iy, devide, xl, yl, pmc, p_dists, p_nx,  p_ny,  p_nz);
	delete[] pmc;
}

void TsdfModel::freeData()
{
	int ret = freeSdf();
}

void TsdfModel::getSample(float sdevide, float sminX, float sminY, float sminZ, int sXlen, int sYlen, int sZlen, float* absrate)
{
	sampleSdfData(sdevide, sminX, sminY, sminZ, sXlen, sYlen, sZlen, absrate);
}