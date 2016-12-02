#include "SNPGenerator.h"
#include "TriangleMesh.h"
#include "Util.h"
#include <time.h>

namespace Rgbd
{
	bool comparePP(ProjectPoint& a, ProjectPoint& b)
	{
		return(abs(a.theta - b.theta) < 0.00000001)?a.radius < b.radius:a.theta<b.theta;
	}

	//judge point on which side of a segment
	double direction(ProjectPoint p1,ProjectPoint p2,ProjectPoint p3)
	{
		return (p2.x-p1.x)*(p3.y-p1.y)-(p3.x-p1.x)*(p2.y-p1.y);
	}

	//judge whether a point in on the segment
	bool onSegment(ProjectPoint p1,ProjectPoint p2,ProjectPoint p3)
	{
		double left,right;

		if(p1.x<=p2.x)
		{
			left=p1.x;
			right=p2.x;
		}
		else
		{
			left=p2.x;
			right=p1.x;
		}

		if(left<=p3.x&&p3.x<=right)
		{
			return true;
		}
		else
		{
			return false;
		}
	}

	bool segmentsIntersect(ProjectPoint p1,ProjectPoint p2,ProjectPoint p3,ProjectPoint p4)
	{
		double d1,d2,d3,d4; 

		d1=direction(p3,p4,p1);
		d2=direction(p3,p4,p2);
		d3=direction(p1,p2,p3);
		d4=direction(p1,p2,p4);

		if(d1*d2<0&&d3*d4<0)
		{
			return true;
		}

		else if(0==d1&&onSegment(p3,p4,p1)==true)
		{
			return true;
		}
		else if(0==d2&&onSegment(p3,p4,p2)==true)
		{
			return true;
		}
		else if(0==d3&&onSegment(p1,p2,p3)==true)
		{
			return true;
		}
		else if(0==d4&&onSegment(p1,p2,p4)==true)
		{
			return true;
		}
		else
		{
			return false;
		}
	}


	SNPGenerator::SNPGenerator(SdfModel* model, double dv):mu(2),surfaceAng(M_PI/4),minAng(M_PI/18),maxAng(2*M_PI/3),devide(dv),
		                                                  sdf(model),mesh(new GLMesh()),oldp(NULL),buildId(0)
	{
	}

	SNPGenerator::~SNPGenerator()
	{
		//scan range
		double ix = -1.5;
		double iy = -1;
		double ax = 1.2;
		double ay = 1;

		int xl = (ax-ix) / devide;
		int yl = (ay-iy) / devide;

		if(oldp != NULL)
		{
			for(int i = 0;i <xl;i++) delete[] oldp[i];
			delete[] oldp;
		}
		delete mesh;
	}

	void SNPGenerator::updateVertexes()
	{
		int size = mesh->gl_vertexes.size();
		for(int i = 0; i < size; i++)
		{
			DoUpdate(i);
		}
	}

	void SNPGenerator::DoUpdate(int vertexId)
	{
		GLVertex &v = mesh->gl_vertexes[vertexId];
		VertexAttribute &va = vertAttrs[vertexId];

		if(va.upcount >0){
			Eigen::Matrix4f tr = raycastTrans[va.tran];
			va.upxyz[0] /= va.upcount;
			va.upxyz[1] /= va.upcount;
			va.upxyz[2] /= va.upcount;
			double ndis = NORM2(tr(0,2),tr(1,2),tr(2,2));
			double mcos = (tr(0,2)*va.upxyz[0]+tr(1,2)*va.upxyz[1]+tr(2,2)*va.upxyz[2])/ndis/ndis;
			v.v[0] += mcos*tr(0,2);
			v.v[1] += mcos*tr(1,2);
			v.v[2] += mcos*tr(2,2);
			va.upxyz[0] = 0;
			va.upxyz[1] = 0;
			va.upxyz[2] = 0;
			va.upcount = 0;
		}
	}

	void SNPGenerator::LaplacianUpdate(int vertexId, int nearId)
	{
		GLVertex &v = mesh->gl_vertexes[vertexId];
		VertexAttribute &va = vertAttrs[vertexId];

		GLVertex &vN = mesh->gl_vertexes[nearId];

		//double dis = sqrt((vN.v[0]-v.v[0])*(vN.v[0]-v.v[0]) + (vN.v[1]-v.v[1])*(vN.v[1]-v.v[1]) + (vN.v[2]-v.v[2])*(vN.v[2]-v.v[2]));
		double dis = NORM2(vN.v[0]-v.v[0],vN.v[1]-v.v[1], vN.v[2]-v.v[2]);
		double weight = 0;
		if(dis < 0.1){
		 weight = pow(2.7182818284590452353602874713526624977572,-500*dis*dis);
		}
		va.upxyz[0] += weight*(vN.v[0]-v.v[0]);
		va.upxyz[1] += weight*(vN.v[1]-v.v[1]);
		va.upxyz[2] += weight*(vN.v[2]-v.v[2]);
		va.upcount += weight;
	}

	int SNPGenerator::putPoint(double x,double y, double d,
					double nx, double ny, double nz,
					Eigen::Matrix4f tran)
	{
		GLVertex v;
		VertexAttribute va;

		Eigen::Matrix4f mc =  tran;
		v.v[0] = mc(0,0)*x + mc(0,1)*y + mc(0,2)*d + mc(0,3);
		v.v[1] = mc(1,0)*x + mc(1,1)*y + mc(1,2)*d + mc(1,3);
		v.v[2] = mc(2,0)*x + mc(2,1)*y + mc(2,2)*d + mc(2,3);
		v.v[3] = nx;
		v.v[4] = ny;
		v.v[5] = nz;
		va.dist = d;

		mesh->gl_vertexes.push_back(v);
		vertAttrs.push_back(va);

		return mesh->gl_vertexes.size()-1;
	}

	void SNPGenerator::addDegree(int vertexId, int near1, int near2)
	{
		GLVertex &v = mesh->gl_vertexes[vertexId];
		VertexAttribute &va = vertAttrs[vertexId];

		double* v1 = mesh->gl_vertexes[near1].v;
		double* v2 = mesh->gl_vertexes[near2].v;

		double cosLen1 = ((v1[0]-v.v[0])*v.v[3]+(v1[1]-v.v[1])*v.v[4]+(v1[2]-v.v[2])*v.v[5]);
		double t1[3];
		t1[0] = (v1[0]-v.v[0]) - cosLen1*v.v[3];
		t1[1] = (v1[1]-v.v[1]) - cosLen1*v.v[4];
		t1[2] = (v1[2]-v.v[2]) - cosLen1*v.v[5];

		double cosLen2 = ((v2[0]-v.v[0])*v.v[3]+(v2[1]-v.v[1])*v.v[4]+(v2[2]-v.v[2])*v.v[5]);
		double t2[3];
		t2[0] = (v2[0]-v.v[0]) - cosLen2*v.v[3];
		t2[1] = (v2[1]-v.v[1]) - cosLen2*v.v[4];
		t2[2] = (v2[2]-v.v[2]) - cosLen2*v.v[5];

		//double theta = (t1[0]*t2[0] + t1[1]*t2[1] + t1[2]*t2[2])/sqrt(t1[0]*t1[0] + t1[1]*t1[1] + t1[2]*t1[2])/sqrt(t2[0]*t2[0] + t2[1]*t2[1] + t2[2]*t2[2]);
		double theta = (t1[0]*t2[0] + t1[1]*t2[1] + t1[2]*t2[2])/NORM2(t1[0], t1[1], t1[2])/NORM2(t2[0], t2[1], t2[2]);
		double ac = acos(theta);
		va.degree += ac;
	}

	int** SNPGenerator::castPieceMeshes(int tranID,int** pre,int lastID)
	{
		Eigen::Matrix4f& tran = raycastTrans[tranID];
		Eigen::Matrix4f& last = raycastTrans[lastID];
		Eigen::Matrix4f mc =  tran.inverse();
		Eigen::Matrix4f lc =  last.inverse();

		//scan range
		double ix = -1.5-devide*0.5;
		double iy = -1-devide*0.5;
		double ax = 1.2;
		double ay = 1;

		int xl = (ax-ix) / devide;
		int yl = (ay-iy) / devide;

		int ** planePoints = NULL;
		if(xl > 0 ) planePoints = new int*[xl];
		for(int i = 0;i < xl;i++)
		{
			planePoints[i] = new int[yl];
		}

		

		//cover the old meshes on new plane
		//in case of overlap
		
		/*
		if(mesh->gl_meshes.size() > 0)
		{
			std::map<std::tuple<int,int,int>,GLTriangle>::iterator it = mesh->gl_meshes.begin();
			for(;it != mesh->gl_meshes.end();it++)
			{
				int v[3];
				v[0] = it->second.vertexes[0];
				v[1] = it->second.vertexes[1];
				v[2] = it->second.vertexes[2];
				coverPointinTriangle(v,planePoints,xl,yl,mc,ix,iy);
			}
		}
		*/
		
		
		std::set<int> bounds;
		int forbid = mesh->gl_vertexes.size();
		float* p_dists = new float[xl*yl];
		float* p_nx = new float[xl*yl];
		float* p_ny = new float[xl*yl];
		float* p_nz = new float[xl*yl];
		clock_t start, finish, duration;

	    start = clock(); 
		sdf->rayCast(ix, iy, ax, ay, devide, tran, p_dists, p_nx, p_ny, p_nz);
		finish = clock(); 
		duration = (double)(finish - start) * 1000 / CLOCKS_PER_SEC ; 
		std::cout<<"raycast:"<<duration<<std::endl;
		start = clock();

		for(int i = 0;i < xl;i++)
		{
			for(int j = 0;j < yl;j++)
			{
				
				if(planePoints[i][j] < 0)
				{

					int pid = (j*xl)+i;
					if(p_dists[pid] > -10000 && !ISNAN(p_nx[pid])){
						int vid = putPoint(ix+i*devide,iy+j*devide,p_dists[pid], p_nx[pid], p_ny[pid], p_nz[pid], tran);
						GLVertex &v1 = mesh->gl_vertexes[vid];
						VertexAttribute &va1 = vertAttrs[vid];
						va1.tran = tranID;
						planePoints[i][j] =  vid;
					}else{
						planePoints[i][j] = -1;
					}
				}
				
			    
				if(i > 0 && j > 0 && planePoints[i][j] >= 0)
				{
					bool tic = buildMeshesAtR(planePoints,i,j,tran,forbid);
				
					if(tic)
					{
						if(planePoints[i][j] >=  forbid) bounds.insert(planePoints[i][j]);
						if(planePoints[i-1][j] >=  forbid) bounds.insert(planePoints[i-1][j]);
						if(planePoints[i][j-1] >=  forbid) bounds.insert(planePoints[i][j-1]);
						if(planePoints[i-1][j-1] >= forbid) bounds.insert(planePoints[i-1][j-1]);
					}
				
				} 
				
			}
		}
		delete[] p_dists;
		delete[] p_nx;
		delete[] p_ny;
		delete[] p_nz;
		
		
		updateVertexes();

		finish = clock(); 
		duration = (double)(finish - start) * 1000 / CLOCKS_PER_SEC ; 
		std::cout<<"build:"<<duration<<std::endl;
		start = clock();
	    
		int m = 1;
		if(pre)
		{
			for(int i = 1;i < xl;i++)
			{
				if(pre[i][0] >= 0)
				{
					int except[2];
					except[0]= -1;
					except[1]= -1;
					if(i+1 < xl && pre[i+1][0] >= 0) except[0] = pre[i+1][0];
					if(i-1 >= 0 && pre[i-1][0] >= 0) except[1] = pre[i-1][0];
					buildTic(i,0,ix,iy,xl,yl,pre,planePoints,mc,tran, forbid,except,m);
				}
			}
			for(int i = 1;i < yl;i++)
			{
				if(pre[xl-1][i] >= 0)
				{
					int except[2];
					except[0]= -1;
					except[1]= -1;
					if(i+1 < yl && pre[xl-1][i+1]  >= 0) except[0] = pre[xl-1][i+1];
					if(i-1 >= 0 && pre[xl-1][i-1]  >= 0) except[1] = pre[xl-1][i-1];
					buildTic(xl-1,i,ix,iy,xl,yl,pre,planePoints,mc,tran, forbid,except,m);
				}
			}
			for(int i = xl-2;i >= 0;i--)
			{
				if(pre[i][yl-1] >= 0)
				{
					int except[2];
					except[0]= -1;
					except[1]= -1;
					if(i-1 >= 0 && pre[i-1][yl-1]  >= 0) except[0] = pre[i-1][yl-1];
					if(i+1 < xl && pre[i+1][yl-1]  >= 0) except[1] = pre[i+1][yl-1];
					buildTic(i,yl-1,ix,iy,xl,yl,pre,planePoints,mc,tran, forbid,except,m);
				}
			}
			for(int i = yl-2;i >= 0;i--)
			{
				if(pre[0][i] >= 0)
				{
					int except[2];
					except[0]= -1;
					except[1]= -1;
					if(i-1 >= 0 && pre[0][i-1]  >= 0) except[0] = pre[0][i-1];
					if(i+1 < yl && pre[0][i+1]  >= 0) except[1] = pre[0][i+1];
					buildTic(0,i,ix,iy,xl,yl,pre,planePoints,mc,tran, forbid,except,m);
				}
			}
		}
		m = 2;
		if(pre)
		{
			for(int i = 1;i < xl;i++)
			{
				if(pre[i][0] >= 0)
				{
					int except[2];
					except[0]= -1;
					except[1]= -1;
					if(i+1 < xl && pre[i+1][0] >= 0) except[0] = pre[i+1][0];
					if(i-1 >= 0 && pre[i-1][0] >= 0) except[1] = pre[i-1][0];
					buildTic(i,0,ix,iy,xl,yl,pre,planePoints,mc,tran, forbid,except,m);
				}
			}
			for(int i = 1;i < yl;i++)
			{
				if(pre[xl-1][i] >= 0)
				{
					int except[2];
					except[0]= -1;
					except[1]= -1;
					if(i+1 < yl && pre[xl-1][i+1]  >= 0) except[0] = pre[xl-1][i+1];
					if(i-1 >= 0 && pre[xl-1][i-1]  >= 0) except[1] = pre[xl-1][i-1];
					buildTic(xl-1,i,ix,iy,xl,yl,pre,planePoints,mc,tran, forbid,except,m);
				}
			}
			for(int i = xl-2;i >= 0;i--)
			{
				if(pre[i][yl-1] >= 0)
				{
					int except[2];
					except[0]= -1;
					except[1]= -1;
					if(i-1 >= 0 && pre[i-1][yl-1]  >= 0) except[0] = pre[i-1][yl-1];
					if(i+1 < xl && pre[i+1][yl-1]  >= 0) except[1] = pre[i+1][yl-1];
					buildTic(i,yl-1,ix,iy,xl,yl,pre,planePoints,mc,tran, forbid,except,m);
				}
			}
			for(int i = yl-2;i >= 0;i--)
			{
				if(pre[0][i] >= 0)
				{
					int except[2];
					except[0]= -1;
					except[1]= -1;
					if(i-1 >= 0 && pre[0][i-1]  >= 0) except[0] = pre[0][i-1];
					if(i+1 < yl && pre[0][i+1]  >= 0) except[1] = pre[0][i+1];
					buildTic(0,i,ix,iy,xl,yl,pre,planePoints,mc,tran, forbid,except,m);
				}
			}
		}
	
	
		for(std::set<int>::iterator it = bounds.begin();it != bounds.end();it++)
		{
			int b = *it;
			std::set<int> nearest;
			GLVertex& R = mesh->gl_vertexes[b];
			Eigen::Vector4f v;
			v[0] = R.v[0];
			v[1] = R.v[1];
			v[2] = R.v[2];
			v[3] = 1;
			Eigen::Vector4f t1 = mc*v; 
			int xid1 = static_cast<int>((t1[0]-ix)/devide+0.5);
			int yid1 = static_cast<int>((t1[1]-iy)/devide+0.5);
			findNearest(planePoints,xid1,yid1,xl,yl,b,nearest,2,forbid);
			Eigen::Vector4f t2 = lc*v; 
			int xid2 = static_cast<int>((t2[0]-ix)/devide+0.5);
			int yid2 = static_cast<int>((t2[1]-iy)/devide+0.5);
			findNearest(pre,xid2,yid2,xl,yl,b,nearest,2,0);
			nearest.erase(b);
			int except[2];
			except[0]= -1;
			except[1]= -1;
			buildMeshesGreedy(nearest,b,tran,0,except);
		}
	    
		finish = clock(); 
		duration = (double)(finish - start) * 1000 / CLOCKS_PER_SEC ; 
		std::cout<<"sew:"<<duration<<std::endl;
		start = clock();

		return planePoints;
	}

	void SNPGenerator::castNextMeshes(Eigen::Matrix4f tran)
	{
		raycastTrans.push_back(tran);

		//scan range
		double ix = -1.5;
		double iy = -1;
		double ax = 1.2;
		double ay = 1;

		int xl = (ax-ix) / devide;
		int yl = (ay-iy) / devide;

		int** newp = NULL;
		if(buildId > 0)
			newp = castPieceMeshes(buildId,oldp,buildId-1);
		else
			newp = castPieceMeshes(buildId,oldp,buildId);
		if(oldp != NULL)
		{
			for(int i = 0;i <xl;i++) delete[] oldp[i];
			delete[] oldp;
		}
		oldp = newp;

		buildId++;
	}

	void SNPGenerator::buildTic(int px,int py,double ix,double iy,double xl,double yl,
		                     int** pre,int** planePoints,
							 Eigen::Matrix4f mc, Eigen::Matrix4f tran,
							 int forbid,int* except,int m)
	{
		std::set<int> nearest;
		GLVertex& R = mesh->gl_vertexes[pre[px][py]];
		Eigen::Vector4f v;
		Eigen::Vector4f t;
		v[0] = R.v[0];
		v[1] = R.v[1];
		v[2] = R.v[2];
		v[3] = 1;
		t = mc*v; //tranform to current coord
		int xid = static_cast<int>((t[0]-ix)/devide+0.5);
		int yid = static_cast<int>((t[1]-iy)/devide+0.5);
		//findNearest(planePoints,xid,yid,xl,yl,pre[px][py],nearest,m,forbid);
		if(nearest.size() <= 0) return;
		//findNearest(pre,px,py,xl,yl,pre[px][py],nearest,m,0);
		nearest.erase(pre[px][py]);
		buildMeshesGreedy(nearest,pre[px][py],tran,forbid,except);
	}

	void SNPGenerator::findNearest(int** planePoints,
		                           int xid,int yid,int xl,int yl,
								   int Rid,std::set<int>& nearest,int m,int forbid)
	{
		GLVertex R =mesh->gl_vertexes[Rid];
		for(int jx = xid-m;jx <= xid+m;jx++)
		{
			if(jx < 0) continue;
			if(jx >= xl) continue;
			for(int jy = yid-m;jy <= yid+m;jy++)
			{
				if(jy < 0) continue;
				if(jy >= yl) continue;
				if(planePoints[jx][jy] >= forbid)
				{
					GLVertex nv = mesh->gl_vertexes[planePoints[jx][jy]];
					//double dis = sqrt((nv.v[0]-R.v[0])*(nv.v[0]-R.v[0]) + (nv.v[1]-R.v[1])*(nv.v[1]-R.v[1]) + (nv.v[2]-R.v[2])*(nv.v[2]-R.v[2]));
					double dis = NORM2(nv.v[0]-R.v[0],nv.v[1]-R.v[1],nv.v[2]-R.v[2]);
					if(dis < 6*m*devide) nearest.insert(planePoints[jx][jy]);
				}
			}
		}
	}

	bool SNPGenerator::validTriangle(GLTriangle& m,Eigen::Matrix4f tran)
	{
		GLVertex v[3];
		v[0] = mesh->gl_vertexes[m.vertexes[0]];
		v[1] = mesh->gl_vertexes[m.vertexes[1]];
		v[2] = mesh->gl_vertexes[m.vertexes[2]];
		double th = devide*10;
		//double d1 = sqrt((v[1].v[0] - v[0].v[0])*(v[1].v[0] - v[0].v[0])+(v[1].v[1] - v[0].v[1])*(v[1].v[1] - v[0].v[1])+(v[1].v[2] - v[0].v[2])*(v[1].v[2] - v[0].v[2]));
		//double d2 = sqrt((v[2].v[0] - v[0].v[0])*(v[2].v[0] - v[0].v[0])+(v[2].v[1] - v[0].v[1])*(v[2].v[1] - v[0].v[1])+(v[2].v[2] - v[0].v[2])*(v[2].v[2] - v[0].v[2]));
		//double d3 = sqrt((v[1].v[0] - v[2].v[0])*(v[1].v[0] - v[2].v[0])+(v[1].v[1] - v[2].v[1])*(v[1].v[1] - v[2].v[1])+(v[1].v[2] - v[2].v[2])*(v[1].v[2] - v[2].v[2]));
		double d1 = NORM2(v[1].v[0] - v[0].v[0], v[1].v[1] - v[0].v[1], v[1].v[2] - v[0].v[2]);
		double d2 = NORM2(v[2].v[0] - v[0].v[0], v[2].v[1] - v[0].v[1], v[2].v[2] - v[0].v[2]);
		double d3 = NORM2(v[1].v[0] - v[2].v[0], v[1].v[1] - v[2].v[1], v[1].v[2] - v[2].v[2]);
	
		if(d1 > th) return false;
		if(d2 > th) return false;
		if(d3 > th) return false;

		//double angle1 =  ((v[1].v[0] - v[0].v[0])*(v[2].v[0] - v[0].v[0])+(v[1].v[1] - v[0].v[1])*(v[2].v[1] - v[0].v[1])+(v[1].v[2] - v[0].v[2])*(v[2].v[2] - v[0].v[2]))/d1/d2;
		//double angle2 =  ((v[0].v[0] - v[1].v[0])*(v[2].v[0] - v[1].v[0])+(v[0].v[1] - v[1].v[1])*(v[2].v[1] - v[1].v[1])+(v[0].v[2] - v[1].v[2])*(v[2].v[2] - v[1].v[2]))/d1/d3;
		//double angle3 =  ((v[1].v[0] - v[2].v[0])*(v[0].v[0] - v[2].v[0])+(v[1].v[1] - v[2].v[1])*(v[0].v[1] - v[2].v[1])+(v[1].v[2] - v[2].v[2])*(v[0].v[2] - v[2].v[2]))/d2/d3;

		//if(angle1 > 0.995) return false;
		//if(angle2 > 0.995) return false;
		//if(angle3 > 0.995) return false;
	
		double a[3];
		double b[3];
		double c[3];
		double d[3];
		a[0] = v[1].v[0] - v[0].v[0];
		a[1] = v[1].v[1] - v[0].v[1];
		a[2] = v[1].v[2] - v[0].v[2];
		b[0] = v[2].v[0] - v[0].v[0];
		b[1] = v[2].v[1] - v[0].v[1];
		b[2] = v[2].v[2] - v[0].v[2];

		c[0] = a[1]*b[2] - a[2]*b[1];
		c[1] = a[2]*b[0] - a[0]*b[2];
		c[2] = a[0]*b[1] - a[1]*b[0];
		double clen = NORM2(c[0],c[1],c[2]);

		d[0] = tran(0,2);
		d[1] = tran(1,2);
		d[2] = tran(2,2);
		double dlen = NORM2(d[0], d[1], d[2]);

		double angle = (c[0]*d[0]+c[1]*d[1]+c[2]*d[2])/clen/dlen;

 		if(abs(angle) < 0.05) return false;
	
		return true;
	}

	void SNPGenerator::TryAddRightTriangle(int a,int b,int c,Eigen::Matrix4f tran)
	{
		if( mesh->findEdgeDegree(a,b) < 2 && mesh->findEdgeDegree(b,c) < 2 && mesh->findEdgeDegree(c,a) < 2)
		{
			TryaddTriangle(a,b,c,tran);
		}
	}

	void SNPGenerator::TryaddTriangle(int v1,int v2,int v3,Eigen::Matrix4f tran)
	{
		GLTriangle m1(v1,v2,v3);	   
		if(validTriangle(m1,tran)){
			int a = v1;
			int b = v2;
			int c = v3;
			if (a>b) std::swap(a, b);
			if (b>c) 
			{ 
				std::swap(b, c);
				if (a > b) std::swap(a, b);
			}
			if(mesh->gl_meshes.find(std::tuple<int,int,int>(a,b,c)) != mesh->gl_meshes.end()) return;
			std::pair<std::tuple<int,int,int>,GLTriangle> ins(std::tuple<int,int,int>(a,b,c),m1);		
			mesh->gl_meshes.insert(ins);
			
			mesh->addEdge(v1,v2);
			mesh->addEdge(v2,v3);
			mesh->addEdge(v3,v1);
			addDegree(v1, v2, v3);
			addDegree(v2, v1, v3);
			addDegree(v3, v1, v2);
			
		}
			
	}

	inline bool SNPGenerator::buildMeshesAtR(int** points,int i,int j,Eigen::Matrix4f tran,int forbid)
	{
		bool canbuild = false;
		//upper Triangle
		if(points[i-1][j-1] >= 0 && points[i-1][j] >= 0)
		{
			if(points[i-1][j-1] >= forbid || points[i-1][j] >= forbid || points[i][j] >= forbid){
		
				//do laplacian filter
				int vv1 = points[i-1][j-1];
				int vv2 = points[i-1][j];
				int vv3 = points[i][j];

				LaplacianUpdate(vv1,vv2);
				LaplacianUpdate(vv1,vv3);
				LaplacianUpdate(vv2,vv1);
				LaplacianUpdate(vv2,vv3);
				LaplacianUpdate(vv3,vv1);
				LaplacianUpdate(vv3,vv2);

				if(points[i-1][j-1] >= forbid && points[i-1][j] >= forbid && points[i][j] >= forbid)
				{
					TryaddTriangle(points[i][j],points[i-1][j],points[i-1][j-1],tran);
				}
				else
				{
					//TryaddMesh(points[i][j],points[i-1][j],points[i-1][j-1],tran);
					//TryAddRightMesh(points[i][j],points[i-1][j],points[i-1][j-1],tran);
					canbuild = true;
				}
			}
		}

		//lower trianlge
		if(points[i-1][j-1] >= 0 && points[i][j-1] >= 0)
		{
			if(points[i-1][j-1] >= forbid || points[i][j-1] >= forbid || points[i][j] >= forbid){

				//do laplacian filter
				int vv1 = points[i-1][j-1];
				int vv2 = points[i][j-1];
				int vv3 =points[i][j];

				LaplacianUpdate(vv1,vv2);
				LaplacianUpdate(vv1,vv3);
				LaplacianUpdate(vv2,vv1);
				LaplacianUpdate(vv2,vv3);
				LaplacianUpdate(vv3,vv1);
				LaplacianUpdate(vv3,vv2);

				if(points[i-1][j-1] >= forbid && points[i][j-1] >= forbid && points[i][j] >= forbid)
				{
					TryaddTriangle(points[i][j],points[i-1][j-1], points[i][j-1],tran);
				}
				else
				{
					//TryaddMesh(points[i][j],points[i-1][j-1], points[i][j-1],tran);
					//TryAddRightMesh(points[i][j],points[i-1][j-1], points[i][j-1],tran);
					canbuild = true;
				}
			}
		}
		return canbuild;
	}

	
	void SNPGenerator::buildMeshesGreedy(std::set<int> nearest,int Rid,Eigen::Matrix4f tran,int forbid, int* except)
	{
		if(nearest.size() <=0 ) return;
		//cal the project plane
		GLVertex& R = mesh->gl_vertexes[Rid];
		VertexAttribute& Ra = vertAttrs[Rid];

		ProjectPoint tempR;
		tempR.x = 0;
		tempR.y = 0;
		double mcos = (-tran(0,1)*R.v[3]-tran(1,1)*R.v[4]-tran(2,1)*R.v[5]);
		double my[3];
		my[0] = - tran(0,1) - mcos*R.v[3];
		my[1] = - tran(1,1) - mcos*R.v[4];
		my[2] = - tran(2,1) - mcos*R.v[5];
		double mylen = NORM2(my[0],my[1],my[2]);
		my[0] /= mylen;
		my[1] /= mylen;
		my[2] /= mylen;
		double mx[3];
		mx[0] = my[1]*R.v[5]-my[2]*R.v[4];
		mx[1] = my[2]*R.v[3]-my[0]*R.v[5];
		mx[2] = my[0]*R.v[4]-my[1]*R.v[3];
		double mxlen = NORM2(mx[0],mx[1],mx[2]);
		mx[0] /= mxlen;
		mx[1] /= mxlen;
		mx[2] /= mxlen;

		//cal project coord and sort
		std::vector<ProjectPoint> nearPro;
		for(std::set<int>::iterator it = nearest.begin();it != nearest.end();it++)
		{
			ProjectPoint pp;
			pp.vertexID = *it;
			pp.v = &mesh->gl_vertexes[*it];
			//project to the plane
			double distance = NORM2(pp.v->v[0]-R.v[0], pp.v->v[1]-R.v[1], pp.v->v[2]-R.v[2]);
			double cosAng = ((pp.v->v[0]-R.v[0])*R.v[3]+(pp.v->v[1]-R.v[1])*R.v[4]+(pp.v->v[2]-R.v[2])*R.v[5])/distance;
			
			double t[3];
			t[0] = (pp.v->v[0]-R.v[0]) - cosAng*distance*R.v[3];
			t[1] = (pp.v->v[1]-R.v[1]) - cosAng*distance*R.v[4];
			t[2] = (pp.v->v[2]-R.v[2]) - cosAng*distance*R.v[5];
			pp.x = t[0]*mx[0]+t[1]*mx[1]+t[2]*mx[2];
			pp.y = t[0]*my[0]+t[1]*my[1]+t[2]*my[2];
			double tlen = NORM2(t[0],t[1],t[2]);
			pp.radius = tlen;
			double the = acos(pp.x/tlen);
			if(pp.y >= 0) pp.theta = the;
			else pp.theta = 2*M_PI - the;

		
			//if(pp.v.degree >= 2*M_PI) pp.deleted = true;
			if(pp.vertexID < forbid) pp.deleted = true;
			else pp.deleted = false;
			//pp.radius = sqrt(t[0]*t[0]+t[1]*t[1]+t[2]*t[2]);
			//pp.costheta = (t[0]*mx[0]+t[1]*mx[1]+t[2]*mx[2])/pp.radius;
			nearPro.push_back(pp);
		}
	
		std::sort(nearPro.begin(),nearPro.end(),comparePP);
		std::set<std::pair<int,int>> crossEdges;
		std::vector<bool> block1;
		for(int i = 0;i < nearPro.size();i++)
		{
			block1.push_back(false);
			if(i > 0 && (abs(nearPro[i].theta - nearPro[i-1].theta) < 0.00000001)) nearPro[i].deleted = true;
		}

		//prune by visibility
		for(int i = 0;i < nearPro.size();i++)
		{
			for(int j = i+1;j < nearPro.size();j++)
			{
				if(mesh->findEdge(nearPro[i].vertexID,nearPro[j].vertexID))
				{
					//test direction of angle
					double theta = nearPro[j].theta-nearPro[i].theta;
					if(theta < 0) theta += 2*M_PI;
					if(mesh->findMesh(Rid,nearPro[i].vertexID,nearPro[j].vertexID))//mesh->findEdge(nearPro[i].vertexID,Rid) && mesh->findEdge(nearPro[j].vertexID,Rid))
					{
						//if(theta < 0) theta += 2*M_PI;
						Ra.degree += theta;
						//block points between i and j
						if(theta < M_PI)
						{
							block1[i] = true;
							for(int k = i+1;k <j;k++)
							{
								nearPro[k].deleted = true;
								block1[k] = true;
							}
						}else{
							block1[j] = true;
							for(int k = 0;k <i;k++)
							{
								nearPro[k].deleted = true;
								block1[k] = true;
							}
							for(int k = j+1;k <nearPro.size();k++)
							{
								nearPro[k].deleted = true;
								block1[k] = true;
							}
						}
					}else{
						//block point out of triangle with i and j
						if(theta < M_PI)
						{
							for(int k = i+1;k <j;k++)
							{
								if(segmentsIntersect(nearPro[i],nearPro[j],tempR,nearPro[k])){
									nearPro[k].deleted = true;
								}
							}
						}else{
							for(int k = 0;k <i;k++)
							{
								if(segmentsIntersect(nearPro[i],nearPro[j],tempR,nearPro[k])){
									nearPro[k].deleted = true;
								}
							}
							for(int k = j+1;k <nearPro.size();k++)
							{
								if(segmentsIntersect(nearPro[i],nearPro[j],tempR,nearPro[k])){
									nearPro[k].deleted = true;
								}
							}
						}
					}
				}
			}
		}
	
		//ge all visiale points
		std::vector<ProjectPoint> canSee;
		std::vector<bool> block2;
		bool block = false;
		for(int i = 0;i < nearPro.size();i++)
		{
			if(!nearPro[i].deleted || nearPro[i].vertexID == except[0] || nearPro[i].vertexID == except[1]){
				canSee.push_back(nearPro[i]);
				block2.push_back(block);
				block = false;
			}
			if(block1[i]) block = true;
		}
		if(block && block2.size() > 0) block2[0] = true;
	
		for(int i = 0;i < canSee.size();i++)
		{
			int a = i;
			int b = i+1;
			if(i == canSee.size()-1) b = 0;
			//if(GL_Mesh::findEdge(canSee[a],Rid) && GL_Mesh::findEdge(canSee[b],Rid)){
			if(block2[b]){
				//triangle already exists

				//double theta = canSee[b].theta-canSee[a].theta;
				//if(theta < 0) theta += 2*M_PI;
				//R.degree += theta;
			}else{
				//build triangle
				double theta = canSee[b].theta-canSee[a].theta;
				if(theta < 0) theta += 2*M_PI;
				if(theta < M_PI){
					TryAddRightTriangle(Rid,canSee[a].vertexID,canSee[b].vertexID,tran);
					Ra.degree += theta;
				}
			}
		}
	}
	

	void SNPGenerator::coverPointinMeshes(int** pre,int** pts,int i, int j, int xl,int yl,Eigen::Matrix4f mc,double minx,double miny)
	{
	
		if(pre[i-1][j-1] >= 0 && pre[i-1][j] >= 0)
		{
			if(mesh->findMesh(pre[i][j],pre[i-1][j-1],pre[i-1][j]))
			{
				int v[3];
				v[0] = pre[i][j];
				v[1] = pre[i-1][j-1];
				v[2] = pre[i-1][j];
				coverPointinTriangle(v,pts,xl,yl,mc,minx,miny);
			}
		}
	
		if(pre[i-1][j-1] >= 0 && pre[i][j-1] >= 0)
		{
			if(mesh->findMesh(pre[i][j],pre[i-1][j-1],pre[i][j-1]))
			{
				int v[3];
				v[0] = pre[i][j];
				v[1] = pre[i-1][j-1];
				v[2] = pre[i][j-1];
				coverPointinTriangle(v,pts,xl,yl,mc,minx,miny);
			}
		}
	
	}

	void SNPGenerator::coverPointinTriangle(int v[3],int** pts, int xl,int yl,Eigen::Matrix4f mc,double minx,double miny)
	{
		int ix,iy,ax,ay;
		Eigen::Vector4f t[3];
		for(int i = 0;i < 3;i++)
		{
			GLVertex vv = mesh->gl_vertexes[v[i]];
			//project point to current local
			Eigen::Vector4f vec;
			vec[0] = vv.v[0];
			vec[1] = vv.v[1];
			vec[2] = vv.v[2];
			vec[3] = 1;
			t[i] = mc*vec;
		}
		int x0 = static_cast<int>((t[0][0]-minx)/devide);
		int y0 = static_cast<int>((t[0][1]-miny)/devide);
		ix = x0;
		iy = y0;
		ax = x0+1;
		ay = y0+1;
		for(int i = 1;i < 3;i++)
		{
			int xi = static_cast<int>((t[i][0]-minx)/devide);
			int yi = static_cast<int>((t[i][1]-miny)/devide);
			if(xi < ix) ix = xi;
			if(yi < iy) iy = yi;
			if(xi+1 > ax) ax = xi+1;
			if(yi+1 > ay) ay = yi+1;
		}
		for(int i = ix;i <= ax;i++)
		{
			if(i < 0) continue;
			if(i >= xl) break;
			for(int j = iy; j<= ay;j++)
			{
				if(j < 0) continue;
				if(j >= yl) break;
				double x = minx + i*devide;
				double y = miny + j*devide;
				if(inTriangle(t,x,y))
				{
					double dis1 = sqrt((t[0][0]-x)*(t[0][0]-x)+(t[0][1]-y)*(t[0][1]-y));
					double dis2 = sqrt((t[1][0]-x)*(t[1][0]-x)+(t[1][1]-y)*(t[1][1]-y));
					double dis3 = sqrt((t[2][0]-x)*(t[2][0]-x)+(t[2][1]-y)*(t[2][1]-y));
					if(dis1 <= dis2 && dis1 <= dis3) pts[i][j] = v[0];
					else if(dis2 <= dis1 && dis2 <= dis3) pts[i][j] = v[1];
					else pts[i][j] = v[2];
				}
			}
		}
	}
}