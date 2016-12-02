#include "Plane3D.h"
#include "Util.h"
#include <math.h>
#include <queue>
#include <algorithm>

#define ROTATE(a,i,j,k,l) g=a[i][j];h=a[k][l];a[i][j]=g-s*(h+g*tau);\
	a[k][l]=h+s*(g-h*tau);

using namespace std;

namespace Rgbd
{

	static void jacobi(double **a, int n, double d[], double **v, int *nrot)
	{
	  int j,iq,ip,i;
	  double tresh,theta,tau,t,sm,s,h,g,c,*b,*z;
  
	  b= new double[n+1];
	  z= new double[n+1];
	  for (ip=1;ip<=n;ip++) {
		for (iq=1;iq<=n;iq++) v[ip][iq]=0.0;
		v[ip][ip]=1.0;
	  }
	  for (ip=1;ip<=n;ip++) {
		b[ip]=d[ip]=a[ip][ip];
		z[ip]=0.0;
	  }
	  *nrot=0;
	  for (i=1;i<=50;i++) {
		sm=0.0;
		for (ip=1;ip<=n-1;ip++) {
		  for (iq=ip+1;iq<=n;iq++)
			sm += fabs(a[ip][iq]);
		}
		if (sm == 0.0) {
		  delete z;
		  delete b;
		  //free_vector(z,1,n);
		  //free_vector(b,1,n);
		  return;
		}
		if (i < 4)
		  tresh=0.2*sm/(n*n);
		else
		  tresh=0.0;
		for (ip=1;ip<=n-1;ip++) {
		  for (iq=ip+1;iq<=n;iq++) {
			g=100.0*fabs(a[ip][iq]);
			if (i > 4 && (double)(fabs(d[ip])+g) == (double)fabs(d[ip])
				&& (double)(fabs(d[iq])+g) == (double)fabs(d[iq]))
			  a[ip][iq]=0.0;
			else if (fabs(a[ip][iq]) > tresh) {
			  h=d[iq]-d[ip];
			  if ((double)(fabs(h)+g) == (double)fabs(h))
				t=(a[ip][iq])/h;
			  else {
				theta=0.5*h/(a[ip][iq]);
				t=1.0/(fabs(theta)+sqrt(1.0+theta*theta));
				if (theta < 0.0) t = -t;
			  }
			  c=1.0/sqrt(1+t*t);
			  s=t*c;
			  tau=s/(1.0+c);
			  h=t*a[ip][iq];
			  z[ip] -= h;
			  z[iq] += h;
			  d[ip] -= h;
			  d[iq] += h;
			  a[ip][iq]=0.0;
			  for (j=1;j<=ip-1;j++) {
				ROTATE(a,j,ip,j,iq)
				}
			  for (j=ip+1;j<=iq-1;j++) {
				ROTATE(a,ip,j,j,iq)
				}
			  for (j=iq+1;j<=n;j++) {
				ROTATE(a,ip,j,iq,j)
				}
			  for (j=1;j<=n;j++) {
				ROTATE(v,j,ip,j,iq)
				}
			  ++(*nrot);
			}
		  }
		}
		for (ip=1;ip<=n;ip++) {
		  b[ip] += z[ip];
		  d[ip]=b[ip];
		  z[ip]=0.0;
		}
	  }
	}

	void computePCA(PointCloud::Ptr cloud, double lamda[3], double vec[3][3])
	{
		double w[4];
		double v[4][4];
		int i;
		double wt = 1.0, wts = 0.0;
		double cx = 0;
		double cy = 0;
		double cz = 0;

		for(i=0; i< cloud->size(); i++)
		{
			PointT& p = cloud->at(i);

			cx += p.x * wt;
			cy += p.y * wt;
			cz += p.z * wt;
			wts += wt;
		}
		cx /= wts;
		cy /= wts;
		cz /= wts;

		//data for Jacobi method
		double a[4][4], *a2[4], **A, *v2[4], **v3;
		int nrot;
		A = &a2[0];
		v3 = &v2[0];
		for(i=1; i<4; i++)
		{
			A[i] = &a[i][0];
			A[i][1] = A[i][2] = A[i][3] = 0;
			v3[i] = &v[i][0];
		}

		//CV matrix
		for(i=0; i< cloud->size(); i++)
		{
			PointT& p = cloud->at(i);

			double vx = (p.x - cx);
			double vy = (p.y - cy);
			double vz = (p.z - cz);

			A[1][1] += vx*vx;
			A[1][2] += vx*vy;
			A[1][3] += vx*vz;

			A[2][2] += vy*vy;
			A[2][3] += vy*vz;

			A[3][3] += vz*vz;
		}
		A[2][1] = A[1][2];
		A[3][1] = A[1][3];
		A[3][2] = A[3][2];

		jacobi(A, 3, w, v3, &nrot);

		int l[3] = {1, 2, 3};
		if(w[l[0]] < w[l[1]]) swap(l[0], l[1]);
		if(w[l[1]] < w[l[2]]) swap(l[1], l[2]);
		if(w[l[0]] < w[l[1]]) swap(l[0], l[1]);

		for(int i = 0; i < 3; i++)
		{
			lamda[i] = w[l[i]];
			for(int j = 0; j < 3; j++)
			{
				vec[i][j] = v[l[i]][j + 1];
			}
		}
	}

	

	void computePCA(double cov[3][3], double lamda[3], double vec[3][3])
	{
		double w[4];
		double v[4][4];


		//data for Jacobi method
		double a[4][4], *a2[4], **A, *v2[4], **v3;
		int nrot;
		A = &a2[0];
		v3 = &v2[0];
		for(int i=1; i<4; i++)
		{
			A[i] = &a[i][0];
			v3[i] = &v[i][0];
		}

		//CV matrix
		for(int i=0; i< 3; i++)
		{
			for(int j = 0; j < 3; j++)
			{
				A[i + 1][j + 1] = cov[i][j];
			}
		}

		jacobi(A, 3, w, v3, &nrot);

		int l[3] = {1, 2, 3};
		if(w[l[0]] < w[l[1]]) swap(l[0], l[1]);
	    if(w[l[1]] < w[l[2]]) swap(l[1], l[2]);
		if(w[l[0]] < w[l[1]]) swap(l[0], l[1]);

		for(int i = 0; i < 3; i++)
		{
			lamda[i] = w[l[i]];
			for(int j = 0; j < 3; j++)
			{
				vec[i][j] = v[j + 1][l[i]];
			}
		}
	}

	bool More_Plane3D(const Plane3D& a,const Plane3D& b)
	{
		return a.inliers > b.inliers;
	}


	Plane3D coff_cloud(pcl::PointCloud<pcl::Normal>::Ptr normals, PointCloud::Ptr cloud)
	{
		//using coff
		double Ex = 0,Ey = 0,Ez = 0,Ex2 = 0,Ey2 = 0,Ez2 = 0,Exy = 0,Eyz = 0,Exz = 0;
		for(int i = 0;i < cloud->size();i++)
		{
			if(!ISNAN(normals->at(i).normal_x))
			{
			
					pcl::PointXYZRGB pp = cloud->at(i);
					//update
					Ex += pp.x;
					Ey += pp.y;
					Ez += pp.z;
					Ex2 += pp.x*pp.x;
					Ey2 += pp.y*pp.y;
					Ez2 += pp.z*pp.z;
					Exy += pp.x*pp.y;
					Eyz += pp.y*pp.z;
					Exz += pp.x*pp.z;

			
			}
		}
		double fem = -Ex2*Ey2*Ez2 + Ex2*Eyz*Eyz + Ez2*Exy*Exy + Ey2*Exz*Exz - 2*Exy*Eyz*Exz;
		Plane3D coff;
		if(fem - 0 < 1e-10 && fem - 0 > -1e-10){
			coff.a = (Exz*Ey2 - Exy*Eyz)/(Exy*Exy - Ex2*Ey2);
			coff.b = -(Exy*Exz - Ex2*Eyz)/(Exy*Exy - Ex2*Ey2);
			coff.c = 1;
			coff.d = 0;
		}else{
			coff.a = (Ex*Eyz*Eyz - Exz*Ey*Eyz - Ex*Ey2*Ez2 + Exy*Ey*Ez2 + Exz*Ey2*Ez - Exy*Eyz*Ez)/fem;
			coff.b = (Exz*Exz*Ey - Ex*Exz*Eyz + Ex*Exy*Ez2 - Exy*Exz*Ez - Ex2*Ey*Ez2 + Ex2*Eyz*Ez)/fem;
			coff.c = (Exy*Exy*Ez + Ex*Exz*Ey2 - Ex*Exy*Eyz - Exy*Exz*Ey + Ex2*Ey*Eyz - Ex2*Ey2*Ez)/fem;
			coff.d = 1;
		}
		double u = NORM2(coff.a, coff.b, coff.c);
		coff.a /= u;
		coff.b /= u;
		coff.c /= u;
		coff.d /= u;

		//cout<<"("<<coff.a<<","<<coff.b<<","<<coff.c<<","<<coff.d<<") "<<"inliers:"<<coff.inliers<<endl;
		return coff;
	}

	void dbplane_segment(PointCloud::Ptr cloud, PointNormal::Ptr normals,
		                 int minInlier, double sthre, double athre, double dthre,
						 std::vector<Plane3D>& coffs, vector<int>& seg)
	{
		//int minInlier = 1000;
		vector<int> valid;		
		int csize = 0;

		for(int i = 0;i < cloud->size();i++)
		{
			if(seg[i] == -1 && !ISNAN(normals->at(i).normal_x))
			{
				int id = csize;
				int count = 1;
				seg[i] = id;
				queue<int> search;
				search.push(i);

				double Ex = 0,Ey = 0,Ez = 0,Ex2 = 0,Ey2 = 0,Ez2 = 0,Exy = 0,Eyz = 0,Exz = 0;

				double avern[3];
				int averc = 0;
				avern[0] = 0;
				avern[1] = 0;
				avern[2] = 0;
				while(!search.empty())
				{
					int p = search.front();
					search.pop();
					int y = p / cloud->width;
					int x = p - y*cloud->width;

					pcl::Normal np = normals->at(p);
					
					avern[0] = (avern[0]*averc + np.normal_x)/(averc + 1);
					avern[1] = (avern[1]*averc + np.normal_y)/(averc + 1);
					avern[2] = (avern[2]*averc + np.normal_z)/(averc + 1);
					averc++;
					double norm = NORM2(avern[0], avern[1], avern[2]);
					avern[0] /= norm;
					avern[1] /= norm;
					avern[2] /= norm;
					

					pcl::PointXYZRGB pp = cloud->at(p);


					Ex += pp.x;
					Ey += pp.y;
					Ez += pp.z;
					Ex2 += pp.x*pp.x;
					Ey2 += pp.y*pp.y;
					Ez2 += pp.z*pp.z;
					Exy += pp.x*pp.y;
					Eyz += pp.y*pp.z;
					Exz += pp.x*pp.z;

					/*
					avern[0] = (Ex*Eyz*Eyz - Exz*Ey*Eyz - Ex*Ey2*Ez2 + Exy*Ey*Ez2 + Exz*Ey2*Ez - Exy*Eyz*Ez);
					avern[1] = (Exz*Exz*Ey - Ex*Exz*Eyz + Ex*Exy*Ez2 - Exy*Exz*Ez - Ex2*Ey*Ez2 + Ex2*Eyz*Ez);
					avern[2] = (Exy*Exy*Ez + Ex*Exz*Ey2 - Ex*Exy*Eyz - Exy*Exz*Ey + Ex2*Ey*Eyz - Ex2*Ey2*Ez);
					double norm = NORM2(avern[0], avern[1], avern[2]);
					avern[0] /= norm;
					avern[1] /= norm;
					avern[2] /= norm;
					*/

					for(int jx = x-1;jx <= x+1;jx++)
					{
						if(jx < 0) continue;
						if(jx >= cloud->width) continue;
						for(int jy = y-1;jy <= y+1;jy++)
						{
							if(jy < 0) continue;
							if(jy >= cloud->height) continue;
							int nid = jy*cloud->width+jx;
							if(seg[nid] == -1 && !ISNAN(normals->at(nid).normal_x))
							{
								pcl::Normal& nn = normals->at(nid);
								pcl::PointXYZRGB& pn = cloud->at(nid);
								//double disn = sqrt((np.normal_x-nn.normal_x)*(np.normal_x-nn.normal_x)+(np.normal_y-nn.normal_y)*(np.normal_y-nn.normal_y)+(np.normal_z-nn.normal_z)*(np.normal_z-nn.normal_z));
								double angn = abs(avern[0] * nn.normal_x + avern[1] * nn.normal_y + avern[2] * nn.normal_z);
								double disp = NORM2(pp.x-pn.x, pp.y-pn.y, pp.z-pn.z);
								//double dis = sqrt(disn+disp);
								if(angn > athre && disp < sthre)
								{
									search.push(nid);
									seg[nid] = id;
									count++;
								}
							}
						}
					}
				}
				double fem = -Ex2*Ey2*Ez2 + Ex2*Eyz*Eyz + Ez2*Exy*Exy + Ey2*Exz*Exz - 2*Exy*Eyz*Exz;
				Plane3D coff;
				if(fem - 0 < 1e-10 && fem - 0 > -1e-10){
					coff.a = (Exz*Ey2 - Exy*Eyz)/(Exy*Exy - Ex2*Ey2);
					coff.b = -(Exy*Exz - Ex2*Eyz)/(Exy*Exy - Ex2*Ey2);
					coff.c = 1;
					coff.d = 0;
				}else{
					coff.a = (Ex*Eyz*Eyz - Exz*Ey*Eyz - Ex*Ey2*Ez2 + Exy*Ey*Ez2 + Exz*Ey2*Ez - Exy*Eyz*Ez)/fem;
					coff.b = (Exz*Exz*Ey - Ex*Exz*Eyz + Ex*Exy*Ez2 - Exy*Exz*Ez - Ex2*Ey*Ez2 + Ex2*Eyz*Ez)/fem;
					coff.c = (Exy*Exy*Ez + Ex*Exz*Ey2 - Ex*Exy*Eyz - Exy*Exz*Ey + Ex2*Ey*Eyz - Ex2*Ey2*Ez)/fem;
					coff.d = 1;
				}
				double u = NORM2(coff.a, coff.b, coff.c);
				coff.a /= u;
				coff.b /= u;
				coff.c /= u;
				coff.d /= u;
				/*
				if(count > minInlier)
				{
					for(int j = 0;j < cloud->size();j++)
					{
						if((seg[j] == -1 || (seg[j] < valid.size() && valid[seg[j]] == -1)) && !isnan(normals->at(j).normal_x))
						{
							pcl::PointXYZRGB pj = cloud->at(j);
							if(abs(coff.a * pj.x + coff.b * pj.y + coff.c * pj.z - coff.d) < dthre)
							{
								seg[j] = id;
							
								Ex += pj.x;
								Ey += pj.y;
								Ez += pj.z;
								Ex2 += pj.x*pj.x;
								Ey2 += pj.y*pj.y;
								Ez2 += pj.z*pj.z;
								Exy += pj.x*pj.y;
								Eyz += pj.y*pj.z;
								Exz += pj.x*pj.z;
								count++;
							}
						}
					}
					double fem2 = -Ex2*Ey2*Ez2 + Ex2*Eyz*Eyz + Ez2*Exy*Exy + Ey2*Exz*Exz - 2*Exy*Eyz*Exz;
					if(fem2 - 0 < 1e-10 && fem2 - 0 > -1e-10){
						coff.a = (Exz*Ey2 - Exy*Eyz)/(Exy*Exy - Ex2*Ey2);
						coff.b = -(Exy*Exz - Ex2*Eyz)/(Exy*Exy - Ex2*Ey2);
						coff.c = 1;
						coff.d = 0;
					}else{
						coff.a = (Ex*Eyz*Eyz - Exz*Ey*Eyz - Ex*Ey2*Ez2 + Exy*Ey*Ez2 + Exz*Ey2*Ez - Exy*Eyz*Ez)/fem;
						coff.b = (Exz*Exz*Ey - Ex*Exz*Eyz + Ex*Exy*Ez2 - Exy*Exz*Ez - Ex2*Ey*Ez2 + Ex2*Eyz*Ez)/fem;
						coff.c = (Exy*Exy*Ez + Ex*Exz*Ey2 - Ex*Exy*Eyz - Exy*Exz*Ey + Ex2*Ey*Eyz - Ex2*Ey2*Ez)/fem;
						coff.d = 1;
					}
					u = sqrt(coff.a*coff.a + coff.b*coff.b + coff.c*coff.c);
					coff.a /= u;
					coff.b /= u;
					coff.c /= u;
					coff.d /= u;
				}
				*/
				coff.inliers = count;
				coff.id = id;
				csize++;
				if(coff.inliers > minInlier)
				{
					
					coffs.push_back(coff);
					valid.push_back(coffs.size()-1);
				}
				else
				{
					valid.push_back(-1);
				}
			}
		}
		sort(coffs.begin(), coffs.end(),More_Plane3D);
		for(int i = 0;i < coffs.size();i++)
		{
			coffs[i].inliers = 0;
			valid[coffs[i].id] = i;
		}
		for(int i = 0;i < cloud->size();i++)
		{
			if(seg[i] >= 0)
			{
				if(valid[seg[i]] < 0) seg[i] = -1;
				else
				{
					pcl::PointXYZRGB& pn = cloud->at(i);
					pcl::Normal& nn = normals->at(i);
					Plane3D& coff = coffs[valid[seg[i]]];
					double angn = abs(coff.a * nn.normal_x + coff.b * nn.normal_y + coff.c * nn.normal_z);
					if(angn > athre)
					{
						seg[i] = valid[seg[i]];
						coffs[seg[i]].inliers++;
					}
					else seg[i] = -1;
				}
			}
			
		}
		
#ifdef DEBUG_PLANE_SEGMENT
		  for(int i =0;i < coffs.size();i++) cout<<"("<<coffs[i].a<<","<<coffs[i].b<<","<<coffs[i].c<<","<<coffs[i].d<<") "<<"inliers:"<<coffs[i].inliers<<endl;
		  cout<<endl;
#endif
	}

	Plane3D& coffCal::getCoff()	
	{
		if(plane.inliers <= 0)
		{
			double fem = -Ex2*Ey2*Ez2 + Ex2*Eyz*Eyz + Ez2*Exy*Exy + Ey2*Exz*Exz - 2*Exy*Eyz*Exz;
			Plane3D coff;
			if(fem - 0 < 1e-10 && fem - 0 > -1e-10){
				plane.a = (Exz*Ey2 - Exy*Eyz)/(Exy*Exy - Ex2*Ey2);
				plane.b = -(Exy*Exz - Ex2*Eyz)/(Exy*Exy - Ex2*Ey2);
				plane.c = 1;
				plane.d = 0;
			}else{
				plane.a = (Ex*Eyz*Eyz - Exz*Ey*Eyz - Ex*Ey2*Ez2 + Exy*Ey*Ez2 + Exz*Ey2*Ez - Exy*Eyz*Ez)/fem;
				plane.b = (Exz*Exz*Ey - Ex*Exz*Eyz + Ex*Exy*Ez2 - Exy*Exz*Ez - Ex2*Ey*Ez2 + Ex2*Eyz*Ez)/fem;
				plane.c = (Exy*Exy*Ez + Ex*Exz*Ey2 - Ex*Exy*Eyz - Exy*Exz*Ey + Ex2*Ey*Eyz - Ex2*Ey2*Ez)/fem;
				plane.d = 1;
			}
			double u = NORM2(plane.a, plane.b, plane.c);
			plane.a /= u;
			plane.b /= u;
			plane.c /= u;
			plane.d /= u;
			plane.inliers = inliers;
		}

		return plane;
	}

	void coffCal::calAxis()
	{
		computePCA(cov, lamda, vec);
	}

	double p_distance(Plane3D& a, Plane3D& b)
	{
		return sqrt((a.a - b.a) * (a.a - b.a) + (a.b - b.b) * (a.b - b.b) + (a.c - b.c) * (a.c - b.c) + (a.d - b.d) * (a.d - b.d));
	}

	double p_distance(Plane3D& p, double a, double b, double c, double d)
	{
		return sqrt((p.a - a) * (p.a - a) + (p.b - b) * (p.b - b) + (p.c - c) * (p.c - c) + (p.d - d) * (p.d - d));
	}

	double p_angle(Plane3D& a, Plane3D& b)
	{
		return abs(a.a * b.a + a.b * b.b + a.c * b.c)/sqrt(a.a*a.a + a.b*a.b + a.c*a.c)/sqrt(b.a*b.a + b.b*b.b + b.c*b.c);
	}

	double p_angle(Plane3D& p, double a, double b, double c, double d)
	{
		return abs(p.a * a + p.b * b + p.c * c)/sqrt(p.a*p.a + p.b*p.b + p.c*p.c)/sqrt(a*a + b*b + c*c);
	}

	double p_angle(Plane3D& p, coffCal& c)
	{
		double* normal = c.vec[2];
		return abs(p.a * normal[0] + p.b * normal[1] + p.c * normal[2])/sqrt(p.a*p.a + p.b*p.b + p.c*p.c)/sqrt(normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2]);
	}

	double p_angle(coffCal& a, coffCal& b)
	{
		double* n1 = a.vec[2];
		double* n2 = b.vec[2];
		return abs(n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2])/sqrt(n1[0]*n1[0] + n1[1]*n1[1] + n1[2]*n1[2])/sqrt(n2[0]*n2[0] + n2[1]*n2[1] + n2[2]*n2[2]);
	}
}