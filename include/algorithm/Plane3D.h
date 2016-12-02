#ifndef PLANE3D_H
#define PLANE3D_H

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

//#define DEBUG_PLANE_SEGMENT

namespace Rgbd
{
	typedef pcl::PointXYZRGB PointT;
    typedef pcl::PointCloud<PointT> PointCloud;
	typedef pcl::PointCloud<pcl::Normal> PointNormal;

	

	struct Plane3D{
		double a;
		double b;
		double c;
		double d;
		int inliers;

		int id; //∂ÓÕ‚ ≈≈–Ú”√
	};

	struct coffCal
	{
		//belong to seged Blocks mark
		int seged_id;

		double Ex;
		double Ey;
		double Ez;
		double Ex2;
		double Ey2;
		double Ez2;
		double Exy;
		double Eyz;
		double Exz;
		int inliers;
		double cov[3][3];
		double lamda[3];
		double vec[3][3];
		Plane3D plane;

		coffCal():seged_id(-1),Ex(0),Ey(0),Ez(0),Ex2(0),Ey2(0),Ez2(0),Exy(0),Eyz(0),Exz(0),inliers(0)
		{
			for(int i = 0; i < 3; i++)
			{
				for(int j = 0; j < 3; j++)
				{
					cov[i][j] = 0;
				}
			}
			plane.inliers = 0;
		}

		void addPoint(PointT pp)
		{

			Ex += pp.x;
			Ey += pp.y;
			Ez += pp.z;
			Ex2 += pp.x*pp.x;
			Ey2 += pp.y*pp.y;
			Ez2 += pp.z*pp.z;
			Exy += pp.x*pp.y;
			Eyz += pp.y*pp.z;
			Exz += pp.x*pp.z;
			inliers++;

			if(inliers > 1)
			{
				double v[3];
				v[0] = (pp.x - Ex / inliers);
				v[1] = (pp.y - Ey  / inliers);
				v[2] = (pp.z - Ez  / inliers);

				for(int i = 0; i < 3; i++)
				{
					for(int j = 0; j < 3; j++)
					{
						cov[i][j] += v[i]*v[j] * inliers / (inliers - 1);
					}
				}
			}
		}

		void merge(coffCal& b)
		{
			double dEsp[3];
			dEsp[0] = Ex - b.Ex;
			dEsp[1] = Ey - b.Ey;
			dEsp[2] = Ez - b.Ez;

			for(int i = 0; i < 3; i++)
		    {
				for(int j = 0; j < 3; j++)
				{
					cov[i][j] = cov[i][j] + b.cov[i][j] + dEsp[i]*dEsp[j] * inliers * b.inliers / (inliers + b.inliers);
				}
			}

			Ex += b.Ex;
			Ey += b.Ey;
			Ez += b.Ez;
			Ex2 += b.Ex2;
			Ey2 += b.Ey2;
			Ez2 += b.Ez2;
			Exy += b.Exy;
			Eyz += b.Eyz;
			Exz += b.Exz;
			inliers += b.inliers;

			plane.inliers = 0;
			getCoff();
		    calAxis();
		 }

		/*
		void addPointForCov(PointT pp)
		{
			double v[3];
			v[0] = (pp.x - Ex / inliers);
			v[1] = (pp.y - Ey  / inliers);
			v[2] = (pp.z - Ez  / inliers);
			
			for(int i = 0; i < 3; i++)
			{
				for(int j = 0; j < 3; j++)
				{
					cov[i][j] += v[i]*v[j];
				}
			}
		}
		*/

		Plane3D& getCoff();
		void calAxis();
	};


	double p_distance(Plane3D& a, Plane3D& b);
	double p_distance(Plane3D& p, double a, double b, double c, double d);

	double p_angle(Plane3D& a, Plane3D& b);
	double p_angle(Plane3D& p, double a, double b, double c, double d);

	double p_angle(Plane3D& p, coffCal& c);
	double p_angle(coffCal& a, coffCal& b);

	Plane3D coff_cloud(pcl::PointCloud<pcl::Normal>::Ptr normals, PointCloud::Ptr cloud);

	void dbplane_segment(PointCloud::Ptr cloud, PointNormal::Ptr normals,
		                 int minInlier, double sthre, double athre, double dthre,
						 std::vector<Plane3D>& coffs, std::vector<int>& seg);
}

#endif // PLANE3D_H
