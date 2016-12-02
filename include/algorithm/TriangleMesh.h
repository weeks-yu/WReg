#ifndef TRIANGLE_MESH_H
#define TRIANGLE_MESH_H

#include <vector>
#include <map>
#include <tuple>
#include <pcl/point_types.h>

namespace Rgbd
{
	class GLMesh;

	
	struct GLTriangle{
		int vertexes[3];
		inline GLTriangle(int v1,int v2,int v3)
		{
			vertexes[0] = v1;
			vertexes[1] = v2;
			vertexes[2] = v3;
		}
	};

	struct GLVertex{
		double v[6];  //x,y,z,nx,ny,nz
	};


	class GLMesh{
	public:
	  std::vector<GLVertex> gl_vertexes;
	  std::map<std::tuple<int,int,int>,GLTriangle> gl_meshes;
	  std::map<std::pair<int,int>,int> gl_edges;

	  inline void addEdge(int a,int b)
	  {
		  
		  if(a < b)
	     {
			 std::pair<int,int> ins(a,b);
			 gl_edges[ins]++;
		  }
		  else if(a > b)
		 {
			 std::pair<int,int> ins(b,a);
			 gl_edges[ins]++;
		 }
	  }

	  inline bool findEdge(int a,int b)
	  {
		std::map<std::pair<int,int>,int>::iterator it;
		if(a < b) it = gl_edges.find(std::pair<int,int>(a,b));
		else if(a > b) it = gl_edges.find(std::pair<int,int>(b,a));
    
		if(it !=  gl_edges.end()) return true;
		else return false;
	  }

	  inline bool findMesh(int a,int b,int c)
	  {
		 if (a>b) std::swap(a, b);
		if (b>c) 
		{ 
			std::swap(b, c);
			if (a > b) std::swap(a, b);
		}
		std::map<std::tuple<int,int,int>,GLTriangle>::iterator it = gl_meshes.find(std::tuple<int,int,int>(a,b,c));
    
		if(it !=  gl_meshes.end()) return true;
		else return false;
	  }

	  inline int findEdgeDegree(int a,int b)
	  {
		std::map<std::pair<int,int>,int>::iterator it;
		if(a < b) it = gl_edges.find(std::pair<int,int>(a,b));
		else if(a > b) it = gl_edges.find(std::pair<int,int>(b,a));
		else return 0;
    
		if(it !=  gl_edges.end()) return it->second;
		else return 0;
	  }
	};
}

#endif // TRIANGLE_MESH_H
