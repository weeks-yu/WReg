/**************************************
*                                     *
*   Jeff Molofee's Basecode Example   *
*          nehe.gamedev.net           *
*                2001                 *
*                                     *
**************************************/

#include <windows.h>												// Header File For Windows
#include <gl\gl.h>													// Header File For The OpenGL32 Library
#include <gl\glu.h>													// Header File For The GLu32 Library
#include "NeHeGL.h"													// Header File For NeHeGL

#include "math.h"												    // NEW: Needed For Sqrtf
#include "ArcBall.h"												// NEW: ArcBall Header
#include "glwindows.h"
#include "TriangleMesh.h"

#pragma comment( lib, "opengl32.lib" )								// Search For OpenGL32.lib While Linking
#pragma comment( lib, "glu32.lib" )									// Search For GLu32.lib While Linking
//#pragma comment( lib, "glaux.lib" )									// Search For GLaux.lib While Linking

#ifndef CDS_FULLSCREEN												// CDS_FULLSCREEN Is Not Defined By Some
#define CDS_FULLSCREEN 4											// Compilers. By Defining It This Way,
#endif																// We Can Avoid Errors

GL_Window*	g_window;
Keys*		g_keys;

// User Defined Variables
GLUquadricObj *quadratic;											// Used For Our Quadric

const float PI2 = 2.0*3.1415926535f;								// PI Squared

Matrix4fT   Transform   = {  1.0f,  0.0f,  0.0f,  0.0f,				// NEW: Final Transform
                             0.0f,  1.0f,  0.0f,  0.0f,
                             0.0f,  0.0f,  1.0f,  0.0f,
                             0.0f,  0.0f,  0.0f,  1.0f };

Matrix3fT   LastRot     = {  1.0f,  0.0f,  0.0f,					// NEW: Last Rotation
                             0.0f,  1.0f,  0.0f,
                             0.0f,  0.0f,  1.0f };

Matrix3fT   ThisRot     = {  1.0f,  0.0f,  0.0f,					// NEW: This Rotation
                             0.0f,  1.0f,  0.0f,
                             0.0f,  0.0f,  1.0f };

ArcBallT    ArcBall(640.0f, 480.0f);				                // NEW: ArcBall Instance
Point2fT    MousePt;												// NEW: Current Mouse Point
bool        isClicked  = false;										// NEW: Clicking The Mouse?
bool        isRClicked = false;										// NEW: Clicking The Right Mouse Button?
bool        isDragging = false;					                    // NEW: Dragging The Mouse?

bool        isMClicked = false;
float zoomSize = 1.0;
Point2fT    LastMPt;	

float LastMoveX = 0.0f;
float LastMoveY = 0.0f;

float NowMoveX = 0.0f;
float NowMoveY = 0.0f;

float g_width = 640.0f;
float g_height = 480.0f;

float Materials[][4][4] = {
    //»ÆÍ­
    {{0.329412, 0.223529, 0.027451, 1.000000},
    {0.780392, 0.568627, 0.113725, 1.000000},
    {0.992157, 0.941176, 0.807843, 1.000000},
    {27.897400}},
 
    //ÇàÍ­
    {{0.212500, 0.127500, 0.054000, 1.000000},
    {0.714000, 0.428400, 0.181440, 1.000000},
    {0.393548, 0.271906, 0.166721, 1.000000},
    {25.600000}},
 
    //ÁÁÇàÍ­
    {{0.250000, 0.148000, 0.064750, 1.000000},
    {0.400000, 0.236800, 0.103600, 1.000000},
    {0.774597, 0.458561, 0.200621, 1.000000},
    {76.800003}},
 
    //¸õ
    {{0.250000, 0.250000, 0.250000, 1.000000},
    {0.400000, 0.400000, 0.400000, 1.000000},
    {0.774597, 0.774597, 0.774597, 1.000000},
    {76.800003}},
 
    //Í­
    {{0.191250, 0.073500, 0.022500, 1.000000},
    {0.703800, 0.270480, 0.082800, 1.000000},
    {0.256777, 0.137622, 0.086014, 1.000000},
    {12.800000}},
 
 
//ÁÁÍ­
    {{0.229500, 0.088250, 0.027500, 1.000000},
    {0.550800, 0.211800, 0.066000, 1.000000},
    {0.580594, 0.223257, 0.069570, 1.000000},
    {51.200001}},
 
//½ð
    {{0.247250, 0.199500, 0.074500, 1.000000},
    {0.751640, 0.606480, 0.226480, 1.000000},
    {0.628281, 0.555802, 0.366065, 1.000000},
    {51.200001}},
 
//ÁÁ½ð
    {{0.247250, 0.224500, 0.064500, 1.000000},
    {0.346150, 0.314300, 0.090300, 1.000000},
    {0.797357, 0.723991, 0.208006, 1.000000},
    {83.199997}},
 
//°×À¯
    {{0.105882, 0.058824, 0.113725, 1.000000},
    {0.427451, 0.470588, 0.541176, 1.000000},
    {0.333333, 0.333333, 0.521569, 1.000000},
    {9.846150}},
 
//Òø
    {{0.192250, 0.192250, 0.192250, 1.000000},
    {0.507540, 0.507540, 0.507540, 1.000000},
    {0.508273, 0.508273, 0.508273, 1.000000},
    {51.200001}},
 
//ÁÁÒøÉ«
    {{0.231250, 0.231250, 0.231250, 1.000000},
    {0.277500, 0.277500, 0.277500, 1.000000},
    {0.773911, 0.773911, 0.773911, 1.000000},
    {89.599998}},
 
//ôä´ä¡¢×æÄ¸ÂÌ
    {{0.021500, 0.174500, 0.021500, 0.550000},
    {0.075680, 0.614240, 0.075680, 0.550000},
    {0.633000, 0.727811, 0.633000, 0.550000},
    {76.800003}},
 
//±ÌÓñ
    {{0.135000, 0.222500, 0.157500, 0.950000},
    {0.540000, 0.890000, 0.630000, 0.950000},
    {0.316228, 0.316228, 0.316228, 0.950000},
    {12.800000}},
 
//ºÚê×Ê¯
    {{0.053750, 0.050000, 0.066250, 0.820000},
    {0.182750, 0.170000, 0.225250, 0.820000},
    {0.332741, 0.328634, 0.346435, 0.820000},
    {38.400002}},
 
//ÕäÖé
    {{0.250000, 0.207250, 0.207250, 0.922000},
    {1.000000, 0.829000, 0.829000, 0.922000},
    {0.296648, 0.296648, 0.296648, 0.922000},
    {11.264000}},
 
//ºì±¦Ê¯
    {{0.174500, 0.011750, 0.011750, 0.550000},
    {0.614240, 0.041360, 0.041360, 0.550000},
    {0.727811, 0.626959, 0.626959, 0.550000},
    {76.800003}},
 
//ÂÌ±¦Ê¯¡¢ÂÌËÉÊ¯
    {{0.100000, 0.187250, 0.174500, 0.800000},
    {0.396000, 0.741510, 0.691020, 0.800000},
    {0.297254, 0.308290, 0.306678, 0.800000},
    {12.800000}},
 
//ºÚËÜÁÏ
    {{0.000000, 0.000000, 0.000000, 1.000000},
    {0.010000, 0.010000, 0.010000, 1.000000},
    {0.500000, 0.500000, 0.500000, 1.000000},
    {32.000000}},
 
 
//ºÚÏð½º
    {{0.020000, 0.020000, 0.020000, 1.000000},
    {0.010000, 0.010000, 0.010000, 1.000000},
    {0.400000, 0.400000, 0.400000, 1.000000},
    {10.000000}},
 
//×ÏÂÞÀ¼
    {{0.110000, 0.060000, 0.090000, 1.000000},
    {0.430000, 0.470000, 0.540000, 1.000000},
    {0.330000, 0.330000, 0.520000, 1.000000},
    {22.000000}}
};



BOOL Initialize (GL_Window* window, Keys* keys)						// Any GL Init Code & User Initialiazation Goes Here
{
	g_window	= window;
	g_keys		= keys;

	// Start Of User Initialization
    isClicked   = false;								            // NEW: Clicking The Mouse?
    isDragging  = false;							                // NEW: Dragging The Mouse?

	glClearColor (0.0f, 0.0f, 0.0f, 0.5f);							// Black Background
	glClearDepth (1.0f);											// Depth Buffer Setup
	glDepthFunc (GL_LEQUAL);										// The Type Of Depth Testing (Less Or Equal)
	glEnable (GL_DEPTH_TEST);										// Enable Depth Testing
	glShadeModel (GL_FLAT);											// Select Flat Shading (Nice Definition Of Objects)
	glHint (GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);				// Set Perspective Calculations To Most Accurate

	quadratic=gluNewQuadric();										// Create A Pointer To The Quadric Object
	gluQuadricNormals(quadratic, GLU_SMOOTH);						// Create Smooth Normals
	gluQuadricTexture(quadratic, GL_TRUE);							// Create Texture Coords

	//float ambientLight[4]={0.2,0.2,0.2,0.2};
	//glLightfv(GL_LIGHT0,GL_AMBIENT, ambientLight);
	glEnable(GL_LIGHT0);											// Enable Default Light
	glEnable(GL_LIGHTING);											// Enable Lighting

	glEnable(GL_COLOR_MATERIAL);									// Enable Color Material

	return TRUE;													// Return TRUE (Initialization Successful)
}

void Deinitialize (void)											// Any User DeInitialization Goes Here
{
	gluDeleteQuadric(quadratic);
}

void Update (DWORD milliseconds)									// Perform Motion Updates Here
{
	if (g_keys->keyDown [VK_ESCAPE] == TRUE)						// Is ESC Being Pressed?
		TerminateApplication (g_window);							// Terminate The Program

	if (g_keys->keyDown [VK_F1] == TRUE)							// Is F1 Being Pressed?
		ToggleFullscreen (g_window);								// Toggle Fullscreen Mode

    if (isRClicked)													// If Right Mouse Clicked, Reset All Rotations
    {
		Matrix3fSetIdentity(&LastRot);								// Reset Rotation
		Matrix3fSetIdentity(&ThisRot);								// Reset Rotation
        Matrix4fSetRotationFromMatrix3f(&Transform, &ThisRot);		// Reset Rotation

		LastMoveX = 
		NowMoveX =
        LastMoveY = 
		NowMoveY = 0.0f;
    }

    if (!isDragging)												// Not Dragging
    {
        if (isClicked)												// First Click
        {
			isDragging = true;										// Prepare For Dragging
			LastRot = ThisRot;										// Set Last Static Rotation To Last Dynamic One
			ArcBall.click(&MousePt);								// Update Start Vector And Prepare For Dragging

			LastMoveX = NowMoveX;
			LastMoveY = NowMoveY;
			LastMPt = MousePt;
        }
    }
    else
    {
        if (isClicked && !isMClicked)												// Still Clicked, So Still Dragging
        {
            Quat4fT     ThisQuat;

            ArcBall.drag(&MousePt, &ThisQuat);						// Update End Vector And Get Rotation As Quaternion
            Matrix3fSetRotationFromQuat4f(&ThisRot, &ThisQuat);		// Convert Quaternion Into Matrix3fT
            Matrix3fMulMatrix3f(&ThisRot, &LastRot);				// Accumulate Last Rotation Into This One
            Matrix4fSetRotationFromMatrix3f(&Transform, &ThisRot);	// Set Our Final Transform's Rotation From This One
        }else if (isClicked && isMClicked){
			NowMoveX = LastMoveX + (MousePt.s.X - LastMPt.s.X)/g_width;
			NowMoveY = LastMoveY + (MousePt.s.Y - LastMPt.s.Y)/g_height;
		}
        else														// No Longer Dragging
            isDragging = false;
    }
}

void Torus(float MinorRadius, float MajorRadius)					// Draw A Torus With Normals
{
	int i, j;
	glBegin( GL_TRIANGLE_STRIP );									// Start A Triangle Strip
		for (i=0; i<20; i++ )										// Stacks
		{
			for (j=-1; j<20; j++)									// Slices
			{
				float wrapFrac = (j%20)/(float)20;
				float phi = PI2*wrapFrac;
				float sinphi = float(sin(phi));
				float cosphi = float(cos(phi));

				float r = MajorRadius + MinorRadius*cosphi;

				glNormal3f(float(sin(PI2*(i%20+wrapFrac)/(float)20))*cosphi, sinphi, float(cos(PI2*(i%20+wrapFrac)/(float)20))*cosphi);
				glVertex3f(float(sin(PI2*(i%20+wrapFrac)/(float)20))*r,MinorRadius*sinphi,float(cos(PI2*(i%20+wrapFrac)/(float)20))*r);

				glNormal3f(float(sin(PI2*(i+1%20+wrapFrac)/(float)20))*cosphi, sinphi, float(cos(PI2*(i+1%20+wrapFrac)/(float)20))*cosphi);
				glVertex3f(float(sin(PI2*(i+1%20+wrapFrac)/(float)20))*r,MinorRadius*sinphi,float(cos(PI2*(i+1%20+wrapFrac)/(float)20))*r);
			}
		}
	glEnd();														// Done Torus
}

void Draw (GLMesh* mesh)
{
	//GLfloat lightposition[] = {1.0, 1.0, 1.0, 1.0 };
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);				// Clear Screen And Depth Buffer
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();												// Reset The Current Modelview Matrix
	//glLightfv(GL_LIGHT0, GL_POSITION, lightposition);
	glTranslatef(-1.5f,0.0f,-6.0f);									// Move Left 1.5 Units And Into The Screen 6.0
	/*Îª¹âÕÕÄ£ÐÍÖ¸¶¨²ÄÖÊ²ÎÊý*/
	int mtr = 8;
	
	glMaterialfv(GL_FRONT, GL_AMBIENT, Materials[mtr][0]);
	glMaterialfv(GL_FRONT, GL_DIFFUSE, Materials[mtr][1]);
	glMaterialfv(GL_FRONT, GL_SPECULAR, Materials[mtr][2]);
	glMaterialfv(GL_FRONT, GL_SHININESS, Materials[mtr][3]);
	glColorMaterial(GL_FRONT,GL_AMBIENT);
	glEnable(GL_COLOR_MATERIAL);

	//lightposition[3] = 1.0 * zoomSize;
	
	
	glScalef(zoomSize,zoomSize,zoomSize);
	glTranslatef (NowMoveX*5, -NowMoveY*5, 0.0);
    glPushMatrix();													// NEW: Prepare Dynamic Transform
    glMultMatrixf(Transform.M);										// NEW: Apply Dynamic Transform
	
	
	
	if(mesh->gl_meshes.size() > 0){
		glBegin(GL_TRIANGLES);
		for(std::map<std::tuple<int,int,int>,GLTriangle>::iterator it = mesh->gl_meshes.begin();it != mesh->gl_meshes.end();it++){
			//glBegin(GL_LINE_LOOP);
			GLTriangle m = it->second;
			GLVertex v[3];
			v[0] = mesh->gl_vertexes.at(m.vertexes[0]);
			v[1] = mesh->gl_vertexes.at(m.vertexes[1]);
			v[2] = mesh->gl_vertexes.at(m.vertexes[2]);
			glNormal3f(v[0].v[3], v[0].v[4], v[0].v[5]);
			glVertex3f(v[0].v[0], v[0].v[1], v[0].v[2]);
			glNormal3f(v[1].v[3], v[1].v[4], v[1].v[5]);
			glVertex3f(v[1].v[0], v[1].v[1], v[1].v[2]);
			glNormal3f(v[2].v[3], v[2].v[4], v[2].v[5]);
			glVertex3f(v[2].v[0], v[2].v[1], v[2].v[2]);
			//glEnd();
		}
		glEnd();
	}
	//
	//else if(Sdf::sdf != NULL){
	//	glBegin(GL_POINTS);
	//	glColor3f(1.0,1.0,1.0); //ÉèÖÃµãÑÕÉ«
	//	Sdf::sdf->TraversalDraw();
	//	glEnd();
	//	
	//}else if(Sdf::tsdf != NULL){
	//	glBegin(GL_POINTS);
	//	glColor3f(1.0,1.0,1.0); //ÉèÖÃµãÑÕÉ«
	//	Sdf::tsdf->TraversalDraw();
	//	glEnd();
	//	
	//}
	

	

	/*
	if(Sdf::sdf->gen != NULL){
		glBegin(GL_TRIANGLES);
		for (int i = 0; i < Sdf::sdf->gen->m_nTriangles; i++) {
			VECTOR3D vec1, vec2, normal;
			unsigned int id[3];
			id[0] = Sdf::sdf->gen->m_piTriangleIndices[i*3];
			id[1] = Sdf::sdf->gen->m_piTriangleIndices[i*3+1];
			id[2] =Sdf::sdf->gen->m_piTriangleIndices[i*3+2];
			for(int j = 0;j < 3;j++){
				glNormal3f(Sdf::sdf->gen->m_pvec3dNormals[id[j]][0],Sdf::sdf->gen->m_pvec3dNormals[id[j]][1],Sdf::sdf->gen->m_pvec3dNormals[id[j]][2]);
				glVertex3f(Sdf::sdf->gen->m_ppt3dVertices[id[j]][0],Sdf::sdf->gen->m_ppt3dVertices[id[j]][1],Sdf::sdf->gen->m_ppt3dVertices[id[j]][2]);
			}
			
		}
		glEnd();
	}
	*/
	
	
	
	
    glPopMatrix();													// NEW: Unapply Dynamic Transform


	glFlush ();														// Flush The GL Rendering Pipeline
}


/*
int DrawGLScene(GLvoid)									// Here's Where We Do All The Drawing
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);	// Clear Screen And Depth Buffer
	glLoadIdentity();									// Reset The Current Modelview Matrix
	glTranslatef(-1.5f,0.0f,-6.0f);						// Move Left 1.5 Units And Into The Screen 6.0
	
	glRotatef(rtri,0.0f,1.0f,0.0f);
	glBegin(GL_TRIANGLES);							// Start Drawing A Triangle
	for(std::vector<GLMesh>::iterator it = GL_Mesh::gl_meshes->begin();it != GL_Mesh::gl_meshes->end();it++){
		GLMesh m = *it;
		glNormal3f(m.mesh[0][3], m.mesh[0][4], m.mesh[0][5]);
		glVertex3f(m.mesh[0][0], m.mesh[0][1], m.mesh[0][2]);
		glNormal3f(m.mesh[1][3], m.mesh[1][4], m.mesh[1][5]);
		glVertex3f(m.mesh[1][0], m.mesh[1][1], m.mesh[1][2]);
		glNormal3f(m.mesh[2][3], m.mesh[2][4], m.mesh[2][5]);
		glVertex3f(m.mesh[2][0], m.mesh[2][1], m.mesh[2][2]);
	}
    glEnd();	
	rtri+=3.0f;	

	glEnable(GL_LIGHTING);
	return TRUE;										// Keep Going
}
*/
