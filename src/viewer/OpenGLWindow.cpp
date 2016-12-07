#include "OpenGLWindow.h"
#include <gl/GLU.h>

OpenGLWindow::OpenGLWindow()
{
	mesh = nullptr;
	m_camera << 0.0f, 0.0f, 10.0f;
	m_center << 0.0f, 0.0f, 0.0f;
	m_up << 0.0f, 1.0f, 0.0f;
	m_scale << 1.0f, 1.0f, 1.0f;

	// 白蜡
	Materials[0][0] = 0.105882;
	Materials[0][1] = 0.058824;
	Materials[0][2] = 0.113725;
	Materials[0][3] = 1.000000;

	Materials[1][0] = 0.427451;
	Materials[1][1] = 0.470588;
	Materials[1][2] = 0.541176;
	Materials[1][3] = 1.000000;

	Materials[2][0] = 0.333333;
	Materials[2][1] = 0.333333;
	Materials[2][2] = 0.521569;
	Materials[2][3] = 1.000000;

	Materials[3][0] = 9.846150;
	Materials[3][1] = 0;
	Materials[3][2] = 0;
	Materials[3][3] = 0;

	openglInit = false;
}

OpenGLWindow::~OpenGLWindow()
{
	if (mesh)
	{
		delete mesh;
	}
}

void OpenGLWindow::setMesh(GLMesh *mesh_)
{
	*mesh = *mesh_;
}

void OpenGLWindow::initializeGL()
{
	initializeOpenGLFunctions();
	openglInit = true;

	glClearColor(0.0, 0.0, 0.0, 1.0);

	int w = this->size().width();
	int h = this->size().height();
	glViewport(0, 0, (GLsizei)w, (GLsizei)h);

	fov = 60.0;
	aspect = (GLfloat)w / (GLfloat)h;
	zNear = 1.0;
	zFar = 20.0;
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(fov, aspect, zNear, zFar);
	//glOrtho(-5, 5, -5, 5, -10, 10);
}

void OpenGLWindow::paintGL()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(m_camera.x(), m_camera.y(), m_camera.z(), m_center.x(), m_center.y(), m_center.z(), m_up.x(), m_up.y(), m_up.z());
	glScalef(m_scale.x(), m_scale.y(), m_scale.z());
	//drawMesh();

	static const GLfloat verts[8][3] = {
		{ -1, -1, -1 },
		{ 1, -1, -1 },
		{ 1, 1, -1 },
		{ -1, 1, -1 },
		{ -1, -1, 1 },
		{ 1, -1, 1 },
		{ 1, 1, 1 },
		{ -1, 1, 1}
	};

	static const GLubyte indices[][4] = {
		{ 0, 1, 2, 3 },
		{ 4, 7, 6, 5 },
		{ 0, 4, 5, 1 },
		{ 3, 2, 6, 7 },
		{ 0, 3, 7, 4 },
		{ 1, 5, 6, 2 }
	};

// 	static const GLubyte indices[][4] = {
// 		{ 0, 3, 2, 1 },
// 		{ 4, 5, 6, 7 },
// 		{ 0, 1, 5, 4 },
// 		{ 3, 7, 6, 2 },
// 		{ 0, 4, 7, 3 },
// 		{ 1, 2, 6, 5 }
// 	};

	static const GLfloat colors[][4] = {
		{ 1, 0, 0, 1 },
		{ 1, 1, 0, 1 },
		{ 1, 0, 1, 1 },
		{ 0, 1, 0, 1 },
		{ 0, 1, 1, 1 },
		{ 0, 0, 1, 1 }
	};

	glPolygonMode(GL_FRONT, GL_FILL);
	glBegin(GL_QUADS);
	for (int i = 0; i < 6; i++)
	{
		glColor4fv(colors[i]);
		glVertex3fv(verts[indices[i][0]]);
		glVertex3fv(verts[indices[i][1]]);
		glVertex3fv(verts[indices[i][2]]);
		glVertex3fv(verts[indices[i][3]]);
	}
	glEnd();

	glFlush();
}

void OpenGLWindow::resizeGL(int w, int h)
{
	if (openglInit)
	{
		glViewport(this->position().x(), this->position().y(), (GLsizei)w, (GLsizei)h);

		aspect = (GLfloat)w / (GLfloat)h;
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		gluPerspective(fov, aspect, zNear, zFar);
//		glOrtho(-50, 50, -50, 50, -100, 100);
	}
}

void OpenGLWindow::mousePressEvent(QMouseEvent *event)
{
	lastPos = event->pos();
}

void OpenGLWindow::mouseMoveEvent(QMouseEvent *event)
{
	float dx = event->x() - lastPos.x();
	float dy = event->y() - lastPos.y();

	if (event->buttons() & Qt::MidButton)
	{
		translate(dx, dy);
	}
	else if (event->buttons() & Qt::LeftButton)
	{
		rotate(-dx, dy);
	}
	lastPos = event->pos();
}

void OpenGLWindow::wheelEvent(QWheelEvent *event)
{
	int pixel = event->delta();
	if (pixel != 0)
	{
		scale(pixel);
	}
}

void OpenGLWindow::drawMesh()
{
	/*为光照模型指定材质参数*/
	int mtr = 8;

	glMaterialfv(GL_FRONT, GL_AMBIENT, Materials[0]);
	glMaterialfv(GL_FRONT, GL_DIFFUSE, Materials[1]);
	glMaterialfv(GL_FRONT, GL_SPECULAR, Materials[2]);
	glMaterialfv(GL_FRONT, GL_SHININESS, Materials[3]);
	glColorMaterial(GL_FRONT, GL_AMBIENT);
	glEnable(GL_COLOR_MATERIAL);

// 	glScalef(zoomSize, zoomSize, zoomSize);
// 	glTranslatef(NowMoveX * 5, -NowMoveY * 5, 0.0);
// 	glPushMatrix();													// NEW: Prepare Dynamic Transform
// 	glMultMatrixf(Transform.M);										// NEW: Apply Dynamic Transform

	if (mesh && mesh->gl_meshes.size() > 0){
		glBegin(GL_TRIANGLES);
		for (std::map<std::tuple<int, int, int>, GLTriangle>::iterator it = mesh->gl_meshes.begin(); it != mesh->gl_meshes.end(); it++){
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

//	glPopMatrix();
}

void OpenGLWindow::translate(float dx, float dy)
{
	float pdx, pdy;
	backproject(dx, dy, pdx, pdy);

	Eigen::Vector3f lookAt = m_center - m_camera;
	Eigen::Vector3f right = m_up.cross(lookAt).normalized();
	Eigen::Vector3f move = pdx * right + pdy * m_up;
	m_camera += move;
	m_center = m_camera + lookAt;
	update();
}

void OpenGLWindow::rotate(float dx, float dy)
{
	Eigen::Vector3f lookAt = m_center - m_camera;
	float angle = dx / 2 * 3.14159265 / 180;
	Eigen::AngleAxisf aa(angle, m_up.normalized());
	Eigen::Affine3f a = Eigen::Affine3f::Identity();
	a.rotate(aa);
	Eigen::Vector3f tLookAt = a * lookAt;
	m_camera = m_center - tLookAt;

	lookAt = m_center - m_camera;
	Eigen::Vector3f right = m_up.cross(lookAt).normalized();
	angle = dy / 2 * 3.14159265 / 180;
	Eigen::AngleAxisf bb(angle, right);
	a = Eigen::Affine3f::Identity();
	a.rotate(bb);
	tLookAt = a * lookAt;
	m_up = a * m_up;
	m_camera = m_center - tLookAt;

	update();
}

void OpenGLWindow::scale(int dx)
{
	if (dx > 0)
	{
		m_scale += Eigen::Vector3f(0.1, 0.1, 0.1);
	}
	else
	{
		m_scale -= Eigen::Vector3f(0.1, 0.1, 0.1);
		if (m_scale(0) <= 0 || m_scale(1) <= 0 || m_scale(2) <= 0)
		{
			m_scale += Eigen::Vector3f(0.1, 0.1, 0.1);
		}
	}
	update();
}

void OpenGLWindow::backproject(float dx, float dy, float &pdx, float &pdy)
{
	GLint viewport[4];
	glGetIntegerv(GL_VIEWPORT, viewport);
	float w = viewport[2];
	float h = viewport[3];
	float angle = (fov / 2) * 3.14159265 / 180.0;
	float hp = 2 * tan(angle) * zNear;
	float wp = hp * aspect;
	pdx = dx * wp / w;
	pdy = dy * hp / h;

	Eigen::Vector3f lookAt = m_center - m_camera;
	float z = lookAt.norm();
	pdx = pdx * z / zNear;
	pdy = pdy * z / zNear;
}