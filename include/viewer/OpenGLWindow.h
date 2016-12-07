#pragma once

#include <QOpenGLWindow>
#include <QOpenGLFunctions>
#include <QMouseEvent>
#include <QWheelEvent>
#include <Eigen/Dense>
#include "TriangleMesh.h"

class OpenGLWindow : public QOpenGLWindow, protected QOpenGLFunctions
{
public:
	OpenGLWindow();
	~OpenGLWindow();

	QSize sizeHint()
	{
		return QSize(400, 400);
	}

	void setMesh(GLMesh *mesh);

protected:
	void initializeGL();
	void paintGL();
	void resizeGL(int w, int h);

	void mousePressEvent(QMouseEvent *event);
	void mouseMoveEvent(QMouseEvent *event);
	void wheelEvent(QWheelEvent *event);

private:
	void drawMesh();
	void translate(float dx, float dy);
	void rotate(float dx, float dy);
	void scale(int dx);
	void backproject(float dx, float dy, float &pdx, float &pdy);

public:
	float Materials[4][4];

private:
	GLMesh *mesh;

	QPoint lastPos;
	
	Eigen::Vector3f m_camera;
	Eigen::Vector3f m_center;
	Eigen::Vector3f m_up;
	Eigen::Vector4f m_ref;
	Eigen::Vector3f m_scale;

	float fov;
	float aspect;
	float zNear;
	float zFar;

	bool openglInit;
};