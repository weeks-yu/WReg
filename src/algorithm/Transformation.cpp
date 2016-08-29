#include "Transformation.h"
#define _USE_MATH_DEFINES
#include <math.h>
#include <algorithm>
#include "Config.h"

Eigen::Matrix3f RotationFromMatrix4f(const Eigen::Matrix4f &matrix)
{
	Eigen::Affine3f a(matrix);
	return a.rotation();
}

Eigen::Quaternionf QuaternionFromMatrix4f(const Eigen::Matrix4f &matrix)
{
	Eigen::Affine3f a(matrix);
	Eigen::Quaternionf q(a.rotation());
	return q;
}

Eigen::Matrix3f RotationFromQuaternion(const Eigen::Quaternionf &q)
{
	return q.matrix();
}

Eigen::Vector3f TranslationFromMatrix4f(const Eigen::Matrix4f &matrix)
{
	Eigen::Affine3f a(matrix);

	return a.translation();
}

Eigen::Vector3f EulerAngleFromQuaternion(const Eigen::Quaternionf &q)
{
	Eigen::Vector3f e;
	e(0) = atan2(2 * (q.w() * q.z() + q.x() * q.y()) , 1 - 2 * (q.z() * q.z() + q.x() * q.x()));
	float temp = 2 * (q.w() * q.x() - q.y() * q.z());
	if (temp < -1.0f) temp = -1.0f;
	if (temp > 1.0f) temp = 1.0f;
	e(1) = asin(temp);
	e(2) = atan2(2 * (q.w() * q.y() + q.z() * q.x()) , 1 - 2 * (q.x() * q.x() + q.y() * q.y()));

	return e;
}

Eigen::Vector3f YawPitchRollFromMatrix4f(const Eigen::Matrix4f &matrix)
{ 
	Eigen::Affine3f a(matrix);
	return a.rotation().eulerAngles(2, 1, 0);
}

Eigen::Matrix4f transformationFromQuaternionsAndTranslation(const Eigen::Quaternionf &q, const Eigen::Vector3f &t)
{
	Eigen::Affine3f a = Eigen::Affine3f::Identity();
	a.translate(t);
	a.rotate(q);
	return a.matrix();
}