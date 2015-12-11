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

Eigen::Quaternionf QuaternionsFromMatrix4f(const Eigen::Matrix4f &matrix)
{
	Eigen::Affine3f a(matrix);
	Eigen::Quaternionf q(a.rotation());
	return q;
}

Eigen::Matrix3f RotationFromQuaternions(const Eigen::Quaternionf &q)
{
	return q.matrix();
}

Eigen::Vector3f TranslationFromMatrix4f(const Eigen::Matrix4f &matrix)
{
	Eigen::Affine3f a(matrix);

	return a.translation();
}

Eigen::Vector3f EulerAngleFromQuaternions(const Eigen::Quaternionf &q)
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

Eigen::Vector3f EulerAngleFromMatrix4f(const Eigen::Matrix4f &matrix)
{
	Eigen::Vector3f e;
	e(0) = atan2(matrix(2, 1), matrix(2, 2));
	e(1) = atan2(-matrix(2, 0), sqrt(matrix(2, 1) * matrix(2, 1) + matrix(2, 2) * matrix(2, 2)));
	e(2) = atan2(matrix(1, 0), matrix(0, 0));

	return e;
}

bool IsTransformationBigEnough(const Eigen::Matrix4f &matrix)
{
	Eigen::Vector3f t = TranslationFromMatrix4f(matrix);
	Eigen::Vector3f e = EulerAngleFromMatrix4f(matrix);

	e *= 180.0 / M_PI;
	double max_angle = std::max(e(0), std::max(e(1), e(2)));

	return (t.norm() > Config::instance()->get<double>("min_translation_meter")
		|| max_angle > Config::instance()->get<double>("min_rotation_degree"));
}

Transformation3 eigenToHogman(const Eigen::Matrix4f &eigen_matrix) {

	Eigen::Affine3f eigen_transform(eigen_matrix);
	Eigen::Quaternionf eigen_quat(eigen_transform.rotation());
	Vector3 translation(eigen_matrix(0, 3), eigen_matrix(1, 3), eigen_matrix(2, 3));
	Quaternion rotation(eigen_quat.x(), eigen_quat.y(), eigen_quat.z(),
		eigen_quat.w());
	Transformation3 result(translation, rotation);

	return result;
}

Eigen::Matrix4f hogmanToEigen(const Transformation3 &hogman_matrix)
{
	Vector3 translation = hogman_matrix.translation();
	Quaternion rotation = hogman_matrix.rotation();
	Eigen::Quaternionf eigen_quaternion(rotation.w(), rotation.x(), rotation.y(), rotation.z());
	Eigen::Vector3f eigen_translation(translation.x(), translation.y(), translation.z());

	Eigen::Affine3f eigen_tran = Eigen::Affine3f::Identity();
	eigen_tran.translate(eigen_translation);
	eigen_tran.rotate(eigen_quaternion);
	return eigen_tran.matrix();
}

Eigen::Matrix4f transformationFromQuaternionsAndTranslation(const Eigen::Quaternionf &q, const Eigen::Vector3f &t)
{
	Eigen::Affine3f a = Eigen::Affine3f::Identity();
	a.translate(t);
	a.rotate(q);
	return a.matrix();
}