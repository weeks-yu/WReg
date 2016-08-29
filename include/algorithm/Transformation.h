#pragma once

#include <limits>
#include <Eigen/Core>
#include <Eigen/Geometry>

Eigen::Matrix3f RotationFromMatrix4f(const Eigen::Matrix4f &matrix);

Eigen::Quaternionf QuaternionFromMatrix4f(const Eigen::Matrix4f &matrix);

Eigen::Matrix3f RotationFromQuaternion(const Eigen::Quaternionf &q);

Eigen::Vector3f EulerAngleFromQuaternion(const Eigen::Quaternionf &q);

Eigen::Vector3f TranslationFromMatrix4f(const Eigen::Matrix4f &matrix);

Eigen::Vector3f YawPitchRollFromMatrix4f(const Eigen::Matrix4f &matrix);

Eigen::Matrix4f transformationFromQuaternionsAndTranslation(const Eigen::Quaternionf &q, const Eigen::Vector3f &t);