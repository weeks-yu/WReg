#include "IcpcudaRegister.h"
#include "Config.h"

IcpcudaRegister::IcpcudaRegister()
{
	this->icpcuda = nullptr;
	this->threads = Config::instance()->get<int>("icpcuda_threads");
	this->blocks = Config::instance()->get<int>("icpcuda_blocks");
	this->depthCutOff = 20.f;
}

IcpcudaRegister::IcpcudaRegister(ICPOdometry *icpcuda, int threads, int blocks, float depthCutOff)
{
	this->icpcuda = icpcuda;
	this->threads = threads;
	this->blocks = blocks;
	this->depthCutOff = depthCutOff;
}

IcpcudaRegister::~IcpcudaRegister()
{

}

bool IcpcudaRegister::getTransformation(void *prev, void *now, Eigen::Matrix4f &tran)
{
	icpcuda->initICPModel((unsigned short *)prev, 20.0f, tran);
	icpcuda->initICP((unsigned short *)now, 20.0f);

	Eigen::Vector3f t = tran.topRightCorner(3, 1);
	Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rot = tran.topLeftCorner(3, 3);

	Eigen::Matrix4f estimated_tran = Eigen::Matrix4f::Identity();
	Eigen::Vector3f estimated_t = estimated_tran.topRightCorner(3, 1);
	Eigen::Matrix<float, 3, 3, Eigen::RowMajor> estimated_rot = estimated_tran.topLeftCorner(3, 3);

	icpcuda->getIncrementalTransformation(t, rot, estimated_t, estimated_rot, threads, blocks);

	tran.topLeftCorner(3, 3) = rot;
	tran.topRightCorner(3, 1) = t;

	return true;
}

float IcpcudaRegister::getCorrespondencePercent(void *last, void *now, Eigen::Matrix4f &estimatedTran)
{
	throw std::exception("Exception : not implemented.");
}

void IcpcudaRegister::setParameters(void **parameters)
{
	this->icpcuda = static_cast<ICPOdometry *>(parameters[0]);
	this->threads = *static_cast<int *>(parameters[1]);
	this->blocks = *static_cast<int *>(parameters[2]);
	this->depthCutOff = *static_cast<float *>(parameters[3]);
}