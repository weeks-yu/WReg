#include "PointCloudModel.h"

PointCloudModel::PointCloudModel()
{
	downsample_rate = 0.01;
	model = PointCloudPtr(new PointCloudT());
	temp = PointCloudPtr(new PointCloudT());
}

PointCloudModel::PointCloudModel(float dr)
{
	downsample_rate = dr;
	model = PointCloudPtr(new PointCloudT());
	temp = PointCloudPtr(new PointCloudT());
}

PointCloudModel::~PointCloudModel()
{

}

void PointCloudModel::dataFusion(PointCloudPtr cloud)
{
	*temp += *cloud;
	if (temp->size() >= 10000000)
	{
		PointCloudPtr down = DownSamplingByVoxelGrid(temp, downsample_rate, downsample_rate, downsample_rate);
		if (model->size() >= std::numeric_limits<int>::max() - down->size())
		{
			model = DownSamplingByVoxelGrid(model, downsample_rate, downsample_rate, downsample_rate);
		}
		*model += *down;
		temp->clear();
	}
}

PointCloudPtr PointCloudModel::getModel()
{
	if (temp->size() > 0)
	{
		PointCloudPtr down = temp;
		if (model->size() >= std::numeric_limits<int>::max() - temp->size())
		{
			down = DownSamplingByVoxelGrid(temp, downsample_rate, downsample_rate, downsample_rate);
			if (model->size() >= std::numeric_limits<int>::max() - down->size())
			{
				model = DownSamplingByVoxelGrid(model, downsample_rate, downsample_rate, downsample_rate);
			}
		}
		*model += *down;
	}
	PointCloudPtr ret(new PointCloudT());
	ret = DownSamplingByVoxelGrid(model, downsample_rate, downsample_rate, downsample_rate);
	return ret;
}

void PointCloudModel::setDownsampleRate(float dr)
{
	downsample_rate = dr;
}