#include "ActiveWindow.h"

ActiveWindow::ActiveWindow()
{
	key_frames = nullptr;
	feature_pool = nullptr;
	this->region.center_x = 0.0;
	this->region.center_y = 0.0;
	this->region.half_width = Config::instance()->get<float>("active_window_size") / 2.0;
	this->region.half_height = this->region.half_width;
}

void ActiveWindow::build(float center_x, float center_y, float size, int capacity /* = 1 */)
{
	if (key_frames != nullptr)
	{
		delete key_frames;
	}
	key_frames = new QuadTree<int>(center_x, center_y, size, nullptr, capacity);
}

void ActiveWindow::move(const vector<Frame*> &graph, const RectangularRegion &r)
{
	region = r;
	active_frames = key_frames->queryRange(region);

	if (feature_pool != nullptr)
	{
		delete this->feature_pool;
	}
	feature_pool = new Feature(true);
	feature_pool->type = graph[0]->f->type;
	for (int i = 0; i < active_frames.size(); i++)
	{
		Frame *now_f = graph[active_frames[i]];
		for (int j = 0; j < now_f->f->feature_pts_3d.size(); j++)
		{
			feature_pool->feature_pts.push_back(now_f->f->feature_pts[j]);
			feature_pool->feature_pts_3d.push_back(now_f->f->feature_pts_3d[j]);
			feature_pool->feature_descriptors.push_back(now_f->f->feature_descriptors.row(j));
			feature_pool->feature_frame_index.push_back(i);
		}
	}
	feature_pool->buildFlannIndex();
}

bool ActiveWindow::insert(float x, float y, int frame_index)
{
	if (!key_frames->insert(x, y, frame_index)) return false;
	return true;
}