#pragma once
#include "QuadTree.h"
#include "Frame.h"

struct ActiveWindow
{
	QuadTree<int>* key_frames;
	vector<int> active_frames;
	RectangularRegion region;
	Feature* feature_pool;

	ActiveWindow();
	void build(float center_x, float center_y, float size, int capacity = 1);
	void move(const vector<Frame*> &graph, const RectangularRegion &region);
	bool insert(float x, float y, int frame_index);
};