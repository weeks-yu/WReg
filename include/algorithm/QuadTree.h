#pragma once
#include <vector>

template <typename ValueType>
class QuadTreePoint
{
public:

	double x;
	double y;
	ValueType val;

public:

	QuadTreePoint(double x, double y, ValueType val)
	{
		this->x = x;
		this->y = y;
		this->val = val;
	}

	bool operator == (const QuadTreePoint<ValueType> &other) const
	{
		return (this->x == other.x && this->y == other.y && this->val == other.val);
	}

};

class RectangularRegion
{
public:

	float center_x;
	float center_y;
	float half_width;
	float half_height;

public:

	RectangularRegion()
	{
		center_x = 0.0;
		center_y = 0.0;
		half_width = 0.0;
		half_height = 0.0;
	}

	RectangularRegion(float centerX, float centerY, float width, float height)
	{
		this->center_x = centerX;
		this->center_y = centerY;
		this->half_width = width / 2;
		this->half_height = height / 2;
	}

	bool containsPoint(float x, float y)
	{
		if (x < this->center_x - this->half_width || x > this->center_x + this->half_width ||
			y < this->center_y - this->half_height || y > this->center_y + this->half_height)
		{
			return false;
		}
		return true;
	}

	bool intersectsRegion(float centerX, float centerY, float width, float height)
	{
		if (abs(centerX - this->center_x) < (this->half_width + width / 2) &&
			abs(centerY - this->center_y) < (this->half_height + height / 2))
		{
			return true;
		}
		return false;
	}

	bool intersectsRegion(RectangularRegion other)
	{
		if (abs(other.center_x - this->center_x) < (this->half_width + other.half_width) &&
			abs(other.center_y - this->center_y) < (this->half_height + other.half_height))
		{
			return true;
		}
		return false;
	}
};

template <typename ValueType>
class QuadTree
{
private:

	RectangularRegion region;

	int capacity;

	std::vector<QuadTreePoint<ValueType>> points;

	QuadTree<ValueType>* parent;
	QuadTree<ValueType>* nw;
	QuadTree<ValueType>* ne;
	QuadTree<ValueType>* sw;
	QuadTree<ValueType>* se;

public:

	QuadTree(float centerX, float centerY, float size, QuadTree<ValueType>* parent = nullptr, int capacity = 1)
	{
		this->region.center_x = centerX;
		this->region.center_y = centerY;
		this->region.half_width = size / 2.0;
		this->region.half_height = this->region.half_width;
		this->capacity = capacity;

		this->parent = parent;
		this->nw = nullptr;
		this->ne = nullptr;
		this->sw = nullptr;
		this->se = nullptr;
	}

	QuadTree(const RectangularRegion &region, QuadTree<ValueType>* parent = nullptr, int capacity = 1)
	{
		this->region = region;
		this->capacity = capacity;

		this->parent = parent;
		this->nw = nullptr;
		this->ne = nullptr;
		this->sw = nullptr;
		this->se = nullptr;
	}

	bool insert(float x, float y, ValueType val)
	{
		if (!this->contiansPoint(x, y))
		{
			return false;
		}

		if (this->points.size() < this->capacity)
		{
			this->points.push_back(QuadTreePoint<ValueType>(x, y, val));
			return true;
		}

		if (this->nw == nullptr)
		{
			subdivide();
		}

		if (this->nw->insert(x, y, val)) return true;
		if (this->ne->insert(x, y, val)) return true;
		if (this->sw->insert(x, y, val)) return true;
		if (this->se->insert(x, y, val)) return true;

		// failed for unknwon reasons (this should never happens)
		return false;
	}

	std::vector<ValueType> queryRange(RectangularRegion region)
	{
		std::vector<ValueType> result;

		if (!this->intersectsRegion(region))
		{
			return result;
		}

		for (int i = 0; i < this->points.size(); i++)
		{
			if (region.containsPoint(this->points[i].x, this->points[i].y))
			{
				result.push_back(this->points[i].val);
			}
		}

		if (this->nw != nullptr)
		{
			this->append(result, this->nw->queryRange(region));
			this->append(result, this->ne->queryRange(region));
			this->append(result, this->sw->queryRange(region));
			this->append(result, this->se->queryRange(region));
		}

		return result;
	}

	QuadTree<ValueType>* getParent()
	{
		return this->parent;
	}

	QuadTree<ValueType>* findQuad(const QuadTreePoint<ValueType> &key)
	{
		QuadTree<ValueType>* result = nullptr;

		for (int i = 0; i < this->points.size(); i++)
		{
			if (this->points[i] == key)
			{
				result = this;
				break;
			}
		}

		if (this->nw != nullptr)
		{
			if (result == nullptr) result = this->nw->findQuad(key);
			if (result == nullptr) result = this->ne->findQuad(key);
			if (result == nullptr) result = this->sw->findQuad(key);
			if (result == nullptr) result = this->se->findQuad(key);
		}
		return result;
	}

	bool find(const QuadTreePoint<ValueType> &key)
	{
		QuadTree<ValueType>* result = this->findQuad(key);

		return (result != nullptr);
	}

	int getCount()
	{
		return this->points.size();
	}

	bool isEmpty()
	{
		return (this->nw == nullptr && this->points.size() == 0);
	}

	bool erase(const QuadTreePoint<ValueType> &key)
	{
		bool exists = false;
		std::vector<QuadTreePoint<ValueType>>::iterator it = this->points.begin();
		for (; it != this->points.end(); it++)
		{
			if (*it == key)
			{
				exists = true;
				break;
			}
		}

		if (exists)
		{
			this->points.erase(it);
			if (this->parent != nullptr) this->parent->balance();
			return true;
		}
		else
		{
			QuadTree<ValueType>* quad = this->findQuad(key);
			if (quad == nullptr) return false;
			return quad->erase(key);
		}
	}

	bool update(float old_x, float old_y, ValueType val, float new_x, float new_y)
	{
		for (int i = 0; i < this->points.size(); i++)
		{
			if (this->points[i].x == old_x && this->points[i].y == old_y && this->points[i].val == val)
			{
				if (this->contiansPoint(new_x, new_y))
				{
					this->points[i].x = new_x;
					this->points[i].y = new_y;
					return true;
				}
				else
				{
					this->erase(QuadTreePoint<ValueType>(old_x, old_y, val));
					return false;
				}
			}
		}

		QuadTree<ValueType>* quad = this->findQuad(QuadTreePoint<ValueType>(old_x, old_y, val));
		if (quad == nullptr)
		{
			return false;
		}

		if (!quad->update(old_x, old_y, val, new_x, new_y))
		{
			if (!this->contiansPoint(new_x, new_y))
			{
				return false;
			}
			this->insert(new_x, new_y, val);
		}
		return true;
	}

	std::vector<ValueType> queryRange(float centerX, float centerY, float width, float height)
	{
		return this->queryRange(RectangularRegion(centerX, centerY, width, height));
	}

	bool contiansPoint(float x, float y)
	{
		return this->region.containsPoint(x, y);
	}

	bool intersectsRegion(float centerX, float centerY, float width, float height)
	{
		return this->region.intersectsRegion(centerX, centerY, width, height);
	}

	bool intersectsRegion(RectangularRegion other)
	{
		return this->region.intersectsRegion(other);
	}

private:

	void subdivide()
	{
		float sub_half_width = this->region.half_width / 2.0;
		float sub_half_height = this->region.half_height / 2.0;

		RectangularRegion region;
		region.half_width = sub_half_width;
		region.half_height = sub_half_height;

		region.center_x = this->region.center_x - sub_half_width;
		region.center_y = this->region.center_y + sub_half_height;
		this->nw = new QuadTree<ValueType>(region, this);

		region.center_x = this->region.center_x + sub_half_width;
		region.center_y = this->region.center_y + sub_half_height;
		this->ne = new QuadTree<ValueType>(region, this);

		region.center_x = this->region.center_x - sub_half_width;
		region.center_y = this->region.center_y - sub_half_height;
		this->sw = new QuadTree<ValueType>(region, this);

		region.center_x = this->region.center_x + sub_half_width;
		region.center_y = this->region.center_y - sub_half_height;
		this->se = new QuadTree<ValueType>(region, this);
	}

	void balance()
	{
		if (this->nw == nullptr) return;
				
		if (!this->nw->isEmpty() || !this->ne->isEmpty() || !this->sw->isEmpty() || !this->se->isEmpty())
			return;

		delete this->nw;
		delete this->ne;
		delete this->sw;
		delete this->se;

		this->nw = nullptr;
		this->ne = nullptr;
		this->sw = nullptr;
		this->se = nullptr;
	}

	inline void append(std::vector<ValueType> &result, const std::vector<ValueType> &other)
	{
		for (std::vector<ValueType>::const_iterator it = other.begin(); it != other.end(); it++)
		{
			result.push_back(*it);
		}
	}
};