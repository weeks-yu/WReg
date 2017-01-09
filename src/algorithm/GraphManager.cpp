#include "GraphManager.h"

GraphManager::GraphManager()
{

}

GraphManager::~GraphManager()
{
	for (int i = 0; i < graph.size(); i++)
	{
		delete graph[i];
	}
}

int GraphManager::size()
{
	return graph.size();
}

vector<Frame*> GraphManager::getGraph()
{
	return graph;
}