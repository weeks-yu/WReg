#include "GraphManager.h"

GraphManager::GraphManager()
{

}

GraphManager::~GraphManager()
{

}

int GraphManager::size()
{
	return graph.size();
}

vector<Frame*> GraphManager::getGraph()
{
	return graph;
}