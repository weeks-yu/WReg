#pragma once

#include "Frame.h"
#include "PairwiseRegister.h"

class GraphManager
{
public:
	GraphManager();
	virtual ~GraphManager();

	virtual bool addNode(Frame* frame, bool isKeyframe = false) = 0;
	virtual Eigen::Matrix4f getTransformation(int k) = 0;
	virtual Eigen::Matrix4f getLastTransformation() = 0;
	virtual Eigen::Matrix4f getLastKeyframeTransformation() = 0;
	virtual void setParameters(void **params) = 0;

	virtual int size();
	virtual vector<Frame*> getGraph();

public:
	vector<Frame*> graph;
//	PairwiseRegister *reg;
};