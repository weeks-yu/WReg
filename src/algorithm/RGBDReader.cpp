#include "RGBDReader.h"

RGBDReader::RGBDReader() : intrDepth(), max_depth(1.0f)
{

}

RGBDReader::~RGBDReader()
{

}

void RGBDReader::setIntrinsicColor(const Intrinsic &intr)
{
	intrColor = intr;
}

void RGBDReader::setIntrinsicDepth(const Intrinsic &intr)
{
	intrDepth = intr;
}