#ifndef EVALTRANSFORMSTREAM_H
#define EVALTRANSFORMSTREAM_H

extern arguments args;

template <class TRANSFORM> Mat Crop(Mat input, cropBound *cBound, imgBound frameBound, Size size);
template <class TRANSFORM> void evalTransformStream(char *inFileName, char *outFileName, bool prePass);

#include "EvalTransformStream.hpp"

#endif
