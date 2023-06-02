#include <iostream> // for standard I/O
#include <string>   // for strings
#include <iomanip>  // for controlling float print precision
#include <sstream>  // string to number conversion
#include <time.h>
#include <numeric>
#include <thread>

#include "svd.h"
#include "structures.h"
#include "settings.h"
#include "coreFuncs.h"
#include "FullFrameTransform2.h"
#include "nullTransform.h"

#include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include "opencv2/imgproc/imgproc_c.h"

#define LAMBDA_INCREASE 2
#define LAMBDA_DECREASE 2
#define MAX_LAMBDA      1000

extern arguments args;

FullFrameTransform2::FullFrameTransform2(TransformationMem *tm)
{
    AssignShiftMem(tm);
}

FullFrameTransform2::FullFrameTransform2(Mat img1, Mat img2, int index0, int index1, TransformationMem *tm)
{
    AssignShiftMem(tm);
    frameBound = {0, img1.cols, 0, img1.rows};

    getWholeFrameTransform(img1, img2);

    #ifdef SHFITS_FILENAME

        evalTransforms(index0, index1, (char*)SHFITS_FILENAME);

    #endif
}

void FullFrameTransform2::AssignShiftMem(TransformationMem *tm)
{
    params = tm->params;
    shiftsX = tm->shiftsX;
    shiftsY = tm->shiftsY;
}

void FullFrameTransform2::getWholeFrameTransform(Mat img1, Mat img2){

    vector<Point2f> corners1, corners2;

    int length = GetPointsToTrack(img1, img2, corners1, corners2);

    memset(params, 0, 3*sizeof(float));

    float *updates = new float[3];

    float lambda = 0.01;
    for(int i=0;i<NUM_STEPS;i++){
        float w = pow(10, log10(START_W) + ((float)i / ((float)NUM_STEPS-1)) * (log10(END_W) - log10(START_W)));
        LMIterationWelsch(corners1, corners2, length, params, updates, lambda, w);

        for(int i=0;i<3;i++)
            params[i] -= updates[i];
    }
}

void FullFrameTransform2::CreateAbsoluteTransformThread(TransformationMem *prevMem, TransformationMem *newMem,
    threadParams tExtent, float decayX, float decayY)
{
    for(int row=tExtent.from;row<tExtent.to;row++)
    {
        for(int col=0;col<imgWidth;col++)
        {
            float x2, y2;

            // Transform
            float x = col + prevMem->shiftsX[row][col];
            float y = row + prevMem->shiftsY[row][col];
            TransformPoint(x, y, x2, y2);

            // Covert to shifts with inertia
            newMem->shiftsX[row][col] = prevMem->shiftsX[row][col] * decayX - x2 + x;
            newMem->shiftsY[row][col] = prevMem->shiftsY[row][col] * decayY - y2 + y;
        }
    }
}

void FullFrameTransform2::CreateAbsoluteTransform(TransformationMem *prevMem, TransformationMem *newMem)
{
    vector<threadParams> tExtent;

    // Dynamic jello decay
    int cx = imgWidth/2;
    int cy = imgHeight/2;
    float x = cx + prevMem->shiftsX[cx][cy];
    float y = cy + prevMem->shiftsY[cx][cy];
    float x2, y2;
    TransformPoint(x, y, x2, y2);
    float maxShift = fmax(imgWidth * args.djdShift, imgHeight * args.djdShift);
    float csX = fmin( abs(x2-cx) / maxShift, 1.0 );
    float csY = fmin( abs(x2-cx) / maxShift, 1.0 );
    float decayX = (1 - pow(csX, args.djdLinear)) * args.djdAmount;
    float decayY = (1 - pow(csY, args.djdLinear)) * args.djdAmount;
    //printf("csX=%f csY=%f decayX=%f decayY=%f\n", csX, csY, decayX, decayY);

    // Prepare threads
    int tNum = args.threads;
    double rowsPerThread = imgHeight/tNum;
    for(int t=0; t<tNum; t++)
    {
        threadParams tp;

        tp.from = lround(t*rowsPerThread);
        if(t<tNum-1) tp.to = lround((t+1)*rowsPerThread);
        else tp.to = imgHeight;

        tExtent.push_back(tp);
    }

    // Create threads
    vector<std::thread> threads;
    for(int t=0; t<tNum; t++)
    {
        std::thread newThr(&FullFrameTransform2::CreateAbsoluteTransformThread, this, prevMem, newMem,
            tExtent.at(t), decayX, decayY);
        threads.push_back(move(newThr));
    }

    // Join threads
    for(int t=0; t<tNum; t++)
    {
        threads.at(t).join();
    }
}

float FullFrameTransform2::getFullModelCostWelsch(vector<Point2f> corners1, vector<Point2f> corners2, float* params, float w){
    float result = 0;

    float r = params[0];
    float tx = params[1];
    float ty = params[2];

    for(int i=0;i<(int)corners1.size();i++){
        float x, y, x2, y2;
        x = corners1[i].x;
        y = corners1[i].y;
        x2 = corners2[i].x;
        y2 = corners2[i].y;

        float x2Pred = x*cos(r) - y*sin(r) + tx;
        float y2Pred = x*sin(r) + y*cos(r) + ty;

        float d = (x2-x2Pred) * (x2-x2Pred) + (y2 - y2Pred) * (y2-y2Pred);
        float e = w*w*(1.0 - exp(-d/(w*w)));
        result += e;
    }

    result /= (float)corners1.size();
    result = sqrt(result);

    return result;
}

void FullFrameTransform2::LMIterationWelsch(vector<Point2f> corners1, vector<Point2f> corners2, int length, float* params, float* &updates, float &lambda, float w){

    arma::Col<float> fVector(3);
    arma::Mat<float> jacob(3,3);
    arma::Col<float> update;
    fVector.zeros();
    jacob.zeros();

    float r0 = params[0];
    float dx0 = params[1];
    float dy0 = params[2];

    float newParams[3];
    float jacobianDiagonals[3];

    float startCost = getFullModelCostWelsch(corners1, corners2, params, w);

    for(int i=0;i<length;i++){
        float x1 = corners1[i].x;
        float y1 = corners1[i].y;
        float x2 = corners2[i].x;
        float y2 = corners2[i].y;

        jacob(0, 0) +=  (2*pow(x1*sin(r0) + y1*cos(r0), 2) - 2*(x1*sin(r0) + y1*cos(r0))*(dy0 + x1*sin(r0) + y1*cos(r0) - y2) + 2*pow(x1*cos(r0) - y1*sin(r0), 2) - 2*(x1*cos(r0) - y1*sin(r0))*(dx0 + x1*cos(r0) - x2 - y1*sin(r0)) - 4*pow((x1*sin(r0) + y1*cos(r0))*(dx0 + x1*cos(r0) - x2 - y1*sin(r0)) - (x1*cos(r0) - y1*sin(r0))*(dy0 + x1*sin(r0) + y1*cos(r0) - y2), 2)/(w*w))*exp(-(pow(dx0 + x1*cos(r0) - x2 - y1*sin(r0), 2) + pow(dy0 + x1*sin(r0) + y1*cos(r0) - y2, 2))/(w*w));
        jacob(1, 0) +=  (-2*x1*sin(r0) - 2*y1*cos(r0) + 4*((x1*sin(r0) + y1*cos(r0))*(dx0 + x1*cos(r0) - x2 - y1*sin(r0)) - (x1*cos(r0) - y1*sin(r0))*(dy0 + x1*sin(r0) + y1*cos(r0) - y2))*(dx0 + x1*cos(r0) - x2 - y1*sin(r0))/(w*w))*exp(-(pow(dx0 + x1*cos(r0) - x2 - y1*sin(r0), 2) + pow(dy0 + x1*sin(r0) + y1*cos(r0) - y2, 2))/(w*w));
        jacob(1, 1) +=  (2 - 4*pow(dx0 + x1*cos(r0) - x2 - y1*sin(r0), 2)/(w*w))*exp(-(pow(dx0 + x1*cos(r0) - x2 - y1*sin(r0), 2) + pow(dy0 + x1*sin(r0) + y1*cos(r0) - y2, 2))/(w*w));
        jacob(2, 0) +=  (2*x1*cos(r0) - 2*y1*sin(r0) + 4*((x1*sin(r0) + y1*cos(r0))*(dx0 + x1*cos(r0) - x2 - y1*sin(r0)) - (x1*cos(r0) - y1*sin(r0))*(dy0 + x1*sin(r0) + y1*cos(r0) - y2))*(dy0 + x1*sin(r0) + y1*cos(r0) - y2)/(w*w))*exp(-(pow(dx0 + x1*cos(r0) - x2 - y1*sin(r0), 2) + pow(dy0 + x1*sin(r0) + y1*cos(r0) - y2, 2))/(w*w));
        jacob(2, 1) +=  (dx0 + x1*cos(r0) - x2 - y1*sin(r0))*(dy0 + x1*sin(r0) + y1*cos(r0) - y2)*exp(-(pow(dx0 + x1*cos(r0) - x2 - y1*sin(r0), 2) + pow(dy0 + x1*sin(r0) + y1*cos(r0) - y2, 2))/(w*w))/(w*w);
        jacob(2, 2) +=  (2 - 4*pow(dy0 + x1*sin(r0) + y1*cos(r0) - y2, 2)/(w*w))*exp(-(pow(dx0 + x1*cos(r0) - x2 - y1*sin(r0), 2) + pow(dy0 + x1*sin(r0) + y1*cos(r0) - y2, 2))/(w*w));

        fVector(0) +=  (-2*(x1*sin(r0) + y1*cos(r0))*(dx0 + x1*cos(r0) - x2 - y1*sin(r0)) + 2*(x1*cos(r0) - y1*sin(r0))*(dy0 + x1*sin(r0) + y1*cos(r0) - y2))*exp(-(pow(dx0 + x1*cos(r0) - x2 - y1*sin(r0), 2) + pow(dy0 + x1*sin(r0) + y1*cos(r0) - y2, 2))/(w*w));
        fVector(1) +=  (2*dx0 + 2*x1*cos(r0) - 2*x2 - 2*y1*sin(r0))*exp(-(pow(dx0 + x1*cos(r0) - x2 - y1*sin(r0), 2) + pow(dy0 + x1*sin(r0) + y1*cos(r0) - y2, 2))/(w*w));
        fVector(2) +=  (2*dy0 + 2*x1*sin(r0) + 2*y1*cos(r0) - 2*y2)*exp(-(pow(dx0 + x1*cos(r0) - x2 - y1*sin(r0), 2) + pow(dy0 + x1*sin(r0) + y1*cos(r0) - y2, 2))/(w*w));
    }

    jacob(2, 1) *= -4.000000;


    jacob(0, 1) = jacob(1, 0);
    jacob(0, 2) = jacob(2, 0);
    jacob(1, 2) = jacob(2, 1);


    for(int i=0;i<3;i++)
        jacobianDiagonals[i] = jacob(i,i);

    while(1){

        for(int i=0;i<3;i++){
            jacob(i, i) = jacobianDiagonals[i] * (lambda + 1);
        }

        update = jacob.i() * fVector;

        for(int i=0;i<3;i++){
            newParams[i] = params[i] - update(i);
        }

        float newCost = getFullModelCostWelsch(corners1, corners2, newParams, w);
        //printf("startcost: %f   newCost: %f\n", startCost, newCost);

        if(isnan(newCost)){
            if(args.warnings) printf("new cost is NAN in model2LMIterationWelsch()");
            lambda *= LAMBDA_INCREASE;
            update(0) = 0;
            update(1) = 0;
            update(2) = 0;
            break;
        } else if(newCost > startCost){
            lambda *= LAMBDA_INCREASE;
            if(lambda > MAX_LAMBDA){
                update(0) = 0;
                update(1) = 0;
                update(2) = 0;
                break;
            }
        } else {
            lambda /= LAMBDA_DECREASE;
            break;
        }

    }

    for(int i=0;i<3;i++){
        updates[i] = update(i);
    }
}

void FullFrameTransform2::TransformPoint(float x, float y, float &x2, float &y2){

    float r = params[0];
    float tx = params[1];
    float ty = params[2];

    x2 = x*cos(r) - y*sin(r) + tx;
    y2 = x*sin(r) + y*cos(r) + ty;
}

void FullFrameTransform2::TransformPointAbs(float x, float y, float &x2, float &y2){

    int ix = round(x);
    int iy = round(y);

    x2 = x - shiftsX[iy][ix];
    y2 = y - shiftsY[iy][ix];
}
