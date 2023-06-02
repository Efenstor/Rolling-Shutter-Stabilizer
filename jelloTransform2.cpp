#define _USE_MATH_DEFINES

#include <iostream> // for standard I/O
#include <string>   // for strings
#include <iomanip>  // for controlling float print precision
#include <sstream>  // string to number conversion
#include <time.h>
#include <numeric>
#include <math.h>
#include <thread>

#include "svd.h"
#include "structures.h"
#include "settings.h"
#include "coreFuncs.h"
#include "jelloTransform2.h"
#include "nullTransform.h"

#include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include "opencv2/imgproc/imgproc_c.h"

extern arguments args;

JelloTransform2::JelloTransform2(TransformationMem *tm)
{
    AssignShiftMem(tm);
}

JelloTransform2::JelloTransform2(Mat img1, Mat img2, int index0, int index1, TransformationMem *tm)
{
    AssignShiftMem(tm);
    frameBound = {0, img1.cols, 0, img1.rows};

    CalcJelloTransform(img1, img2);

    #ifdef SHFITS_FILENAME

        evalTransforms(index0, index1, (char*)SHFITS_FILENAME);

    #endif
}

void JelloTransform2::AssignShiftMem(TransformationMem *tm)
{
    params = tm->params;
    shiftsX = tm->shiftsX;
    shiftsY = tm->shiftsY;
}

void JelloTransform2::CalcJelloTransform(Mat img1, Mat img2){

    Size img_sz = img1.size();
    Mat imgC(img_sz,1);

    int win_size = 15;
    int maxCorners = 1000;

    std::vector<cv::Point2f> corners1 = extractCornersToTrack(img1);
    corners1.reserve(maxCorners);
    std::vector<cv::Point2f> corners2;
    corners2.reserve(maxCorners);

    Size pyr_sz = Size( img_sz.width+8, img_sz.height/3 );

    std::vector<uchar> features_found;
    features_found.reserve(maxCorners);
    std::vector<float> feature_errors;
    feature_errors.reserve(maxCorners);

    calcOpticalFlowPyrLK( img1, img2, corners1, corners2, features_found, feature_errors ,
        Size( args.winSize, args.winSize ), args.maxLevel,
        TermCriteria( TermCriteria::COUNT+TermCriteria::EPS, args.iter, args.epsilon ), 0, args.eigThr );

    #ifdef SHFITS_FILENAME
        char outputFilename[100];
        sprintf(outputFilename, "data/pts%d.txt", processedFrameCount++);
        FILE* fp = fopen(outputFilename, "w");
        if(fp == NULL){

        } else {

            for(int i=0;i<(int)features_found.size();i++){
                fprintf(fp, "%f\t%f\t%f\t%f\n", corners1[i].x, corners1[i].y, corners2[i].x, corners2[i].y);
            }
            fclose(fp);
        }
    #endif

    #if(1)
    float rowDistance = imgHeight / (SVD_ROWS-1);
    for(int row = 0;row<SVD_ROWS;row++)
    {
        float rowPosition = imgHeight *row / (SVD_ROWS-1);

        vector<float> weights((int)features_found.size());

        for(int i=0;i<(int)features_found.size();i++)
        {
            float d = corners1[i].y - rowPosition;
            if(fabs(d) > corners1[i].y - rowPosition && REMOVE_GAUSSIAN_WEIGHT_TAILS){
                weights[i] = 0;
            } else {
                weights[i] = SVD_WEIGHT_FUNC3(d);
            }
        }

        //Transformation t = weightedSvd(corners1, corners2, features_found.size(), weights);
        //Transformation t = prunedWeightedSvd(corners1, corners2, features_found.size(), weights);
        Transformation t = WelschFitWeighted(corners1, corners2, features_found.size(), weights);
        jelloTransforms.push_back(t);
    }

    #else

    for(int row = 0;row<SVD_ROWS;row++)
    {
        float rowPosition = imgHeight *row / (SVD_ROWS-1);
        float halfRowSpacing =  imgHeight / (SVD_ROWS-1);

        vector<Point2f> pts1;
        vector<Point2f> pts2;

        //printf("row: %d   from: %f - %f\n", row, rowPosition-halfRowSpacing, rowPosition + halfRowSpacing);
        for(int i=0;i<(int)features_found.size();i++)
        {
            if(((corners1[i].y - rowPosition) > -halfRowSpacing || row == 0) && ((corners1[i].y - rowPosition) < halfRowSpacing || row == (SVD_ROWS-1))){
                pts1.push_back(corners1[i]);
                pts2.push_back(corners2[i]);
                //printf("%f\n", corners1[i].y);
            }
        }

        Transformation t = RansacNonWeightedSvd(pts1, pts2, pts1.size());
        jelloTransforms.push_back(t);
    }

    #endif
}

void JelloTransform2::CreateAbsoluteTransformThread(TransformationMem *prevMem, TransformationMem *newMem, threadParams tExtent, float decayX, float decayY)
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

void JelloTransform2::CreateAbsoluteTransform(TransformationMem *prevMem, TransformationMem *newMem)
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
        std::thread newThr(&JelloTransform2::CreateAbsoluteTransformThread, this, prevMem, newMem,
            tExtent.at(t), decayX, decayY);
        threads.push_back(move(newThr));
    }

    // Join threads
    for(int t=0; t<tNum; t++)
    {
        threads.at(t).join();
    }
}

/*void JelloTransform2::CreateAbsoluteTransform(TransformationMem *prevMem, TransformationMem *newMem)
{
    for(int row=0;row<imgHeight;row++){
        for(int col=0;col<imgWidth;col++){
            float x2, y2;

            float x = col + prevTransform.shiftsX[row][col];
            float y = row + prevTransform.shiftsY[row][col];

            TransformPoint(x, y, x2, y2);

            shiftsX[row][col] = prevTransform.shiftsX[row][col] * JELLO_DECAY - x2 + x;
            shiftsY[row][col] = prevTransform.shiftsY[row][col] * JELLO_DECAY - y2 + y;
        }
    }
}*/

void JelloTransform2::TransformPoint(float x, float y, float &x2, float &y2){
    float svdIndex = y * ((float)SVD_ROWS) / ((float)imgHeight);
            int svdIndex0 = (int) floor(svdIndex);
            int svdIndex1 = (int) ceil(svdIndex);

            if(svdIndex0 < 0)
                svdIndex0 = 0;
            else if(svdIndex0 >= SVD_ROWS)
                svdIndex0 = SVD_ROWS-1;

            if(svdIndex1 >= SVD_ROWS)
                svdIndex1 = SVD_ROWS-1;
            else if(svdIndex1 < 0)
                svdIndex1 = 0;

            float w = fmod(svdIndex, 1.F);
            if(w < 0)  w = 0.0;
            else if(w > 1) w = 1.0;

            float x2_0, y2_0, x2_1, y2_1;

            GenericTransformPoint(jelloTransforms[svdIndex0], x, y, x2_0, y2_0);
            GenericTransformPoint(jelloTransforms[svdIndex1], x, y, x2_1, y2_1);

            x2 = x2_1 * w + x2_0 * (1-w);
            y2 = y2_1 * w + y2_0 * (1-w);
}

void JelloTransform2::TransformPointAbs(float x, float y, float &x2, float &y2){
    int ix = round(x);
    int iy = round(y);

    x2 = x - shiftsX[iy][ix];
    y2 = y - shiftsY[iy][ix];

}
