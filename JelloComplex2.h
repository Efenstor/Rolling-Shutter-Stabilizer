#ifndef JELLO_COMPLEX_2
#define JELLO_COMPLEX_2

// Video Image PSNR and SSIM
#include <iostream> // for standard I/O
#include <string>   // for strings
#include <iomanip>  // for controlling float print precision
#include <sstream>  // string to number conversion
#include <time.h>

#include "nullTransform.h"
#include "svd.h"
#include "structures.h"
#include "settings.h"
#include "FullFrameTransform.h"

#include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include "opencv2/imgproc/imgproc_c.h"

#define     NUM_PARAMS      24

class JelloComplex2 : public ITransform {
    public:
        float ** shiftsX;
        float ** shiftsY;
        float * params;
        static float centerX;

        int jelloMinX, jelloMinY;

        JelloComplex2(TransformationMem *tm);

        JelloComplex2(Mat img1, Mat img2, int index0, int index1, TransformationMem *tm);

        JelloComplex2(vector<Point2f> corners1, vector<Point2f> corners2, int length, TransformationMem *tm);

		void CreateAbsoluteTransformThread(TransformationMem *prevMem, TransformationMem *newMem, threadParams tExtent, float decayX, float decayY);
        void CreateAbsoluteTransform(TransformationMem *prevMem, TransformationMem *newMem);

        void TransformPoint(float x, float y, float &x2, float &y2);

        void TransformPointAbs(float x, float y, float &x2, float &y2);

        void CalculateModelParameters(vector<Point2f> corners1, vector<Point2f> corners2, int length, float* &params);

        float FullModelCostWelsch(vector<Point2f> corners1, vector<Point2f> corners2, float w, float* params);
        float FullModelCostWelschXY(vector<Point2f> corners1, vector<Point2f> corners2, float w, float* params, float &ex, float &ey);

        float FullModelCostLs(vector<Point2f> corners1, vector<Point2f> corners2, float* params);
    
        vector<Transformation> jelloTransforms; 	

        void CalcJelloTransform(Mat img1, Mat img2);
        void GetSineEstimatesWeighted2(vector<float> ys, vector<float> diffs, vector<float> weights, vector<float> startingWeights, float* &result);
        void ImproveSineEstimatesWelsch(vector<float> ys, vector<float> diffs, vector<float> weights, int length, float* params, float* &updates, float &lambda, float w);
        float SineEstimatesCostWelsch(vector<float> ys, vector<float> diffs, vector<float> weights, float* params, float w);
        void FullModelWelschFit(vector<Point2f> corners1, vector<Point2f> corners2, int length, float* params, float* &updates, float &lambda, float w);
        void AssignShiftMem(TransformationMem *tm);

		void getUpdate(arma::Mat<float> jacobian, arma::Col<float> fVector, int numParams, float* &updates);

        static vector<float> allWelschCosts;
        static vector<float> initialWelschCosts;
        static vector<float> fullFrameWelschCosts;

        float getModel2CostWelsch(vector<Point2f> corners1, vector<Point2f> corners2, float* params, float w);
        void model2LMIterationWelsch(vector<Point2f> corners1, vector<Point2f> corners2, int length, float* params, float* &updates, float &lambda, float w);

        static void analyzeTransformAccuracies();
};


#endif
