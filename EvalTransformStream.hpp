// Video Image PSNR and SSIM
#include <iostream> // for standard I/O
#include <string>   // for strings
#include <iomanip>  // for controlling float print precision
#include <sstream>  // string to number conversion
#include <time.h>
#include <filesystem>
#include <thread>
#include <unistd.h>
#include <argp.h>

#include "coreFuncs.h"
#include "FullFrameTransform.h"
#include "FullFrameTransform2.h"
#include "nullTransform.h"
#include "jelloTransform1.h"
#include "jelloTransform2.h"
#include "JelloComplex1.h"
#include "JelloComplex2.h"
#include "EvalTransformStream.h"

#include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include "opencv2/imgproc/imgproc_c.h"

using namespace std;
using namespace cv;


template <class TRANSFORM>
void AllocateMem(TransformationMem *tm, int width, int height, int numParams)
{
    tm->width = width;
    tm->height = height;
    tm->numParams = numParams;

    tm->shiftsX = new float*[height];
    tm->shiftsY = new float*[height];

    tm->params = new float[numParams];
    memset(tm->params, 0, numParams*sizeof(float));

    for(int row=0;row<height;row++){
        tm->shiftsX[row] = new float[width];
        tm->shiftsY[row] = new float[width];
        memset(tm->shiftsX[row], 0, width * sizeof(float));
        memset(tm->shiftsY[row], 0, width * sizeof(float));
    }
}

template <class TRANSFORM>
void CopyMem(TransformationMem *src, TransformationMem *dst)
{
    memcpy(dst->params, src->params, sizeof(float)*src->numParams);

    for(int row=0;row<src->height;row++){
        memcpy(dst->shiftsX[row], src->shiftsX[row], src->width * sizeof(float));
        memcpy(dst->shiftsY[row], src->shiftsY[row], src->width * sizeof(float));
    }
}

template <class TRANSFORM>
Mat Crop(Mat input, cropBound *cBound, imgBound frameBound, Size size)
{
    // Output frame aspect
    double aspect = (double)size.height/size.width;

    if(!args.twoPass)
    {
        // Smoothen transform boundaries
        if(args.cSmooth>0)
        {
            float maxShift = fmax(size.width * args.djdShift, size.height * args.djdShift);
            int width1 = cBound->maxX-cBound->minX;
            int width2 = frameBound.maxX-frameBound.minX;
            int height1 = cBound->maxY-cBound->minY;
            int height2 = frameBound.maxY-frameBound.minY;
            double cSmoothX = (1 - fmin( abs(width1-width2) / maxShift, 1)) * args.cSmooth;
            double cSmoothY = (1 - fmin( abs(height1-height2) / maxShift, 1)) * args.cSmooth;
            cBound->minX = frameBound.minX+(cBound->minX-frameBound.minX)*cSmoothX;
            cBound->maxX = frameBound.maxX+(cBound->maxX-frameBound.maxX)*cSmoothX;
            cBound->minY = frameBound.minY+(cBound->minY-frameBound.minY)*cSmoothY;
            cBound->maxY = frameBound.maxY+(cBound->maxY-frameBound.maxY)*cSmoothY;
        } else {
            // No decay
            cBound->minX = frameBound.minX;
            cBound->maxX = frameBound.maxX;
            cBound->minY = frameBound.minY;
            cBound->maxY = frameBound.maxY;
        }
    }

    // All values
    int minX = round(cBound->minX);
    int maxX = round(cBound->maxX);
    int minY = round(cBound->minY);
    int maxY = round(cBound->maxY);
    int width = maxX-minX;
    int height = maxY-minY;

    //printf("minX=%i maxX=%i minY=%i maxY=%i\n", minX, maxX, minY, maxY);

    // Conform to aspect
    int pWidth = (int)(height/aspect);      // Proposed width keeping height
    int pHeight = (int)(width*aspect);      // Proposed height keeping width
    if(pHeight<height)
    {
        // Crop vertically
        int c = (maxY-minY)/2;
        minY = minY+(c-pHeight/2);
        maxY = minY+pHeight;
    } else {
        // Crop horizontally
        int c = (maxX-minX)/2;
        minX = minX+(c-pWidth/2);
        maxX = minX+pWidth;
    }

    // Limit (crash-proofing)
    /*if(minX<0) minX = 0;
    if(maxX>size.width) maxX = size.width;
    if(minY<0) minY = 0;
    if(maxY>size.height) maxY = size.height;*/

    // Crop and upscale
    Rect rCrop(minX, minY, maxX-minX, maxY-minY);
    Mat out = Mat(size, input.type());
    cv::resize(Mat(input, rCrop), out, size, 0, 0, INTER_LANCZOS4);

    return out;
}

template <class TRANSFORM>
void evalTransformStream(char *inFileName, char *outFileName, bool prePass)
{
    // Open input file
    VideoCapture capture;
    if(args.encBitrate>=0 || args.encQuality>=0)
    {
        // GStreamer
        printf("Using GStreamer OpenCV decoding backend\n");
        string vcString;
        vcString.append("filesrc location=\"");
        vcString.append(inFileName);
        vcString.append("\" ! decodebin ! videoconvert ! video/x-raw,format=BGR ! appsink");
        capture = VideoCapture(vcString, CAP_GSTREAMER);
    } else {
        // Default
        printf("Using default OpenCV decoding backend\n");
        capture = VideoCapture(inFileName);
    }
    if (capture.isOpened()) {
        printf("Opened %s\n", inFileName);
    } else {
        printf("Could not open %s\n", inFileName);
        exit(1);
    }

    // Get file properties
    int numFrames;
    numFrames = (int)capture.get(CAP_PROP_FRAME_COUNT);
    printf("number of frames: %d\n", numFrames);
    if(numFrames < 3)
    {
        printf("Cannot work with file with less than 3 frames\n");
        exit(1);
    }
    int width = capture.get(CAP_PROP_FRAME_WIDTH);
    int height = capture.get(CAP_PROP_FRAME_HEIGHT);
    double fps = capture.get(CAP_PROP_FPS);
    TRANSFORM::processedFrameCount = 0;
    printf("height: %d   width: %d   fps: %f\n", height, width, fps);

    // Create output file
    VideoWriter outputVideo;
    Size size(width, height);
    if(args.encBitrate>-2 || args.encQuality>-2)
    {
        // GStreamer
        printf("Using GStreamer OpenCV encoding backend\n");
        string gsString;
        gsString.append("appsrc ! videoconvert ! ");
        gsString.append(args.codec);
        if(args.encQuality>=0) {
            char a[50];
            sprintf(a, " quantizer=%i", args.encQuality);
            gsString.append(a);
            printf("Encoder quantizer: %i\n", args.encQuality);
        }
        if(args.encBitrate>=0) {
            char a[50];
            sprintf(a, " bitrate=%i", args.encBitrate);
            gsString.append(a);
            printf("Encoder bitrate: %i kbps\n", args.encBitrate);
        }
        gsString.append(" ! filesink location=\"");
        gsString.append(outFileName);
        gsString.append("\"");
        outputVideo.open(gsString, CAP_GSTREAMER, 0, fps, size, true);
    } else {
        // Default
        printf("Using default OpenCV encoding backend\n");
        outputVideo.open(outFileName, args.fourcc, fps, size, true);
    }
    if(outputVideo.isOpened()) {
        printf("Saving as %s\n", outFileName);
    } else {
        printf("Failed to create output file %s\n", outFileName);
        printf("NOTE: set bitrate or quality to -1 to use codecs which want neither.\n");
        exit(2);
    }

    // Init transforms
    TransformationMem prevMem, newMem;
    AllocateMem<TRANSFORM>(&prevMem, width, height, NUM_PARAMS);
    AllocateMem<TRANSFORM>(&newMem, width, height, NUM_PARAMS);
    TRANSFORM::imgWidth = width;
    TRANSFORM::imgHeight = height;

    // Init others
    Mat greyInput[2];
    static cropBound cBound;

    // Test marker size
    int testMarkerSize;
    if(width>height) testMarkerSize = lround((double)width*TEST_MARKER_SIZE);
    else testMarkerSize = lround((double)height*TEST_MARKER_SIZE);
    if(testMarkerSize<1) testMarkerSize = 1;

    // Read and process
    double procFps = -1;
    time_t tStart = time(NULL);
    int framesRead = 0;
    for(int i=0;i<numFrames;i++)
    {
        Mat frame;
        if (!capture.read(frame)) {
            // Cannot read
            if(args.warnings) printf("warning: cannot get frame %d, skipping\n", i);
        } else {
            // Read
            framesRead++;

            // Print progress
            for(int bs=0; bs<40; bs++) { printf("\b"); }
            printf("%d/%d   fps: ", i, numFrames-2);
            if(procFps>=0) printf("%.2f ", procFps);
            fflush(stdout);     // Make printf work immediately

            // Convert frame to greyscale
            Mat greyMat;
            cvtColor(frame, greyMat, CV_BGR2GRAY);
            if(i>0) greyMat.copyTo(greyInput[1]);
            else greyMat.copyTo(greyInput[0]);

            // Process or test mode
            if(args.test)
            {
                // Test mode
                vector<Point2f> corners = extractCornersToTrack(greyInput[0]);
                for(int i=0;i<(int)corners.size();i++)
                {
                    circle(frame, corners[i], testMarkerSize, Scalar(0, 0, 0), 1);
                    circle(frame, corners[i], testMarkerSize-1, Scalar(255, 255, 255), FILLED);
                }
                outputVideo.write(frame);
            } else {
                // Process if more than 1 frame is read
                if(framesRead>1)
                {
                    // Create a transform matrix using previous the frame and the current
                    TRANSFORM t = TRANSFORM(greyInput[0], greyInput[1], i-1, i, &newMem);
                    t.CreateAbsoluteTransform(&prevMem, &newMem);

                    if(framesRead==2 && (!args.twoPass || prePass))
                    {
                        // Initialize crop bound
                        cBound.minX = t.frameBound.minX;
                        cBound.maxX = t.frameBound.maxX;
                        cBound.minY = t.frameBound.minY;
                        cBound.maxY = t.frameBound.maxY;
                    }

                    if(args.twoPass && prePass && framesRead>2)
                    {
                        // Find the min frame bounds in the whole sequence
                        float cMinX[2], cMaxX[2], cMinY[2], cMaxY[2];
                        cMinX[0] = t.frameBound.minX-newMem.shiftsX[t.frameBound.minY][t.frameBound.minX];
                        cMinX[1] = t.frameBound.minX-newMem.shiftsX[t.frameBound.maxY-1][t.frameBound.minX];
                        cMaxX[0] = t.frameBound.maxX-newMem.shiftsX[t.frameBound.minY][t.frameBound.maxX-1];
                        cMaxX[1] = t.frameBound.maxX-newMem.shiftsX[t.frameBound.maxY-1][t.frameBound.maxX-1];
                        cMinY[0] = t.frameBound.minY-newMem.shiftsY[t.frameBound.minY][t.frameBound.minX];
                        cMinY[1] = t.frameBound.minY-newMem.shiftsY[t.frameBound.maxY-1][t.frameBound.minX];
                        cMaxY[0] = t.frameBound.maxY-newMem.shiftsY[t.frameBound.minY][t.frameBound.maxX-1];
                        cMaxY[1] = t.frameBound.maxY-newMem.shiftsY[t.frameBound.maxY-1][t.frameBound.maxX-1];
                        /*t.TransformPointAbs(t.frameBound.minX, t.frameBound.minY, cMinX[0], cMinY[0]);
                        t.TransformPointAbs(t.frameBound.maxX-1, t.frameBound.minY, cMaxX[0], cMinY[1]);
                        t.TransformPointAbs(t.frameBound.minX, t.frameBound.maxY-1, cMinX[1], cMaxY[0]);
                        t.TransformPointAbs(t.frameBound.maxX-1, t.frameBound.maxY-1, cMaxX[1], cMaxY[1]);*/
                        cBound.minX = fmax(cBound.minX, width-cMaxX[0]);
                        cBound.minX = fmax(cBound.minX, width-cMaxX[1]);
                        cBound.maxX = fmin(cBound.maxX, width-cMinX[0]);
                        cBound.maxX = fmin(cBound.maxX, width-cMinX[1]);
                        cBound.minY = fmax(cBound.minY, height-cMaxY[0]);
                        cBound.minY = fmax(cBound.minY, height-cMaxY[1]);
                        cBound.maxY = fmin(cBound.maxY, height-cMinY[0]);
                        cBound.maxY = fmin(cBound.maxY, height-cMinY[1]);
                        //printf("minX=%f maxX=%f minY=%f maxY=%f\n", cBound.minX, cBound.maxX, cBound.minY, cBound.maxY);
                    } else {
                        // Transform the frame
                        Mat out = t.TransformImage(frame);

                        if(!args.noCrop)
                        {
                            // Crop
                            Mat outCropped = Crop<TRANSFORM>(out, &cBound, t.frameBound, size);
                            outputVideo.write(outCropped);
                        } else {
                            // Not cropped
                            outputVideo.write(out);
                        }
                    }

                    // Shift params
                    CopyMem<TRANSFORM>(&newMem, &prevMem);
                }
            }

            // Shift grey mats
            if(i>0) greyInput[1].copyTo(greyInput[0]);

            // Calculate processing fps
            time_t tEnd = time(NULL);
            if(tEnd-tStart>FPS_AFTER)
            {
                procFps = (double)(i+1)/(tEnd-tStart);
            }
        }
    }

    // Calculate zoom
    if(args.twoPass && prePass)
    {
        printf("\nBoundaries detected: minX=%i, maxX=%i, minY=%i, maxY=%i\n", (int)cBound.minX, (int)cBound.maxX, (int)cBound.minY, (int)cBound.maxY);
    } else {
        outputVideo.release();
    }

    // Analyze accuracies
    //TRANSFORM::analyzeTransformAccuracies();

    printf("\nDone!\n");
}
