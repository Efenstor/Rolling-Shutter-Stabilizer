
// Video Image PSNR and SSIM
#include <iostream> // for standard I/O
#include <string>   // for strings
#include <iomanip>  // for controlling float print precision
#include <sstream>  // string to number conversion
#include <time.h>
#include <thread>

#include "settings.h"
#include "structures.h"
#include "coreFuncs.h"

#include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include "opencv2/imgproc/imgproc_c.h"

using namespace std;
using namespace cv;

extern arguments args;

void GenericTransformPoint(Transformation trans, float x, float y, float &x2, float &y2){
    x2 = (x-trans.ux1) * trans.cos - (y-trans.uy1) * trans.sin + trans.ux2;
    y2 = (x-trans.ux1) * trans.sin + (y-trans.uy1) * trans.cos + trans.uy2;
}

void GenericTransformPointAbs(AbsoluteTransformation absTrans, float x, float y, float &x2, float &y2){
    GenericTransformPoint(absTrans.trans, x, y, x2, y2);
    x2 += absTrans.idx;
    y2 += absTrans.idy;
}

vector<Point2f> extractCornersToTrackColor(Mat img){

	// Create Matrices (make sure there is an image in input!)
	Mat channel[3];

	// The actual splitting.
	split(img, channel);

	/*
	// Create Windows
	namedWindow("Red",1);
	namedWindow("Green",1);
	namedWindow("Blue",1);

	// Display
	imshow("Red", channel[0]);
	imshow("Green", channel[1]);
	imshow("Blue", channel[2]);
	waitKey(0);     
	*/

	vector<Point2f> result = extractCornersToTrack(channel[0], args.corners/3);
	vector<Point2f> addition = extractCornersToTrack(channel[1], args.corners/3);
	vector<Point2f> addition2 = extractCornersToTrack(channel[2], args.corners/3);

	/*
	result.insert(result.end(), addition.begin(), addition.end());
	result.insert(result.end(), addition2.begin(), addition2.end());
	*/

	int startLength = (int)result.size();
	for(int i=0;i<(int)addition.size();i++){
		bool usePoint = true;
		for(int j=0;j<startLength;j++){
			if(norm(addition[i] - result[j]) < 3.0){
				usePoint = false;
				break;
			}

		}

		if(usePoint)
			result.push_back(addition[i]);
	}

	startLength = (int)result.size();
	for(int i=0;i<(int)addition2.size();i++){
		bool usePoint = true;
		for(int j=0;j<startLength;j++){
			if(norm(addition2[i] - result[j]) < 3.0){
				usePoint = false;
				break;
			}

		}

		if(usePoint)
			result.push_back(addition2[i]);
	}

	//printf("remaining points: %d\n", (int)result.size());

	return result;
}

#define minDistance 5.0


vector<Point2f> extractCornersRecursive(Mat img){
	return extractCornersRecursiveInner(img, args.corners, Point2f(0, 0));
}

int *finalStageCounts;

vector<Point2f> extractCornersRecursiveInner(Mat img, int numCorners, Point2f offset){
	vector<Point2f> result;

	goodFeaturesToTrack(img, result, numCorners, args.qualityLevel, minDistance);

	int counts[4];
	memset(counts, 0, 4*sizeof(int));

	int halfHeight = img.rows/2;
	int halfWidth = img.cols/2;
	int minCount = result.size() / 10;	//min of 10% in each quarter

	for(int i=0;i<(int)result.size();i++){
		int index = 0;
		if(result[i].y > halfHeight)
			index++;
		if(result[i].x > halfWidth)
			index+=2;

		counts[index]++;
	}

	//printf("total counts: %d      counts: %d %d %d %d\n", numCorners, counts[0], counts[1], counts[2], counts[3]);
	bool countsUneven = false;
	if(counts[0] < minCount)	countsUneven = true;
	else if(counts[1] < minCount) 	countsUneven = true;
	else if(counts[2] < minCount) 	countsUneven = true;
	else if(counts[3] < minCount) 	countsUneven = true;

	if(countsUneven && numCorners > 4){
		result.erase(result.begin(), result.end());

		//printf("counts are too uneven, doing another level\n");

		Mat topHalf = img.rowRange(0, halfHeight);
		Mat topLeft = topHalf.colRange(0, halfWidth);
		Mat topRight = topHalf.colRange(halfWidth, img.cols);
		Point2f topLeftOffset(0, 0);
		Point2f topRightOffset(halfWidth, 0);

		Mat bottomHalf = img.rowRange(halfHeight, img.rows);
		Mat bottomLeft = bottomHalf.colRange(0, halfWidth);
		Mat bottomRight = bottomHalf.colRange(halfWidth, img.cols);
		Point2f bottomLeftOffset(0, halfHeight);
		Point2f bottomRightOffset(halfWidth, halfHeight);

		vector<Point2f> q0 = extractCornersRecursiveInner(topLeft, numCorners/4, topLeftOffset);
		vector<Point2f> q1 = extractCornersRecursiveInner(topRight, numCorners/4, topRightOffset);
		vector<Point2f> q2 = extractCornersRecursiveInner(bottomLeft, numCorners/4, bottomLeftOffset);
		vector<Point2f> q3 = extractCornersRecursiveInner(bottomRight, numCorners/4, bottomRightOffset);

		result.insert(result.end(), q0.begin(), q0.end());
		result.insert(result.end(), q1.begin(), q1.end());
		result.insert(result.end(), q2.begin(), q2.end());
		result.insert(result.end(), q3.begin(), q3.end());
			
	} else{
		int depth = (int)log2(args.corners / numCorners)/2;
		finalStageCounts[depth] ++;

		 if(result.size()>0 && !args.noSubpix)
		 {
			cornerSubPix( img, result, Size( args.winSize, args.winSize ), Size( -1, -1 ),
				TermCriteria( TermCriteria::COUNT+TermCriteria::EPS, args.iter, args.epsilon ) );
		 }
	}

	for(int i=0;i<(int)result.size();i++){
		result[i] += offset;
	}

	return result;
}

vector<Point2f> extractCornersToTrack(Mat img){
	return extractCornersToTrack(img, args.corners);
}

void extractCornersToTrackThread(Mat img, int numCorners, vector<Point2f> &corners, threadParams tExtent)
{
	for(int col = tExtent.from; col<tExtent.to; col++)
	{
		int xLow = img.cols * col / args.cornerCols;
		int xHigh = img.cols * (col+1) / args.cornerCols;
		for(int row = 0; row<args.cornerRows; row++)
		{
			int yLow = img.rows * row / args.cornerRows;
			int yHigh = img.rows * (row+1) / args.cornerRows;
			
			Mat m1 = img.rowRange(yLow, yHigh);
			Mat m = m1.colRange(xLow, xHigh);
			
			Point2f offset(xLow, yLow);
			vector<cv::Point2f> segmentCorners;
			goodFeaturesToTrack(m, segmentCorners, numCorners/(args.cornerCols*args.cornerRows),
					args.qualityLevel, minDistance);
			for(int i=0; i<(int)segmentCorners.size(); i++)
			{
				corners.push_back(segmentCorners[i] + offset);
			}
		}
	}
}

vector<Point2f> extractCornersToTrack(Mat img, int numCorners)
{
	std::vector<threadParams> tExtent;
	std::vector<Point2f> corners;

	// Prepare threads
	int tNum = args.threads;
	if(tNum>args.cornerCols) tNum = args.cornerCols;
	double colsPerThread = ((double)args.cornerCols/tNum);
	for(int t=0; t<tNum; t++)
	{
		threadParams tp;
		
		tp.from = lround(t*colsPerThread);
		if(t<tNum-1) tp.to = lround((t+1)*colsPerThread);
		else tp.to = args.cornerCols;
		
		tExtent.push_back(tp);
		//printf("tp.from=%d   tp.to==%d\n", tp.from, tp.to);
	}
	std::vector<std::vector<Point2f>> tCorners(tNum);
	
	//double minDistance = 5.0;
	
	// Find features to track
	int type = 2;
	switch(type){
	case 0: goodFeaturesToTrack(img, corners, numCorners, args.qualityLevel, minDistance);
		break;
		
	case 1: goodFeaturesToTrack(img, corners, numCorners, args.qualityLevel, minDistance, noArray(), 3, true); //harris detector
		break;
		
	case 2:
		// Create threads
		std::vector<std::thread> threads;
		for(int t=0; t<tNum; t++)
		{
			std::thread newThr(extractCornersToTrackThread, img, numCorners, ref(tCorners.at(t)), tExtent.at(t));
			threads.push_back(move(newThr));
		}
		// Join threads
		for(int t=0; t<tNum; t++)
		{
			threads.at(t).join();
		}
		// Join vectors
		for(int t=0; t<tNum; t++)
		{
			for(int i=0; i<(int)tCorners.at(t).size(); i++)
			{
				corners.push_back(tCorners.at(t).at(i));
			}
		}
		break;
	}
	
	if(!args.noSubpix)
	{
		cornerSubPix(img, corners, Size( args.winSize, args.winSize ), Size( -1, -1 ), 
			TermCriteria( TermCriteria::COUNT+TermCriteria::EPS, args.iter, args.epsilon ) );
	}
	
	return corners;
}

FeaturesInfo extractFeaturesToTrack(Mat img){
	vector<Point2f> corners = extractCornersToTrack(img);

	vector<Mat> pyramid;
	buildOpticalFlowPyramid(img, pyramid, Size(args.winSize, args.winSize), 3);
	
	FeaturesInfo fi;
	fi.features = corners;
	fi.pyramid = pyramid;
	
	return fi;
}

vector<Mat> getAllInputFrames(VideoCapture* capture, int numFrames){
    vector<Mat> result;
    
    capture->set(CAP_PROP_POS_FRAMES,0);
	
    for(int i=0;i<numFrames;i++)
    {
        Mat m;
        if ( capture->read(m) == false) {
            printf("cannot get frame %d, skipping\n", i);
        } else {
            result.push_back(m);
        }
    }
    
    return result;
}

Mat matToGrayscale(Mat m){
	Mat greyMat;
	cvtColor(m, greyMat, CV_BGR2GRAY);
	return greyMat;
}

vector<Mat> convertFramesToGrayscale(vector<Mat> input){
	vector<Mat> result;
	for(int i=0;i<(int)input.size();i++)
	{
		result.push_back(matToGrayscale(input[i]));
	}
	return result;
}

void writeVideo(vector<Mat> frames, float fps, string filename){
	int width = frames[0].cols;
	int height = frames[0].rows;
	
	VideoWriter outputVideo;
	
	#ifdef ROTATE90
	Size size(height, width);
	#else
	Size size(width, height);
	#endif

	outputVideo.open(filename, VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, size, true);
	if(!outputVideo.isOpened()){
		printf("output video failed to open\n");
		exit(2);
	}

	//namedWindow("window", WINDOW_NORMAL );

	for(int i=0;i<(int)frames.size();i++){
		
		for(int bs=0; bs<20; bs++) { printf("\b"); }
		printf("%d/%d", i, (int)(frames.size()-1));
		fflush(stdout);		// Make printf work immediately
		
		Mat frame = frames[i];

		#ifdef ROTATE90
		outputVideo.write(frame.t());
		#else
		outputVideo.write(frame);
		#endif
	}
	printf("\n");
}

int GetPointsToTrack(Mat img1, Mat img2, vector<Point2f> &corners1, vector<Point2f> &corners2){

	Size img_sz = img1.size();
	Mat imgC(img_sz,1);

	corners1 = extractCornersToTrack(img1);
	//corners1 = extractCornersRecursive(img1);

	corners1.reserve(args.corners);
	corners2.reserve(args.corners);

	//Size pyr_sz = Size( img_sz.width+8, img_sz.height/3 );
	
	std::vector<uchar> features_found; 
	features_found.reserve(args.corners);
	std::vector<float> feature_errors; 
	feature_errors.reserve(args.corners);
    
	calcOpticalFlowPyrLK( img1, img2, corners1, corners2, features_found, feature_errors,
		Size( args.winSize, args.winSize ), args.maxLevel,
		TermCriteria( TermCriteria::COUNT+TermCriteria::EPS, args.iter, args.epsilon ), 0, args.eigThr );

	return (int) features_found.size();
}
