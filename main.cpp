// Video Image PSNR and SSIM
#include <iostream> // for standard I/O
#include <string>   // for strings
#include <iomanip>  // for controlling float print precision
#include <sstream>  // string to number conversion
#include <time.h>
#include <filesystem>
#include <unistd.h>

#include "coreFuncs.h"
#include "FullFrameTransform.h"
#include "FullFrameTransform2.h"
#include "nullTransform.h"
#include "jelloTransform1.h"
#include "jelloTransform2.h"
#include "JelloComplex1.h"
#include "JelloComplex2.h"
//#include "mainHelpers.h"

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
vector<TRANSFORM> getImageTransformsFromGrey(vector<Mat> greyInput){
	vector<TRANSFORM> result;

	TRANSFORM::imgHeight = greyInput[0].rows;
	TRANSFORM::imgWidth = greyInput[0].cols;

	TRANSFORM nullTrans;

	result.push_back(nullTrans);
	
	for(int i=0;i<(int)(greyInput.size()-1);i++){
		printf("\b\b\b\b\b\b\b\b\b%d/%d", i, (int)(greyInput.size()-1));
		fflush(stdout);

		TRANSFORM t(greyInput[i], greyInput[i+1], i, i+1);
		//printf("%f %f %f\n", t.params[0], t.params[1], t.params[2]);
		t.CreateAbsoluteTransform(result[i]);
		//printf("%f\n", t.shiftsX[100][100]);
		result.push_back(t);
	}

	return result;
}

template <class TRANSFORM>
vector<Mat> transformMats(vector<Mat> input, vector<TRANSFORM> transforms){
	vector<Mat> result;

	//transform mats
	for(int i=0;i<(int)input.size();i++){
		printf("\b\b\b\b\b\b\b\b\b\b%d/%d", i, (int)input.size());
		fflush(stdout);

		Mat out = transforms[i].TransformImage(input[i]);
		result.push_back(out);
	}

	//crop mats
	imgBound bound = transforms[0].frameBound;
	printf("image bound: x: %d - %d     y: %d - %d\n", bound.minX, bound.maxX, bound.minY, bound.maxY);

	Rect r(bound.minX, bound.minY, bound.maxX - bound.minX, bound.maxY - bound.minY);
	for(int i=0;i<(int)result.size();i++){
		result[i] = Mat(result[i], r);
	}
	
	return result;
}

template <class TRANSFORM>
void evalTransform(char *inFileName, char *outFileName){
	VideoCapture capture = VideoCapture(inFileName);
	if (capture.isOpened()) {
        printf("Opened %s\n", inFileName);
        fflush(stdout);
    } else {
        printf("Could not open %s\n", inFileName);
        return;
    }

	int numFrames;
	numFrames = (int)capture.get(CAP_PROP_FRAME_COUNT);
	#ifdef MAX_FRAMES
		numFrames = min(numFrames, NUM_FRAMES);
	#endif
	printf("number of frames: %d\n", numFrames);

	int width = capture.get(CAP_PROP_FRAME_WIDTH);
	int height = capture.get(CAP_PROP_FRAME_HEIGHT);
	double fps = capture.get(CAP_PROP_FPS);
	TRANSFORM::processedFrameCount = 0;

	printf("height: %d   width: %d   fps: %f\n", height, width, fps);

	time_t start = time(NULL);

	printf("getting all frames into mat form\n");
	vector<Mat> inputFrames = getAllInputFrames(&capture, numFrames);
	printf("got frames\n");

	printf("making grayscale frames\n");
	vector<Mat> greyInput = convertFramesToGrayscale(inputFrames);
	printf("done\n");

	printf("creating transformations\n");
	vector<TRANSFORM> transforms = getImageTransformsFromGrey<TRANSFORM>(greyInput);
	printf("done\n");
	TRANSFORM::analyzeTransformAccuracies();

	printf("creating transformed matrices\n");
	vector<Mat> outputMats = transformMats<TRANSFORM>(inputFrames, transforms);
	printf("done\n");

	printf("Saving output mats to file\n");
	writeVideo(outputMats, fps, outFileName);
	printf("done\n");


	time_t end = time(NULL);

	float frameTime = (float)(1000*(end-start)) / (float)inputFrames.size();
	printf("average time / frame: %f ms\n", frameTime);
}

float chiSquaredRandomnessTest(vector<Point2f> corners, int height, int width){
	int numDivisionsX = 40, numDivisionsY = 25;

	int countsLength = numDivisionsX*numDivisionsY;
	int *counts = new int[countsLength];
	memset(counts, 0, countsLength*sizeof(int));

	for(int i=0;i<(int)corners.size();i++){
		int x = (int)(corners[i].x/width*numDivisionsX);
		int y = (int)(corners[i].y/height*numDivisionsY);
		counts[y*numDivisionsY+x] ++;
	}

	float result = 0;
	float expected = (float)corners.size()/(float)countsLength;
	for(int i=0;i<countsLength;i++){
		result += pow((float)(counts[i] - expected), 2)/expected;
	}

	return result;
}

float getSumOfMinEigs(Mat input, vector<Point2f> corners){

	Mat minEigs(input.rows, input.cols, CV_32FC1);
	cornerMinEigenVal(input, minEigs, 3);
	float sum = 0;
	for(int i=0;i<(int)corners.size();i++){
		int x = (int)corners[i].x;
		int y = (int)corners[i].y;
		float eig = ((float*)(minEigs.data))[ y*minEigs.step1()+ x*minEigs.channels()];
		sum += eig;
	}

	return sum / (float) corners.size();
}

void testPointExtraction(char *inFileName){
	finalStageCounts = new int[5];
	memset(finalStageCounts, 0, 5*sizeof(int));
	VideoCapture capture = VideoCapture(inFileName);

	int numFrames;
	numFrames = (int)capture.get(CAP_PROP_FRAME_COUNT);
	numFrames = min(numFrames, 50);
	//numFrames = 1;

	int width = capture.get(CAP_PROP_FRAME_WIDTH);
	int height = capture.get(CAP_PROP_FRAME_HEIGHT);
	double fps = capture.get(CAP_PROP_FPS);

	printf("height: %d   width: %d   fps: %f\n", height, width, fps);

	//time_t start = time(NULL);

	printf("getting all frames into mat form\n");
	vector<Mat> inputFrames = getAllInputFrames(&capture, numFrames);
	printf("got frames\n");

	printf("making grayscale frames\n");
	vector<Mat> greyInput = convertFramesToGrayscale(inputFrames);
	printf("done\n");

	//namedWindow("window", WINDOW_NORMAL );

	float avChiSquared = 0;
	float avNumCorners = 0;
	float avMinEig = 0;

	//Ptr<CLAHE> clahe = createCLAHE();
	//clahe->setClipLimit(4);

	for(int i=0;i<(int)inputFrames.size();i++){

		//Mat clahe;
		//clahe->apply(greyInput[i], clahe);

		//vector<Point2f> corners1 = extractCornersToTrack(greyInput[i]);
		//vector<Point2f> corners1 = extractCornersToTrackColor(inputFrames[i]);
		//vector<Point2f> corners1 = extractCornersToTrack(clahe);
		vector<Point2f> corners1 = extractCornersRecursive(greyInput[i]);

		float sumOfEigs = getSumOfMinEigs(greyInput[i], corners1);
		//printf("sum of eigs: %f\n", sumOfEigs);
		avMinEig += sumOfEigs;
		
		Mat out = inputFrames[i];
		for(int i=0;i<(int)corners1.size();i++){
			circle(out, corners1[i], 4, Scalar(0, 0, 255));
			circle(out, corners1[i], 3, Scalar(0, 0, 255));
		}
		//printf("number of extracted features: %d\n", (int)corners1.size());
		//imshow("window", out); waitKey(0);     
		float chiSquared = chiSquaredRandomnessTest(corners1, height, width);
		avChiSquared += chiSquared;
		avNumCorners += (float) corners1.size();
		//printf("chiSquared: %f\n", chiSquared);
	}

	avChiSquared /= (float) inputFrames.size();
	avNumCorners /= (float) inputFrames.size();
	avMinEig /= (float) inputFrames.size();
	printf("average chi squared value: %f\n", avChiSquared);
	printf("average number of corners: %f\n", avNumCorners);
	printf("average min eigenvalue: %f\n", avMinEig);

	printf("counts: %d  %d  %d  %d  %d\n", finalStageCounts[0], finalStageCounts[1], finalStageCounts[2], finalStageCounts[3], finalStageCounts[4]);
}

void chiSquaredRandomBenchmark(){
	int height = 720;
	int width = 1280;

	vector<Point2f> corners1;
	int numCorners = 897;
	for(int i=0;i<numCorners;i++){
		float x = rand() / (double) RAND_MAX * width;
		float y = rand() / (double) RAND_MAX * height;
		corners1.push_back(Point2f(x, y));
	}
	float chiSquared = chiSquaredRandomnessTest(corners1, height, width);
	printf("chi squared value for %d random corners: %f\n", numCorners, chiSquared);
}

void plotCornersOnColor(char *inFileName){

	VideoCapture capture = VideoCapture(inFileName);

	int numFrames;
	numFrames = (int)capture.get(CAP_PROP_FRAME_COUNT);
	numFrames = min(numFrames, 50);
	//numFrames = 1;

	int width = capture.get(CAP_PROP_FRAME_WIDTH);
	int height = capture.get(CAP_PROP_FRAME_HEIGHT);
	double fps = capture.get(CAP_PROP_FPS);

	printf("height: %d   width: %d   fps: %f\n", height, width, fps);

	//time_t start = time(NULL);

	printf("getting all frames into mat form\n");
	vector<Mat> inputFrames = getAllInputFrames(&capture, numFrames);
	printf("got frames\n");

	printf("making grayscale frames\n");
	vector<Mat> greyInput = convertFramesToGrayscale(inputFrames);
	printf("done\n");

	printf("length: %d\n", (int)greyInput.size());

	//namedWindow("window", WINDOW_NORMAL );

	vector<Point2f> corners = extractCornersToTrack(greyInput[0]);
	Mat img = inputFrames[0];
	printf("1\n");
	for(int i=0;i<(int)corners.size();i++)
	{
			circle(img, corners[i], 4, Scalar(0, 0, 255));	
			circle(img, corners[i], 3, Scalar(0, 0, 255));
	}
	printf("2\n");

	imshow("window", img); 
	imwrite("corners.jpg", img);
	waitKey(0);

}
int main(int argc, char* argv[]){

	int pass = 0;

	int opt;
	while((opt = getopt(argc, argv, "p:")) != -1)
	{
		//printf("opt: %c   optopt: %c   optarg: %s\n", (char)opt, (char)optopt, optarg);
		switch (opt)
		{
		case 'p':
			if (strcmp(optarg, "1") && strcmp(optarg, "2")) {
				printf("Please specify pass, either 1 or 2\n");
				return 1;
			}
			pass = atoi(optarg);
			break;
		}
	}

	// Help
    if((argc-optind) < 2) {
        const char *exec_name = std::filesystem::path(argv[0]).filename().c_str();
        printf("Usage: %s [options] <input_video_file> <output_video_file>\n", exec_name);
        //printf("Options:\n");
        //printf("  -p <pass>: Perform only selected pass. Can be either 1 or 2.\n");
        return 1;
    }
    
    // Ordered parameters
    char *inFileName = argv[optind];
    char *outFileName = argv[optind+1];

	// Do processing
	//evalTransform<NullTransform>(inFileName, outFileName);
	//evalTransform<FullFrameTransform>(inFileName, outFileName);
	//evalTransform<FullFrameTransform2>(inFileName, outFileName);
	//evalTransform<JelloTransform1>(inFileName, outFileName);
	//evalTransform<JelloTransform2>(inFileName, outFileName);
	//evalTransform<JelloComplex1>(inFileName, outFileName);
	evalTransform<JelloComplex2>(inFileName, outFileName);
	
	//plotCornersOnColor(inFileName);
	//testPointExtraction(inFileName);
	//chiSquaredRandomBenchmark();

	return 0;
}
