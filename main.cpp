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
//#include "mainHelpers.h"

#include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include "opencv2/imgproc/imgproc_c.h"

#define VERSION "0.5"
#define PASS1EXT ".pass1"

using namespace std;
using namespace cv;

arguments args;

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
Mat Crop(Mat input, imgBound *cropBound, imgBound newBound, Size size)
{
	// Output frame aspect
	double aspect = (double)size.height/size.width;

	// Smoothen transform boundaries
	if(args.cSmooth>0)
	{
		cropBound->minX = lround(newBound.minX+(cropBound->minX-newBound.minX)*args.cSmooth);
		cropBound->maxX = lround(newBound.maxX+(cropBound->maxX-newBound.maxX)*args.cSmooth);
		cropBound->minY = lround(newBound.minY+(cropBound->minY-newBound.minY)*args.cSmooth);
		cropBound->maxY = lround(newBound.maxY+(cropBound->maxY-newBound.maxY)*args.cSmooth);
	}
	
	// All values
	int minX = cropBound->minX;
	int maxX = cropBound->maxX;
	int minY = cropBound->minY;
	int maxY = cropBound->maxY;
	int width = maxX-minX+1;
	int height = maxY-minY+1;

	// Zoom
	int zoomOffX = (width-width/args.zoom)/2;
	int zoomOffY = (height-height/args.zoom)/2;
	minX += zoomOffX;
	maxX -= zoomOffX;
	minY += zoomOffY;
	maxY -= zoomOffY;
	width = maxX-minX+1;
	height = maxY-minY+1;
	
	// Conform to aspect
	int pWidth = (int)(height/aspect);		// Proposed width keeping height
	int pHeight = (int)(width*aspect);		// Proposed height keeping width
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
	if(maxX>size.width-1) maxX = size.width-1;
	if(minY<0) minY = 0;
	if(maxY>size.height-1) maxY = size.height-1;*/
	
	// Crop and upscale
	Rect rCrop(minX, minY, maxX-minX, maxY-minY);
	Mat out = Mat(size, input.type());
	cv::resize(Mat(input, rCrop), out, size, 0, 0, INTER_CUBIC);
	
	return out;
}

template <class TRANSFORM>
void evalTransformStream(char *inFileName, char *outFileName)
{
	// Open input file
	VideoCapture capture = VideoCapture(inFileName);
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
	int width = capture.get(CAP_PROP_FRAME_WIDTH);
	int height = capture.get(CAP_PROP_FRAME_HEIGHT);
	double fps = capture.get(CAP_PROP_FPS);
	TRANSFORM::processedFrameCount = 0;
	printf("height: %d   width: %d   fps: %f\n", height, width, fps);

	// Create output file
	VideoWriter outputVideo;
	Size size(width, height);
	outputVideo.open(outFileName, VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, size, true);
	if(outputVideo.isOpened()) {
		printf("Saving as %s\n", outFileName);
	} else {
		printf("Failed to create output file %s\n", outFileName);
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
    imgBound cropBound;
    capture.set(CAP_PROP_POS_FRAMES, 0);

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
			fflush(stdout);		// Make printf work immediately
			
			// Convert frame to greyscale
			Mat greyMat;
			cvtColor(frame, greyMat, CV_BGR2GRAY);
			if(i>0) greyMat.copyTo(greyInput[1]);
			else greyMat.copyTo(greyInput[0]);
			
			// Process if more than 1 frame is read
			if(framesRead>1)
			{
				// Create a transform matrix using previous the frame and the current
				TRANSFORM t = TRANSFORM(greyInput[0], greyInput[1], i-1, i, &newMem);
				//printf("%f %f %f\n", t.params[0], t.params[1], t.params[2]);
				t.CreateAbsoluteTransform(&prevMem, &newMem);
				//printf("%f\n", t.shiftsX[100][100]);

				// Transform the frame
				Mat out = t.TransformImage(frame);

				// Save the frame to the file
				if(!args.noCrop)
				{
					// Cropped
					if(framesRead==2) cropBound = t.frameBound;  // First frame
					Mat outCropped = Crop<TRANSFORM>(out, &cropBound, t.frameBound, size);
					outputVideo.write(outCropped);
				} else {
					// Full frame
					outputVideo.write(out);
				}
				
				CopyMem<TRANSFORM>(&newMem, &prevMem);
			}
			
			// Shift
			if(i>0) greyInput[1].copyTo(greyInput[0]);
			
			// Calculate processing fps
			time_t tEnd = time(NULL);
			if(tEnd-tStart>FPS_AFTER)
			{
				procFps = (double)(i+1)/(tEnd-tStart);
			}
        }
    }

    // Analyze accuracies
    //TRANSFORM::analyzeTransformAccuracies();
    
    printf("\nDone!\n");
}

template <class TRANSFORM>
vector<TRANSFORM> getImageTransformsFromGrey(vector<Mat> greyInput){
	vector<TRANSFORM> result;

	TRANSFORM::imgHeight = greyInput[0].rows;
	TRANSFORM::imgWidth = greyInput[0].cols;

	TRANSFORM nullTrans;

	result.push_back(nullTrans);

	// Process frames
	double fps = -1;
	time_t tStart = time(NULL);
	for(int i=0;i<(int)(greyInput.size()-1);i++)
	{	
		// Print progress
		for(int bs=0; bs<40; bs++) { printf("\b"); }
		printf("%d/%d   fps: ", i, (int)(greyInput.size()-2));
		if(fps>=0) printf("%.2f", fps);
		fflush(stdout);		// Make printf work immediately

		// Create transforms
		TRANSFORM t(greyInput[i], greyInput[i+1], i, i+1);
		//printf("%f %f %f\n", t.params[0], t.params[1], t.params[2]);
		t.CreateAbsoluteTransform(result[i]);
		//printf("%f\n", t.shiftsX[100][100]);
		
		// Add to the transform vector
		result.push_back(t);
		
		// Calculate fps
		time_t tEnd = time(NULL);
		if(tEnd-tStart>FPS_AFTER)
		{
			fps = (double)(i+1)/(tEnd-tStart);
		}
	}
	printf("\n");

	return result;
}

template <class TRANSFORM>
vector<Mat> transformMats(vector<Mat> input, vector<TRANSFORM> transforms){
	vector<Mat> result;

	//transform mats
	double fps = -1;
	time_t tStart = time(NULL);
	for(int i=0;i<(int)input.size();i++)
	{
		// Print progress
		for(int bs=0; bs<20; bs++) { printf("\b"); }
		printf("%d/%d   fps: ", i, (int)(input.size()-1));
		if(fps>=0) printf("%.2f", fps);
		fflush(stdout);		// Make printf work immediately

		// Create transforms
		Mat out = transforms[i].TransformImage(input[i]);
		
		// Add to the output vector
		result.push_back(out);
		
		// Calculate fps
		time_t tEnd = time(NULL);
		if(tEnd-tStart>FPS_AFTER)
		{
			fps = (double)(i+1)/(tEnd-tStart);
		}
	}
	printf("\n");

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

int checkNumberArg(char *optarg, double min, double max, bool fp)
{
	int i;
	bool point = false;
	bool minus = false;
	
	// Is it a number?
	for(i=0; i<(int)strlen(optarg); i++)
	{
		if(fp) {
			// Must be floating point
			if((optarg[i]<'0' || optarg[i]>'9') && optarg[i]!='-' && optarg[i]!='.' && optarg[i]!=',') {
				return 1;
			} else {
				// Only one point is allowed
				if(optarg[i]=='.' || optarg[i]==',')
				{
					if(point) return 3;
					else point = true;
				}
			}
		} else {
			// Must be integer
			if((optarg[i]<'0' || optarg[i]>'9') && optarg[i]!='-') return 1;
		}
		// Only one minus is allowed
		if(optarg[i]=='-')
		{
			if(minus) return 3;
			else minus = true;
		}
	}

	// Is it in range?
	double val=atof(optarg);
	if(val<min || val>max) return 2;
	
	return 0;
}

const char *argp_program_version = VERSION;
const char *argp_program_bug_address = "https://github.com/Efenstor/Rolling-Shutter-Video-Stabilization/issues";

static char doc[] = "\nRolling Shutter Video Stabilization v" VERSION "\n"
"Algorithms and the OpenCV implementation (C) 2014 Nick Stupich.\n";

static char args_doc[] = "-i <input_file> -o <output_file>";

static struct argp_option options[] = {
	{0,				'h',	0,					OPTION_HIDDEN,	0, 0},
	{0,				'?',	0,					OPTION_HIDDEN,	0, 0},
	{"input",		'i',	"file_name",		0, "Input video file", 0},
	{"output",		'o',	"file_name",		0, "Output video file", 0},
	{"nocrop",		'n',	0,					0, "Do not crop output", 1},
	{"ssmooth",		's',	"float 0..1",		0, "Stabilization smoothness. Default=" JELLO_DECAY_S, 1},
	{"csmooth",		'c',	"float 0..1",		0, "Adaptive crop smoothness. Default=" CROP_SMOOTH_S, 1},
	{"zoom",		'z',	"float .1..100",	0, "Additional zoom. Default=" ZOOM_S, 1},
	{"winsize",		'w',	"1..100000",		0, "Search window size. Default=" WIN_SIZE_S, 1},
//	{"pass",		'p',	"1 or 2",			0, "Do only selected processing pass", 0},
//	{"method",		'm',	"1-7",				0, "Processing method (see the list below)", 1},
	{"threads",		500,	"-1 or >0",			0, "Number of threads to use. Default=-1 (auto)", 2},
	{"tcols",		600,	"0..1000",			0, "Tracker corner columns. Default=" CORNER_COLS_S, 2},
	{"trows",		601,	"0..1000",			0, "Tracker corner rows. Default=" CORNER_ROWS_S, 2},
	{"corners",		602,	"1..100000",		0, "Max number of tracker corners. Default=" NUM_CORNERS_S, 2},
	{"qlevel",		603,	"float 0..1",		0, "Tracker quality level. Default=" QLEVEL_S, 2},
	{"warnings",	501,	0,					0, "Show all warnings/errors", 2},
/*	{0,				0,		0,					OPTION_DOC, "Processing methods:\n"
		"1 = JelloComplex2 (default, best)\n"
		"2 = JelloComplex1\n"
		"3 = JelloTransform2\n"
		"4 = JelloTransform1\n"
		"5 = FullFrameTransform2 (for debug use)\n"
		"6 = FullFrameTransform (for debug use)\n"
		"7 = NullTransform (for debug use)", 0},*/
	{0, 0, 0, 0, 0, 0}
};

static error_t parse_opt (int key, char *arg, struct argp_state *state)
{
	arguments *args = (arguments*)state->input;
	
	switch (key)
    {
		case 'i':
			// Input file name
			args->inFileName = arg;
			break;
			
		case 'o':
			// Output file name
			args->outFileName = arg;
			break;
			
		case 'p':
			// Pass
			if(checkNumberArg(arg, 1, 2, false)) {
				printf("Pass should be either 1 or 2.\n");
				exit(1);
			}
			args->pass = atoi(arg);
			break;
			
		case 'm':
			// Method
			if(checkNumberArg(arg, 1, 7, false)) {
				printf("Method should be a number from 1 to 7.\n");
				exit(1);
			}
			args->method = atoi(arg);
			break;
			
		case 'n':
			// No crop
			args->noCrop = true;
			break;
			
		case 'a':
			// Adaptive crop smoothness
			if(checkNumberArg(arg, 0, 1, true)) {
				printf("Crop smoothness should be a floating-point number from 0 to 1.\n");
				exit(1);
			}
			args->cSmooth = atof(arg);
			break;
			
		case 's':
			// Stabilization smoothness (jello decay)
			if(checkNumberArg(arg, 0, 1, true)) {
				printf("Stabilization smoothness should be a floating-point number from 0 to 1.\n");
				exit(1);
			}
			args->jelloDecay = atof(arg);
			break;

		case 'w':
			// Search window size
			if(checkNumberArg(arg, 1, 100000, false)) {
				printf("Search window size should be from 1 to 100000.\n");
				exit(1);
			}
			args->winSize = atoi(arg);
			break;

		case 'z':
			// Zoom
			if(checkNumberArg(arg, .1, 100, true)) {
				printf("Zoom should be a floating-point number from .1 to 100.\n");
				exit(1);
			}
			args->zoom = atof(arg);
			break;

		case 500:
			// Threads
			if(checkNumberArg(arg, -1, INT_MAX, false)) {
				printf("Number of threads is either -1 (auto) or more than 1.\n");
				exit(1);
			}
			args->threads = atoi(arg);
			break;

		case 501:
			// Show warnings
			args->warnings = true;
			break;
			
		case 600:
			// Corner columns
			if(checkNumberArg(arg, 1, 1000, false)) {
				printf("Number of corner columns should be from 1 to 1000.\n");
				exit(1);
			}
			args->cornerCols = atoi(arg);
			break;

		case 601:
			// Corner rows
			if(checkNumberArg(arg, 1, 1000, false)) {
				printf("Number of corner rows should be from 1 to 1000.\n");
				exit(1);
			}
			args->cornerRows = atoi(arg);
			break;

		case 602:
			// Corners
			if(checkNumberArg(arg, 1, 100000, false)) {
				printf("Number of corners should be from 1 to 100000.\n");
				exit(1);
			}
			args->corners = atoi(arg);
			break;

		case 603:
			// Quality level
			if(checkNumberArg(arg, 0, 1, true)) {
				printf("Quality level should be a floating-point number from 0 to 1.\n");
				exit(1);
			}
			args->qualityLevel = atof(arg);
			break;

		case 'h':
		case '?':
			// Show full help
			argp_state_help(state, state->out_stream, ARGP_HELP_STD_HELP);
			break;

		case ARGP_KEY_SUCCESS:
			if(!args->inFileName || !args->outFileName) {
				// Mandatory parameters are not specified
				printf("Please specify both the input (-i) and output (-o) file name.\n");
				exit(1);
			}
			break;

		case ARGP_KEY_NO_ARGS:
			if(state->argc < 2)
			{
				// No arguments, show usage help
				argp_usage(state);
			}
			break;

		case ARGP_KEY_ARG:
			if (state->arg_num > 1)
			{
				// More than one argument, show usage help
				argp_usage(state);
			}
			break;
			
		default:
			return ARGP_ERR_UNKNOWN;
    }
	return 0;
}

static struct argp argp = { options, parse_opt, args_doc, doc, 0, 0, 0 };

int main(int argc, char* argv[])
{
	// Defaults
	args.inFileName = NULL;
	args.outFileName = NULL;
	args.pass = 0;
	args.method = METHOD;
	args.threads = -1;
	args.warnings = false;
	args.winSize = WIN_SIZE;
	args.corners = NUM_CORNERS;
	args.cornerCols = CORNER_COLS;
	args.cornerRows = CORNER_ROWS;
	args.noCrop = false;
	args.cSmooth = CROP_SMOOTH;
	args.jelloDecay = JELLO_DECAY;
	args.zoom = ZOOM;
	args.qualityLevel = QLEVEL;

	// Parse arguments
	if(argp_parse(&argp, argc, argv, 0, 0, &args)) return 1;

	// Threads
	if(args.threads<1)
	{
		// Get the number of system cores
		args.threads = std::thread::hardware_concurrency();
		if(args.threads<1) {
			printf("Cannot acquire the number of system cores. Using single-thread processing.\n");
			printf("Set the number of threads manually using the --threads option.\n");
			args.threads = 1;
		}
	}
	printf("threads: %d\n", args.threads);
	cv::setNumThreads(args.threads);
	cv::setUseOptimized(true);

	// Do processing
	switch(args.method)
	{
		case 1:
			evalTransformStream<JelloComplex2>(args.inFileName, args.outFileName);
			//evalTransformPass2<JelloComplex2>(args.inFileName, args.outFileName, tJelloComplex2);
			//evalTransform<JelloComplex2>(args.inFileName, args.outFileName);
			break;

		case 2:
			evalTransform<JelloComplex1>(args.inFileName, args.outFileName);
			break;

		case 3:
			evalTransform<JelloTransform2>(args.inFileName, args.outFileName);
			break;

		case 4:
			evalTransform<JelloTransform1>(args.inFileName, args.outFileName);
			break;

		case 5:
			evalTransform<FullFrameTransform2>(args.inFileName, args.outFileName);
			break;

		case 6:
			evalTransform<FullFrameTransform>(args.inFileName, args.outFileName);
			break;

		case 7:
			evalTransform<NullTransform>(args.inFileName, args.outFileName);
			break;

	}
	
	/*switch(args.debugOpt)
	{
		case 1: plotCornersOnColor(args.inFileName); break;
	    case 2: testPointExtraction(args.inFileName); break;
		case 3: chiSquaredRandomBenchmark(); break;
	}*/

	return 0;
}
