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

using namespace std;
using namespace cv;

arguments args;

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
		printf("%d/%d   fps: ", i, (int)(greyInput.size()-1));
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
		printf("%d/%d   fps: ", i, (int)input.size());
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
void evalTransform(char *inFileName, char *outFileName, int pass){
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
"Algorithms and the OpenCV implementation (C) 2014 Nick Stupich.\n\n"
"All processing is done in two passes, the results of the first pass are saved\n"
"into a <input_file>.pass1 file created in the same directory. To tweak\n"
"processing parameters you can skip the first pass if you already did it once.";

static char args_doc[] = "-i <input_file> -o <output_file>";

static struct argp_option options[] = {
	{0, 'h', 0, OPTION_HIDDEN, 0, 0},
	{0, '?', 0, OPTION_HIDDEN, 0, 0},
	{"input", 'i', "file_name", 0, "Input video file", 0},
	{"output", 'o', "file_name", 0, "Output video file", 0},
	{"pass", 'p', "1 or 2", 0, "Do only selected processing pass", 0},
	{"method", 'm', "1..7", 0, "Processing method (see the list below)", 1},
	{"threads", 500, "-1 or >1", 0, "Number of threads to use (default=auto)", 2},
	{"warnings", 501, 0, 0, "Show all warnings/errors", 2},
	{0, 0, 0, OPTION_DOC, "Processing methods:\n"
		"1 = JelloComplex2 (default, best)\n"
		"2 = JelloComplex1\n"
		"3 = JelloTransform2\n"
		"4 = JelloTransform1\n"
		"5 = FullFrameTransform2 (for debug use)\n"
		"6 = FullFrameTransform (for debug use)\n"
		"7 = NullTransform (for debug use)", 0},
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

static struct argp argp = { options, parse_opt, args_doc, doc };

int main(int argc, char* argv[]){

	// Defaults
	args.inFileName = NULL;
	args.outFileName = NULL;
	args.pass = 0;
	args.method = 1;
	args.threads = -1;
	args.warnings = false;

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
		case 1: evalTransform<JelloComplex2>(args.inFileName, args.outFileName, args.pass); break;
		case 2: evalTransform<JelloComplex1>(args.inFileName, args.outFileName, args.pass); break;
		case 3: evalTransform<JelloTransform2>(args.inFileName, args.outFileName, args.pass); break;
		case 4: evalTransform<JelloTransform1>(args.inFileName, args.outFileName, args.pass); break;
		case 5: evalTransform<FullFrameTransform2>(args.inFileName, args.outFileName, args.pass); break;
		case 6: evalTransform<FullFrameTransform>(args.inFileName, args.outFileName, args.pass); break;
		case 7: evalTransform<NullTransform>(args.inFileName, args.outFileName, args.pass); break;
	}
	
	/*switch(args.debugOpt)
	{
		case 1: plotCornersOnColor(args.inFileName); break;
	    case 2: testPointExtraction(args.inFileName); break;
		case 3: chiSquaredRandomBenchmark(); break;
	}*/

	return 0;
}
