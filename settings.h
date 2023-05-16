#ifndef SETTINGS_H
#define SETTINGS_H

#define	TRANSLATION_DECAY				0.5   //1.0
#define ROTATION_DECAY					0.5   //1.0
#define JELLO_DECAY						0.95   //0.95

#define SHOW_CORNERS					0   //0

#define SVD_PRUNE_MAX_DIST				2.0   //2.0

#define WIN_SIZE						21   //21
#define NUM_CORNERS						1000   //1000

#define SVD_WEIGHT_FUNC3(d)				(exp(-pow((d)/40., 2)))   //(exp(-pow((d)/40., 2)))
#define SVD_ROWS						41   //41

#define DO_CORNER_SUBPIX				1   //1

//#define	MAX_FRAMES					10

#define REMOVE_GAUSSIAN_WEIGHT_TAILS	0   //0

#endif
