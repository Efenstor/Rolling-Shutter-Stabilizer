#ifndef SETTINGS_H
#define SETTINGS_H

// _S are string representations for the help message

#define JELLO_DECAY						.95
#define JELLO_DECAY_S					".95"

#define WIN_SIZE						21
#define WIN_SIZE_S						"21"

#define NUM_CORNERS						1000
#define NUM_CORNERS_S					"1000"

#define CORNER_COLS						20
#define CORNER_COLS_S					"20"

#define CORNER_ROWS						15
#define CORNER_ROWS_S					"15"

#define CROP_SMOOTH						.9
#define CROP_SMOOTH_S					".9"

#define ZOOM							1.1
#define ZOOM_S							"1.1"

#define QLEVEL							.02
#define QLEVEL_S						".02"

#define METHOD							1

#define FPS_AFTER						5	// Seconds


// Old methods

#define	TRANSLATION_DECAY				0.5
#define ROTATION_DECAY					0.5

#define SHOW_CORNERS					0

#define SVD_PRUNE_MAX_DIST				2.0

#define SVD_WEIGHT_FUNC3(d)				(exp(-pow((d)/40., 2)))
#define SVD_ROWS						41

#define DO_CORNER_SUBPIX				1

//#define	MAX_FRAMES					10

#define REMOVE_GAUSSIAN_WEIGHT_TAILS	0

#endif
