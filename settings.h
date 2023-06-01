#ifndef SETTINGS_H
#define SETTINGS_H

// _S are string representations for the help message

#define FOURCC                          "H264"
#define CODEC                           "x264enc"

#define DJD_SHIFT                       .3
#define DJD_SHIFT_S                     ".3"

#define DJD_LINEAR                      3
#define DJD_LINEAR_S                    "3"

#define DJD_AMOUNT                      .9
#define DJD_AMOUNT_S                    ".9"

#define WIN_SIZE                        10
#define WIN_SIZE_S                      "10"

#define NUM_CORNERS                     1000
#define NUM_CORNERS_S                   "1000"

#define CORNER_COLS                     40
#define CORNER_COLS_S                   "40"

#define CORNER_ROWS                     20
#define CORNER_ROWS_S                   "20"

#define CROP_SMOOTH                     .5
#define CROP_SMOOTH_S                   ".5"

#define ZOOM                            1.2
#define ZOOM_S                          "1.2"

#define QLEVEL                          .02
#define QLEVEL_S                        ".02"

#define MAX_LEVEL                       100
#define MAX_LEVEL_S                     "100"

#define ITER                            30
#define ITER_S                          "30"

#define EPSILON                         .1
#define EPSILON_S                       ".1"

#define EIG_THR                         0
#define EIG_THR_S                       "0"

#define METHOD                          1
#define METHOD_S                        "1"

#define FPS_AFTER                       5   // Seconds

#define TEST_MARKER_SIZE                .0015

// Old

#define JELLO_DECAY                     .85

#define TRANSLATION_DECAY               0.5
#define ROTATION_DECAY                  0.5

#define SVD_PRUNE_MAX_DIST              2.0

#define SVD_WEIGHT_FUNC3(d)             (exp(-pow((d)/40., 2)))
#define SVD_ROWS                        41

//#define DO_CORNER_SUBPIX              1

//#define   MAX_FRAMES                  10

#define REMOVE_GAUSSIAN_WEIGHT_TAILS    0

#endif
