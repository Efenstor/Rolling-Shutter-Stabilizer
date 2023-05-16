Rolling-Shutter-Video-Stabilization
===================================

### Summary:

A Linux command line program for stabilizing shaky video clips. Very effectively eliminates the infamous "jello" effect caused by CMOS sensors. Although not very fast, does not require GPU.

**Memory overflow warning:** Currently video is processed in memory in uncompressed form, so be sure to have lots of RAM and avoid processing long clips.

### Licensing problem:

Most of the code comes from the Nick Stupich's master's work, which was published without any license, therefore, until the licensing issue is clarified, all the original code Copyright (C) Nick Stupich, All Rights Reserved.

All that you are legally allowed to do with any of such code, excluding later commits made by 3rd parties, is to study it and, if it is not contradictory to the laws of your country, compile and use it for your own personal purposes.

Read more about the problem [here](https://www.gnu.org/licenses/license-list.html#NoLicense). 

### Library requirements:

* OpenCV 4.5 or later
* Armadillo 10 or later (linear algebra library)
* g++ 10 or later

### Compilation:

make

### Usage:

rsvs <input_video_file> <output_video_file>
