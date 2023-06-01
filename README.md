Rolling-Shutter-Video-Stabilization
===================================

**Work in progress! Use at your own risk.**

### Summary:

A Linux command line program for stabilizing shaky video clips. Very effectively eliminates the infamous "jello" effect caused by rolling shutter in CMOS sensors. Uses OpenCV for processing but currently all calculations are done in CPU, so don't expect much performance.

### Licensing problem:

99% of the processing code is done by Nick Stupich for his master's work. The code was published without any license, therefore, until the licensing issue is clarified, all the original code Copyright (C) Nick Stupich, All Rights Reserved.

All that you are legally allowed to do with such code (probably excluding later commits made by 3rd parties) is to study it and, if it is not contradictory to the laws of your country, compile and use it for your own personal purposes.

Read more about the problem [here](https://www.gnu.org/licenses/license-list.html#NoLicense).

### Library requirements:

* *OpenCV 4.5* or later
* *Armadillo 10* or later (linear algebra library)
* *g++ 10* or later

### Compilation:

*make*

### Basic usage:

*rsvs -i <input_file> -o <output_file>*

Run *rsvs -h* for full help.

### Notes:

By default OpenCV does not allow setting output encoder bitrate or quality, it is hardcoded to some "reasonable" value. Therefore, if you set the quality (-q) or bitrate (-b) parameters then GStreamer will be attempted to be used instead of the default built-in OpenCV encoder. The output picture produced in this case may suffer from slight imperfections, such as oversaturation, the solution to which is yet to be found.
