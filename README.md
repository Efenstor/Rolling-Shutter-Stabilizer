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

* OpenCV 4.5 or later
* Armadillo 10 or later (linear algebra library)
* g++ 10 or later

### Compilation:

make

### Basic usage:

*rsvs -i <input_file> -o <output_file>*

Run *rsvs -h* for full help.

### Tweaking: ###

Some clips, such as those with monotonic areas (e.g. sky) or fast movements may require additional tweaking:


> *-w* or *\--winsize* (default = 50): Try increasing this for scenes with fast panning movements.

> *\--cols* (default = 20) and *\--rows* (default = 15): Try increasing these values for scenes with complex motion or decreasing for scenes with monotonic textures. Be aware that if these values are set too high the algorithm almost gets stuck, that means you have to roll back a little bit.
