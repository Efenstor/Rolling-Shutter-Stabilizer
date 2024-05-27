Rolling Shutter Video Stabilizer
================================

**Eternal alpha version, so be ready for some crashtastic galore.**

### Summary:

A Linux command line program for stabilizing shaky video clips. Heavily based upon the code by [Nick Stupich](https://github.com/NickStupich). Very effectively eliminates the infamous "jello" effect caused by rolling shutter in CMOS sensors. Uses OpenCV for processing but currently all calculations are done in CPU, so don't expect much performance.

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

### Methods (-m):

* 1 = JelloComplex2 (default)
* 2 = JelloComplex1
* 3 = JelloTransform2
* 4 = JelloTransform1
* 5 = FullFrameTransform2

Generally 1 is the most advanced and 5 is the simplest. "Jello" methods work best for hand-held scenes shot from a static point of view. Scenes with complex parallax movements or lots of monotonic textures, such as sky, may not be fixable with any of the advanced methods, in this case you have to resort to *FullFrameTransform2*. Also try playing with the *-r* (*-\-errthr*) parameter, set it to something from .01 to .1.

For adjusting any parameters the test mode (*-\-test*) may be very helpful: it does not process the image but shows the motion vectors detected instead.

### Output codec:

The default output codec is H264. OpenCV 4.5 does not allow setting output encoder bitrate or quality, it is hardcoded to some "reasonable" value (~20 Mbps for 4K H264). Therefore, if you set the quality (*-q*) or bitrate (*-b*) parameters then the GStreamer backend will be used instead of the default (if OpenCV was compiled with the GStreamer support). The output picture produced in this case may suffer from some imperfections, such as oversaturation - the solution to which is yet to be found.

To specify the encoder (*-c*) you need either to specify its FOURCC code (the built-in OpenCV encoder will be used in this case) or name of a GStreamer encoder plugin. In the latter case you have to specify either quality (*-q*) or bitrate (*-b*) parameters. If you specify *-q* or *-b* without *-c* then the default GStreamer encoder will be used (x264enc). If a GStreamer encoder wants neither quality or bitrate specified, just set any of these parameters to -1 (e.g. *-b-1* or *-q-1*).

The list of FOURCC codes: https://www.free-codecs.com/guides/fourcc.htm

The list of GStreamer plugins: https://gstreamer.freedesktop.org/documentation/plugins_doc.html
