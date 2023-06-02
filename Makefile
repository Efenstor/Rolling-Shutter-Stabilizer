PROGRAM=rsvs

HFILES = coreFuncs.h EvalTransformStream.hpp FullFrameTransform.h FullFrameTransform2.h ITransform.h JelloComplex1.h JelloComplex2.h jelloTransform1.h jelloTransform2.h nullTransform.h settings.h structures.h svd.h

OFILES = coreFuncs.o FullFrameTransform.o FullFrameTransform2.o ITransform.o JelloComplex1.o JelloComplex2.o jelloTransform1.o jelloTransform2.o nullTransform.o main.o svd.o

CFLAGS = -g -I /usr/include/opencv4 -I /usr/local/include/opencv4 -g -Wall -Wextra -Wpedantic -std=c++17 -O3
LDFLAGS = -lm -lpthread -lopencv_core -lopencv_imgproc -lopencv_calib3d -lopencv_video -lopencv_features2d -lopencv_ml -lopencv_highgui -lopencv_objdetect -lopencv_imgcodecs -lopencv_videoio -larmadillo

.PHONY: all
all: $(PROGRAM)

.PHONY: run
run: $(PROGRAM)
	./$(PROGRAM)

%.o: %.cpp $(HFILES)
	g++ $(CFLAGS) -c $< -o $@

$(PROGRAM): $(OFILES)
	g++ $(OFILES) $(LDFLAGS) -o $@

.PHONY: clean
clean:
	-rm -f *.o $(program)
