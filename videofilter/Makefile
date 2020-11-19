EXE=videofilter
CVCFLAGS=`pkg-config --cflags opencv4`
CVLDLIBS=`pkg-config --libs opencv4`
# DBGFLAGS= -pg
GXX=g++

LDLIBS=$(CVLDLIBS)
CXXFLAGS=-pg -g -Wall -Wextra -fpermissive -std=gnu++17 -Werror -O3 -ftree-vectorize -fstack-protector-strong
CPPFLAGS=-DCL_TARGET_OPENCL_VERSION=220 $(CVCFLAGS) -DSHOW

run: all
	./$(EXE)

all:${EXE}

%.o:%.cl

profile: run
	gprof ./$(EXE) gmon.out > prof.txt

clean:
	rm -rf ${EXE} *.o
