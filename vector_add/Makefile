EXE=vector_add
GXX=g++

CXXFLAGS=-g -Wall -Wextra -fpermissive -std=gnu++17 -Werror -O3 -ftree-vectorize -fstack-protector-strong -Wno-deprecated-declarations
CPPFLAGS=-DCL_TARGET_OPENCL_VERSION=220 -DMAPPED
LDFLAGS=-lOpenCL -lnvidia-opencl

%.o: %.cl

run: all
	./$(EXE)

all: $(EXE)

clean:
	rm -rf $(EXE) $(EXE).o
