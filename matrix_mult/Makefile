EXE=matrix_prod
GXX=g++

CXXFLAGS=-g -Wall -Wextra -fpermissive -std=gnu++17 -Werror -O3 -ftree-vectorize -fstack-protector-strong
CPPFLAGS=-DCL_TARGET_OPENCL_VERSION=220 -DGROUPS=10
LDFLAGS=-lOpenCL -lnvidia-opencl

%.o:%.cl

run: all
	./$(EXE)

all: ${EXE}

clean:
	rm -rf ${EXE} ${EXE}.o
