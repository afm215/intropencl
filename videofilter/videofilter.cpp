#include "opencv2/opencv.hpp"
#include "videocl.hpp"
#include <fstream>
#include <iostream> // for standard I/O
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

using namespace cv;
using namespace std;

int main(int, char **) {
    VideoCapture camera("./bourne.mp4");
    if (!camera.isOpened()) // check if we succeeded
        return -1;

    const string NAME = "./output.avi"; // Form the new name with container
    int ex = static_cast<int>(VideoWriter::fourcc('M', 'J', 'P', 'G'));
    Size S = Size((int)camera.get(CAP_PROP_FRAME_WIDTH), // Acquire input size
                  (int)camera.get(CAP_PROP_FRAME_HEIGHT));
    // Size S =Size(1280,720);
    cout << "SIZE:" << S << endl;

    VideoWriter outputVideo; // Open the output
    outputVideo.open(NAME, ex, 25, S, true);

    if (!outputVideo.isOpened()) {
        cout << "Could not open the output video for write: " << NAME << endl;
        return -1;
    }

#ifndef OPENCV
    cl_init();
    cl_mem graybuff = cl_getmem(S.width * S.height);
    cl_mem edgebuff = cl_getmem(S.width * S.height);
    cl_mem edge_x = cl_getmem(S.width * S.height);
    cl_mem edge_y = cl_getmem(S.width * S.height);
    uint8_t *buffedge;
#ifndef MAPPED
    buffedge = new uint8_t[S.width * S.height];
#endif
#endif
    struct timespec start, end;
    double diff, tot = 0;
    int count = 0;
#ifdef SHOW
    const char *windowName = "filter"; // Name shown in the GUI window.
    namedWindow(windowName); // Resizable window, might not work on Windows.
#endif

    while (true) {
        Mat cameraFrame, displayframe;
        count = count + 1;
        if (count > 299)
            break;
        camera >> cameraFrame;
        Mat grayframe, edge_inv;

        cvtColor(cameraFrame, grayframe, COLOR_BGR2GRAY);

        clock_gettime(CLOCK_REALTIME, &start);

#ifdef OPENCV
        GaussianBlur(grayframe, grayframe, Size(3, 3), 0, 0);
        GaussianBlur(grayframe, grayframe, Size(3, 3), 0, 0);
        GaussianBlur(grayframe, grayframe, Size(3, 3), 0, 0);

        Mat edge_x, edge_y, edge;
        Scharr(grayframe, edge_x, CV_8U, 0, 1, 1, 0, BORDER_CONSTANT);
        Scharr(grayframe, edge_y, CV_8U, 1, 0, 1, 0, BORDER_CONSTANT);
        addWeighted(edge_x, 0.5, edge_y, 0.5, 0, edge);
#else
        cl_memwrite(grayframe.data, graybuff, S.width * S.height);
        cl_blur(graybuff, edge_x, S.width, S.height);
        cl_blur(edge_x, edge_y, S.width, S.height);
        cl_blur(edge_y, graybuff, S.width, S.height);

        cl_Scharr(graybuff, edge_x, S.width, S.height, true);
        cl_Scharr(graybuff, edge_y, S.width, S.height, false);

        cl_average(edge_x, edge_y, edgebuff, S.width, S.height);

#ifdef MAPPED
        grayframe.data = (uint8_t *)cl_map_mem(graybuff, S.width * S.height);
        buffedge = (uint8_t *)cl_map_mem(edgebuff, S.width * S.height);
#else
        cl_memread(graybuff, grayframe.data, S.width * S.height);
        cl_memread(edgebuff, buffedge, S.width * S.height);
#endif

        Mat edge(S, CV_8UC1, buffedge);
#endif

        threshold(edge, edge, 80, 255, THRESH_BINARY_INV);
        clock_gettime(CLOCK_REALTIME, &end);
        cvtColor(edge, edge_inv, COLOR_GRAY2BGR);
        // Clear the output image to black, so that the cartoon line drawings
        // will be black (ie: not drawn).
        memset((char *)displayframe.data, 0,
               displayframe.step * displayframe.rows);
        grayframe.copyTo(displayframe, edge);
        cvtColor(displayframe, displayframe, COLOR_GRAY2BGR);
        outputVideo << displayframe;

#ifdef MAPPED
        cl_unmap_mem(graybuff, grayframe.data);
        cl_unmap_mem(edgebuff, buffedge);
#endif
#ifdef SHOW
        imshow(windowName, displayframe);
#endif
        diff = (double)(end.tv_sec - start.tv_sec) +
               (double)(end.tv_nsec - start.tv_nsec) / 1000000000.;
        tot += diff;
    }

    outputVideo.release();
    camera.release();
    printf("%lf\n", tot);
    printf("FPS %.2lf .\n", 299.0 / tot);

#ifndef OPENCV
    cl_releasemem(graybuff);
    cl_releasemem(edgebuff);
    cl_releasemem(edge_x);
    cl_releasemem(edge_y);
#ifndef MAPPED
    delete[] buffedge;
#endif
    cl_clean();
#endif

    return EXIT_SUCCESS;
}
