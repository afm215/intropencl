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
    void *graybuff = cl_getmem(S.width * S.height);
    void *edge_x = cl_getmem(S.width * S.height);
    void *edge_y = cl_getmem(S.width * S.height);
#endif

    time_t start, end;
    double diff, tot = 0;
    int count = 0;
#ifdef SHOW
    const char *windowName = "filter"; // Name shown in the GUI window.
    namedWindow(windowName); // Resizable window, might not work on Windows.
#endif
    cl_init();
    while (true) {
        Mat cameraFrame, displayframe;
        count = count + 1;
        if (count > 299)
            break;
        camera >> cameraFrame;

        Mat grayframe, edge_inv;

        cvtColor(cameraFrame, grayframe, COLOR_BGR2GRAY);
        time(&start);
        GaussianBlur(grayframe, grayframe, Size(3, 3), 0, 0);
        GaussianBlur(grayframe, grayframe, Size(3, 3), 0, 0);
        GaussianBlur(grayframe, grayframe, Size(3, 3), 0, 0);

#ifdef OPENCV
        Mat edge_x, edge_y, edge;
        Scharr(grayframe, edge_x, CV_8U, 0, 1, 1, 0, BORDER_DEFAULT);
        Scharr(grayframe, edge_y, CV_8U, 1, 0, 1, 0, BORDER_DEFAULT);
        addWeighted(edge_x, 0.5, edge_y, 0.5, 0, edge);
#else
        cl_memcopy(grayframe.data, graybuff, S.width * S.height);
        ScharrCL(graybuff, edge_x, true);
        ScharrCL(graybuff, edge_y, false);
        addWeightedCL(edge_x, edge_y, edge, S.width, S.height);
        cl_memcopy();
#endif

        threshold(edge, edge, 80, 255, THRESH_BINARY_INV);
        time(&end);
        cvtColor(edge, edge_inv, COLOR_GRAY2BGR);
        // Clear the output image to black, so that the cartoon line drawings
        // will be black (ie: not drawn).
        memset((char *)displayframe.data, 0,
               displayframe.step * displayframe.rows);
        grayframe.copyTo(displayframe, edge);
        cvtColor(displayframe, displayframe, COLOR_GRAY2BGR);
        outputVideo << displayframe;
#ifdef SHOW
        imshow(windowName, displayframe);
#endif
        diff = difftime(end, start);
        tot += diff;
    }
    cl_clean();
    outputVideo.release();
    camera.release();
    printf("FPS %.2lf .\n", 299.0 / tot);

    return EXIT_SUCCESS;
}
