#include "opencv2/opencv.hpp"
#include "videocl.hpp"
#include <fstream>
#include <iostream> // for standard I/O
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

using namespace cv;
using namespace std;

void check() {
    cl_init();
    srand(42);

    uint8_t arr1[16];
    uint8_t arr2[16] = {0};
    for (int i = 0; i < 16; i++) {
        arr1[i] = rand();
    }

    cl_mem buffer1 = cl_getmem(16);
    cl_mem buffer2 = cl_getmem(16);

    cl_memwrite(arr1, buffer1, 16);
    cl_Scharr(buffer1, buffer2, 4, 4, true);
    cl_memread(buffer2, arr2, 16);

    Mat mat(Size(4, 4), CV_8UC1, arr1);
    Mat out;

    Scharr(mat, out, CV_8U, 1, 0, 1, 0, BORDER_DEFAULT);

    cout << "input" << endl;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            cout << (int)mat.at<uint8_t>(i, j) << ' ';
        }
        cout << endl;
    }

    cout << "output cl" << endl;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            cout << (int)arr2[i * 4 + j] << ' ';
        }
        cout << endl;
    }

    cout << "output opencv" << endl;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            cout << (int)out.at<uint8_t>(i, j) << ' ';
        }
        cout << endl;
    }

    for (int i = 0; i < 16; i++) {
        if (out.at<uint8_t>(i / 4, i % 4) == arr2[i]) {
            cout << "erreur à la position " << i << endl;
            exit(1);
        }
    }
    exit(0);
}

int main(int, char **) {
    check();

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
    uint8_t *buffedge = new uint8_t[S.width * S.height];
#endif

    time_t start, end;
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
        cl_memwrite(grayframe.data, graybuff, S.width * S.height);
        cl_Scharr(graybuff, edge_x, S.width, S.height, true);
        cl_Scharr(graybuff, edge_y, S.width, S.height, false);
        cl_average(edge_x, edge_y, edgebuff, S.width, S.height);
        cl_memread(edgebuff, buffedge, S.width * S.height);
        Mat edge(S, CV_8UC1, buffedge);
#endif

        if (count == 150) {
            cout << "image 150:" << endl;
            for (int i = 0; i < 30; i++) {
                for (int j = 0; j < 30; j++) {
                    cout << (int)edge.at<uint8_t>(i, j) << ' ';
                }
                cout << endl;
            }
        }

        // if (count == 150) {
        //     for (int i = 0; i < 20; i++) {
        //         for (int j = 0; j < 20; j++) {
        //             cout << (int)edge.at<uint8_t>(i, j) << ' ';
        //         }
        //         cout << endl;
        //     }
        // }

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

    outputVideo.release();
    camera.release();
    printf("FPS %.2lf .\n", 299.0 / tot);

#ifndef OPENCV
    cl_releasemem(graybuff);
    cl_releasemem(edgebuff);
    cl_releasemem(edge_x);
    cl_releasemem(edge_y);

    cl_clean();
    delete[] buffedge;
#endif

    return EXIT_SUCCESS;
}
