#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <linux/unistd.h>
#include <linux/kernel.h>
#include <linux/types.h>
#include <sys/syscall.h>
#include <pthread.h>
#include <time.h>
#include <signal.h>
#include <sys/shm.h>
#include <sched.h>
#include <linux/futex.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/wait.h>
// Libraries for OpenCV image processing
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>
#include <opencv2/core/types_c.h>
#include <opencv2/core/utility.hpp>

using namespace std;
using namespace cv;

#define gettid() syscall(__NR_gettid)
#define SCHED_DEADLINE	6
#define MILLI_TO_NANO 1000000
#define MY_CLOCK CLOCK_REALTIME
#define NUM_THREADS 5


//thread variables
static volatile int done;

//image variables
VideoCapture cap(0); //Declaring an object to capture stream of frames from default camera//
Mat myImage; // Declaring a matrix to load images into
vector<Point>  contrails; // Preallocate array for contrail points
Mat prep_image;
vector<vector<Point> > detected_contours;

//FPS variables
int fps = 0;
int frames = 0;
chrono::steady_clock::time_point beginFrameTimer;
chrono::steady_clock::time_point endFrameTimer;

//sched_attribute struct required by sched_deadline, but not implemented by linux
struct sched_attr {
	__u32 size;

	__u32 sched_policy;
	__u64 sched_flags;

	/* SCHED_NORMAL, SCHED_BATCH */
	__s32 sched_nice;

	/* SCHED_FIFO, SCHED_RR */
	__u32 sched_priority;;

	/* SCHED_DEADLINE (nsec) */
	__u64 sched_runtime;
	__u64 sched_deadline;
	__u64 sched_period;
 };
 
//sched_deadline set_attr system call
int sched_setattr(pid_t pid,
		  const struct sched_attr *attr,
		  unsigned int flags) {
	return syscall(__NR_sched_setattr, pid, attr, flags);
 }
//sched_deadline get_attr system call
int sched_getattr(pid_t pid,
		  struct sched_attr *attr,
		  unsigned int size,
		  unsigned int flags) {
	return syscall(__NR_sched_getattr, pid, attr, size, flags);
 }

//Called at the start of each thread to set the scheduling attributes for each thread
//sets threads to sched_deadline and sets C, D, and P of each thread (task)
struct sched_attr SetupThread(void *data)
{
    struct sched_attr attr = *((struct sched_attr*)data);
    int timesRan = 0;
    int ret;

    attr.size = sizeof(attr);
    attr.sched_runtime *= MILLI_TO_NANO;
    attr.sched_deadline *= MILLI_TO_NANO;
    attr.sched_period *= MILLI_TO_NANO;

    ret = sched_setattr(0, &attr, 0);
    if (ret < 0) {
        done = 0;
        perror("sched_setattr");
        exit(-1);
    }
    return attr;
}

//returns average in nanoseconds
long long CalcRollingAverage(long long average, long newData, int dataCount)
{
    return (average * (dataCount - 1) + (long long)newData) / (long long)dataCount;
}
//When thread dies display its average and WCET execution time
void DisplayRuntimeData(char* threadName, long long average, long WCET)
{
    cout << threadName << " : Average Execution Time = " << average/*/(float)MILLI_TO_NANO*/ << " : WCET = " << WCET/*/(float)MILLI_TO_NANO*/ << "\n";
}

#define MY_CLOCK_RES CLOCK_THREAD_CPUTIME_ID
//Task 1
void *GetFrame(void *data)
{
    //thread setup
    struct sched_attr attr = SetupThread(data);
    //OpenCV variables
    Mat myImage_hsv; // Declaring a matrix to load modified image into
    Mat myImage_resized; // Declaring a matrix to load resized image into
    Mat myImage_blurred; // Declaring a matrix to load blurred image into
    Mat myImage_masked; // Declaring a matrix to load color-filtered image into
    Mat myImage_eroded; // Declaring a matrix to load eroded image into
    Mat myImage_dilated; // Declaring a matrix to load dilated image into
    Mat myImage_results; // Declaring matrix to draw final results on
    Mat element;

    //Timing Code
    int timesRan = 0;
    long long average = 0;
    struct timespec startTime = {0,0};
    struct timespec endTime = {0,0};
    long elapsedTime = 0;
    long WCET = 0;

    //Task loop
    while (!done)
    {
        //Timing code
        clock_gettime(MY_CLOCK_RES, &startTime);
        /////////OPENCV CODE
        // Create a structuring element (SE)
        int morph_size = 2;
        
        if (!cap.isOpened()){  //This section prompt an error message if no video stream is found//
            cout << "No video stream detected" << endl;
            system("pause");
        }
        
        cap >> myImage;
        // cap.read(myImage);
        if (myImage.empty()){ //Breaking the loop if no video frame is detected//
            cout << "No frame detected" << endl;
            done = 1;
        }
        
        // Implement colorspace conversion function; convert to HSV
        cvtColor(myImage, myImage_hsv, COLOR_RGB2HSV); 
        
        // Implement Gaussian blur function
        GaussianBlur(myImage_hsv, myImage_blurred, Size(11,11),0);
        
        // Implement Color masking function (outputs a binary image)
        // Define variables for color masking (Note: the values below filter for green
        inRange(myImage_blurred, Scalar(20, 86, 6), Scalar(60, 255, 255), myImage_masked);
        
        // Implement Erosion and Dilation functions
        element = getStructuringElement(MORPH_RECT, Size(2*morph_size + 1, 2*morph_size + 1), Point(morph_size, morph_size));
        
        // For erosion
        erode(myImage_masked, myImage_eroded, element, Point(-1,-1),2);
        
        // For Dilation
        dilate(myImage_eroded, myImage_dilated, element, Point(-1,-1),2);
        
        prep_image = myImage_dilated;
        ////////////////// end OpenCV code
	    
        //Timing code
        clock_gettime(MY_CLOCK_RES, &endTime);
        ++timesRan;
        elapsedTime = endTime.tv_nsec - startTime.tv_nsec;
        if (elapsedTime > WCET)
            WCET = elapsedTime;
        average = CalcRollingAverage(average, elapsedTime, timesRan);

        //Relenquish thread remaining cputime until next period
        sched_yield();
    }
    //Timing code
    DisplayRuntimeData("GetFrame", average, WCET);
    return NULL;
}

void *FindContours(void *data)
{
    //thread setup
    struct sched_attr attr = SetupThread(data);

    //Timing code
    int timesRan = 0;
    long long average = 0;
    struct timespec startTime = {0,0};
    struct timespec endTime = {0,0};
    long elapsedTime = 0;
    long WCET = 0;

    while (!done)
    {
        //Timing code
        clock_gettime(MY_CLOCK_RES, &startTime);
	    
        /////////OPENCV CODE
        if (!prep_image.empty())
        {
            vector<Vec4i> hierarchy;
            findContours(prep_image, detected_contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        }
        //////////End OpenCV code

        //Timing code
        clock_gettime(MY_CLOCK_RES, &endTime);
        ++timesRan;
        elapsedTime = endTime.tv_nsec - startTime.tv_nsec;
        if (elapsedTime > WCET)
            WCET = elapsedTime;
        average = CalcRollingAverage(average, elapsedTime, timesRan);

        //Relenquish thread remaining cputime until next period
        sched_yield();
    }
    //Timing code
    DisplayRuntimeData("FindContours", average, WCET);
    return NULL;
}

void *DrawEnclosingCircle(void *data)
{
    //thread setup
    struct sched_attr attr = SetupThread(data);

    //Timing code
    int timesRan = 0;
    long long average = 0;
    struct timespec startTime = {0,0};
    struct timespec endTime = {0,0};
    long elapsedTime = 0;
    long WCET = 0;

    while (!done)
    {
        //Timing code
        clock_gettime(MY_CLOCK_RES, &startTime);
        
        /////////OPENCV CODE
        // Initialize variable for contour area
        double largest_area = 0;
        double area = 0;
        int contour_index = 0;
        int contrail_index = 0;
        Point2f center;
        float r;	
        
        // Only proceed if at least one contour was found
        if (detected_contours.size() > 0) 
        {
            // Find the largest detected contour
            for (int i = 0;i<detected_contours.size();i++)
            {
                // Calculate area one contour at a time
                area = contourArea(detected_contours[i], false);
                // Determine if the detected contour is larger than the previous
                if (area > largest_area) 
                {
                    largest_area = area;
                    contour_index = i;
                }		
            }      
            
            // Calculate a bouding circle for the detected contour
            const vector<Point> cnt = detected_contours[contour_index];
            if (cnt.size() > 0)
            {
                minEnclosingCircle(cnt, center, r);
                
                // Only draw the bounding circle if it meets the minimum size requirement
                // Then compute the centroid of the bounding circle and append to the running vector
                if (r>10)
                {
                    circle(myImage, center, r, Scalar(0, 0, 255), 2);
                    contrails.insert(contrails.begin(), center);
                    if (contrails.size() > 42)
                        contrails.pop_back();
                } 
            }
        }
        /////////end of OPENCV CODE

        //Timing code
        clock_gettime(MY_CLOCK_RES, &endTime);
        ++timesRan;
        elapsedTime = endTime.tv_nsec - startTime.tv_nsec;
        if (elapsedTime > WCET)
            WCET = elapsedTime;
        average = CalcRollingAverage(average, elapsedTime, timesRan);

        //Relenquish thread remaining cputime until next period
        sched_yield();
        
    }
    //Timing code
    DisplayRuntimeData("DrawEnclosingCircle", average, WCET);
    return NULL;
}

void *DrawTrackedPoints(void *data)
{
    //thread setup
    struct sched_attr attr = SetupThread(data);

    //Timing code
    int timesRan = 0;
    long long average = 0;
    struct timespec startTime = {0,0};
    struct timespec endTime = {0,0};
    long elapsedTime = 0;
    long WCET = 0;

    while (!done)
    {
        //Timing code
        clock_gettime(MY_CLOCK_RES, &startTime);
        
        /////////OPENCV CODE
        int i = 0;
        if (contrails.size() > 0)
        {
            
            for (vector<Point>::iterator iter = contrails.begin() + 1; iter != contrails.end(); ++iter)
            {
                line(myImage, *(iter - 1), *iter, Scalar(0, 255, 0), 2, LINE_8);
                ++i;
            }
        }
        /////////end of OPENCV CODE
        
        //Timing code
        clock_gettime(MY_CLOCK_RES, &endTime);
        ++timesRan;
        elapsedTime = endTime.tv_nsec - startTime.tv_nsec;
        if (elapsedTime > WCET)
            WCET = elapsedTime;
        average = CalcRollingAverage(average, elapsedTime, timesRan);

        //Relenquish thread remaining cputime until next period
        sched_yield();
    }
    //Timing code
    DisplayRuntimeData("DrawTrackedPoints", average, WCET);
    return NULL;
}

void *DisplayFrame(void *data)
{
    struct sched_attr attr = SetupThread(data);
    namedWindow("Video Player"); //Declaring the video to show the video//
    
    //Timing code
    int timesRan = 0;
    long long average = 0;
    struct timespec startTime = {0,0};
    struct timespec endTime = {0,0};
    long elapsedTime = 0;
    long WCET = 0;

    while (!done)
    {
        //Timing code
        clock_gettime(MY_CLOCK_RES, &startTime);
        
        ++frames;
        endFrameTimer = chrono::steady_clock::now();
        chrono::duration<double> elapsed_seconds = endFrameTimer - beginFrameTimer;
        
        if (elapsed_seconds.count() > 1) //second has passed
        {
            fps = frames; //record number of frames the past second
            beginFrameTimer = chrono::steady_clock::now();  //reset start of second
            frames = 0; // reset frame in second
        }
        //draw fps to frame if myImage is not empty
        if (!myImage.empty())
        {
            putText(myImage, to_string(fps), cvPoint(30,30), 0, 0.8, cvScalar(200,200,250),1 , LINE_AA);
            imshow("Video Player", myImage); // Draw frame
            	// Display results
            //imshow("Video Player", myImage); //Showing the video//
            char c = (char)waitKey(25); //Allowing 25 milliseconds frame processing time and initiating break condition//
            if (c == 27){ //If 'Esc' is entered break the loop//
                break;
            }
        }
        
        //Timing code
        clock_gettime(MY_CLOCK_RES, &endTime);
        ++timesRan;
        elapsedTime = endTime.tv_nsec - startTime.tv_nsec;
        if (elapsedTime > WCET)
            WCET = elapsedTime; // Multiply by 0.000001 to convert to milliseconds
        average = CalcRollingAverage(average, elapsedTime, timesRan);

        //Relenquish thread remaining cputime until next period
        sched_yield();
    }
    destroyAllWindows();
    //Timing code
    DisplayRuntimeData("DisplayFrame", average, WCET);
    return NULL;
}

int main (int argc, char **argv)
{
    //set opencv to run sequientially (no threading)
    setNumThreads(0);
    pthread_t thread[NUM_THREADS];
    //sched_deadline attributes for each task
    struct sched_attr task_attr[5] =
    {
        {0, SCHED_DEADLINE, 0, 0, 0, 44, 45, 83},
        {0, SCHED_DEADLINE, 0, 0, 0, 10, 56, 83},
        {0, SCHED_DEADLINE, 0, 0, 0, 5, 61, 83},
        {0, SCHED_DEADLINE, 0, 0, 0, 5, 66, 83},
        {0, SCHED_DEADLINE, 0, 0, 0, 15, 82, 83}    
    }; 

    //start frame timer for fps
    beginFrameTimer = chrono::steady_clock::now();
    //create the tasks with the appropriate attibutes
   pthread_create(&thread[0], NULL, GetFrame, (void*)&task_attr[0]);
    pthread_create(&thread[1], NULL, FindContours, (void*)&task_attr[1]);
    pthread_create(&thread[2], NULL, DrawEnclosingCircle, (void*)&task_attr[2]);
    pthread_create(&thread[3], NULL, DrawTrackedPoints, (void*)&task_attr[3]);
    pthread_create(&thread[4], NULL, DisplayFrame, (void*)&task_attr[4]);

    //main sleeps for 30 seconds
    sleep(30);
    //main triggers all task loops to finish
    done = 1;
    
    for (int i = 0; i < NUM_THREADS; ++i) 
        pthread_join(thread[i], NULL);
    cap.release();
    return 0;
}
