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

//mutex variables
pthread_mutex_t my_mutex = PTHREAD_MUTEX_INITIALIZER;
int sharedInteger = 0;

//FPS variables
int fps = 0;
int frames = 0;
chrono::steady_clock::time_point beginFrameTimer;
chrono::steady_clock::time_point endFrameTimer;

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
 
int sched_setattr(pid_t pid,
		  const struct sched_attr *attr,
		  unsigned int flags) {
	return syscall(__NR_sched_setattr, pid, attr, flags);
 }

 int sched_getattr(pid_t pid,
		  struct sched_attr *attr,
		  unsigned int size,
		  unsigned int flags) {
	return syscall(__NR_sched_getattr, pid, attr, size, flags);
 }

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

void *GetFrame(void *data)
{
    struct sched_attr attr = SetupThread(data);
    Mat myImage_hsv; // Declaring a matrix to load modified image into
	Mat myImage_resized; // Declaring a matrix to load resized image into
	Mat myImage_blurred; // Declaring a matrix to load blurred image into
	Mat myImage_masked; // Declaring a matrix to load color-filtered image into
	Mat myImage_eroded; // Declaring a matrix to load eroded image into
	Mat myImage_dilated; // Declaring a matrix to load dilated image into
	Mat myImage_results; // Declaring matrix to draw final results on
	Mat element;

printf("Get frame setup\n");
    while (!done)
    {
        if (pthread_mutex_trylock(&my_mutex))
        {
            if (sharedInteger == 0)
            {
                // Create a structuring element (SE)
                int morph_size = 2;
                
                if (!cap.isOpened()){  //This section prompt an error message if no video stream is found//
                    cout << "No video stream detected" << endl;
                    system("pause");
                }
                
                cap.read(myImage);
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
                
                sharedInteger = 1;
                pthread_mutex_unlock(&my_mutex);

                sched_yield();
            }
            else //mutex was locked but was out of order
                pthread_mutex_unlock(&my_mutex);
        }
    }
    return NULL;
}

void *FindContours(void *data)
{
    struct sched_attr attr = SetupThread(data);
    while (!done)
    {
	    //printf("Find contours while...\n");
        if (pthread_mutex_trylock(&my_mutex))
        {
            if (sharedInteger == 1)
            {
                vector<Vec4i> hierarchy;
                findContours(prep_image, detected_contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

                sharedInteger = 2;
                pthread_mutex_unlock(&my_mutex);

                sched_yield();
            }
            else //mutex was locked but was out of order
                pthread_mutex_unlock(&my_mutex);
        }
    }
    return NULL;
}

void *DrawEnclosingCircle(void *data)
{
    struct sched_attr attr = SetupThread(data);
    while (!done)
    {
        if (pthread_mutex_trylock(&my_mutex))
        {
            if (sharedInteger == 2)
            {
                // Initialize variable for contour area
                double largest_area = 0;
                double area = 0;
                vector<point> *contour_index;
                int contrail_index = 0;
                Point2f center;
                float r;	
                
                // Only proceed if at least one contour was found
                if (detected_contours.size() > 0) 
                {
                    // Find the largest detected contour
                    for (auto iter = detected_contours.begin(), end = detected_contours.end(); iter != end; ++iter)
                    {
                        // Calculate area one contour at a time
                        area = contourArea(*iter, false);
                        // Determine if the detected contour is larger than the previous
                        if (area > largest_area) 
                        {
                            largest_area = area;
                            contour_index = iter;
                        }		
                    }      
                    
                    // Calculate a bouding circle for the detected contour
                    const vector<Point> cnt = *contour_index;
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

                sharedInteger = 3;
                pthread_mutex_unlock(&my_mutex);
                sched_yield();
            }
            else //mutex was locked but was out of order
                pthread_mutex_unlock(&my_mutex);
        }
    }
    return NULL;
}

void *DrawTrackedPoints(void *data)
{
    struct sched_attr attr = SetupThread(data);
    while (!done)
    {
	    if (pthread_mutex_trylock(&my_mutex))
        {
            if (sharedInteger == 3)
            {
                int i = 0;
                if (contrails.size() > 0)
                {
                    
                    for (auto iter = contrails.begin() + 1, end = contrails.end(); iter != end; ++iter)
                    {
                        line(myImage, *(iter - 1), *iter, Scalar(0, 255, 0), 2, LINE_8);
                        ++i;
                    }
                }
                sharedInteger = 4;
                pthread_mutex_unlock(&my_mutex);
                sched_yield();
            } 
            else //mutex was locked but was out of order
                pthread_mutex_unlock(&my_mutex);
        }
    }
    return NULL;
}

void *DisplayFrame(void *data)
{
    struct sched_attr attr = SetupThread(data);
    while (!done)
    {
	    if (pthread_mutex_trylock(&my_mutex))
        {
             if (sharedInteger == 4)
            {
                ++frames;
                endFrameTimer = chrono::steady_clock::now();
                chrono::duration<double> elapsed_seconds = endFrameTimer - beginFrameTimer;
                
                if (elapsed_seconds.count() > 1) //second has passed
                {
                    fps = frames; //record number of frames the past second
                    beginFrameTimer = chrono::steady_clock::now();  //reset start of second
                    frames = 0; // reset frame in second
                }
                //draw fps to frame
                putText(myImage, to_string(fps), cvPoint(30,30), 0, 0.8, cvScalar(200,200,250),1 , LINE_AA);
                imshow("Video Player", myImage); // Draw frame
                
                sharedInteger = 0;
                pthread_mutex_unlock(&my_mutex);
                sched_yield();
            }
            else //mutex was locked but was out of order
                pthread_mutex_unlock(&my_mutex);
        }
    }
    return NULL;
}

int main (int argc, char **argv)
{
    pthread_t thread[NUM_THREADS];
    struct sched_attr task_attr[5] =
    {
        {0, SCHED_DEADLINE, 0, 0, 0, 34, 35, 83},
        {0, SCHED_DEADLINE, 0, 0, 0, 20, 55, 83},
        {0, SCHED_DEADLINE, 0, 0, 0, 10, 66, 83},
        {0, SCHED_DEADLINE, 0, 0, 0, 10, 77, 83},
        {0, SCHED_DEADLINE, 0, 0, 0, 5, 83, 83}    
    };

    namedWindow("Video Player"); //Declaring the video to show the video//
    
    beginFrameTimer = chrono::steady_clock::now();
    
    pthread_create(&thread[0], NULL, GetFrame, (void*)&task_attr[0]);
    pthread_create(&thread[1], NULL, FindContours, (void*)&task_attr[1]);
    pthread_create(&thread[2], NULL, DrawEnclosingCircle, (void*)&task_attr[2]);
    pthread_create(&thread[3], NULL, DrawTrackedPoints, (void*)&task_attr[3]);
    pthread_create(&thread[4], NULL, DisplayFrame, (void*)&task_attr[4]);

    sleep(40);
    done = 1;
    for (int i = 0; i < NUM_THREADS; ++i) 
        pthread_join(thread[i], NULL);
    return 0;
}
