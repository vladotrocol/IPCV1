#include <stdio.h>
#include <iostream>

#include <opencv2/opencv.hpp>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"

using namespace cv;
using namespace std;

/** Global variables */
//Haar Globals
String cascade_name = "dartcascade.xml";
CascadeClassifier cascade;

//Convert int to string
string int_to_string(int i){
    std::ostringstream oss;
    oss << i;
    string s = oss.str();
    return s;
};

int main( int argc, const char** argv )
{ 

  for(int i=0;i<12;i++){
    
    string s  = int_to_string(i); //current image index

    //Read source image
    Mat frame = imread((string)"dart"+s+(string)".jpg", CV_LOAD_IMAGE_COLOR);

    if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
    
    //Show source
    imshow((string)"dart",frame);
    
    //Apply detection and display results
    //detectAndDisplay(frame ,s);
    
    //Hold the windows open
    waitKey();
  }

  return 0;
};