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

int batch = 1;

/** Function Headers */
void detectAndDisplay(Mat frame, string s);
void haarDetect(Mat frame, string s);
void haarCircles(Mat frame, string s);
void detectLines(Mat frame, string s);
void surfDetect(Mat frame, string s);
void detectContours(Mat frame, string s);


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



/**
 * @function Hist_and_Backproj
 * @brief Callback to Trackbar
 */
void Hist_and_Backproj(int, void* )
{
  MatND hist;
  int histSize = MAX( bins, 2 );
  float hue_range[] = { 0, 180 };
  const float* ranges = { hue_range };

  /// Get the Histogram and normalize it
  calcHist( &hue, 1, 0, Mat(), hist, 1, &histSize, &ranges, true, false );
  normalize( hist, hist, 0, 255, NORM_MINMAX, -1, Mat() );

  /// Get Backprojection
  MatND backproj;
  calcBackProject( &hue, 1, 0, hist, backproj, &ranges, 1, true );

  /// Draw the backproj
  imshow( "BackProj", backproj );

  /// Draw the histogram
  int w = 400; int h = 400;
  int bin_w = cvRound( (double) w / histSize );
  Mat histImg = Mat::zeros( w, h, CV_8UC3 );

  for( int i = 0; i < bins; i ++ )
     { rectangle( histImg, Point( i*bin_w, h ), Point( (i+1)*bin_w, h - cvRound( hist.at<float>(i)*h/255.0 ) ), Scalar( 0, 0, 255 ), -1 ); }

  imshow( "Histogram", histImg );
}



//Find features
void surfDetect(Mat img_scene, string s){
  Mat img_object = imread( "dart.bmp", CV_LOAD_IMAGE_GRAYSCALE );

  //Detect the keypoints using SURF Detector
  int minHessian = 400;

  SurfFeatureDetector detector( minHessian );

  std::vector<KeyPoint> keypoints_object, keypoints_scene;

  detector.detect( img_object, keypoints_object );
  detector.detect( img_scene, keypoints_scene );

  //Calculate descriptors (feature vectors)
  SurfDescriptorExtractor extractor;

  Mat descriptors_object, descriptors_scene;

  extractor.compute( img_object, keypoints_object, descriptors_object );
  extractor.compute( img_scene, keypoints_scene, descriptors_scene );

  //Matching descriptor vectors using FLANN matcher
  FlannBasedMatcher matcher;
  std::vector< DMatch > matches;
  matcher.match( descriptors_object, descriptors_scene, matches );

  double max_dist = 0; double min_dist = 100;

  //Calculation of max and min distances between keypoints
  for( int i = 0; i < descriptors_object.rows; i++ )
  { double dist = matches[i].distance;
    if( dist < min_dist ) min_dist = dist;
    if( dist > max_dist ) max_dist = dist;
  }

  //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
  std::vector< DMatch > good_matches;

  for( int i = 0; i < descriptors_object.rows; i++ )
  { if( matches[i].distance < 3*min_dist )
     { good_matches.push_back( matches[i]); }
  }

  //-- Localize the object
  std::vector<Point2f> obj;
  std::vector<Point2f> scene;

  for( int i = 0; i < good_matches.size(); i++ )
  {
    //-- Get the keypoints from the good matches
    obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
    scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
  }

  //-- Draw keypoints ALL
  // Mat img_keypoints_2;
  //drawKeypoints( img_scene, keypoints_scene, img_keypoints_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
  // imshow("hpAll", img_keypoints_2);

  //Draw Matching Keypoints
  Mat frame_tmp = img_scene.clone();
  for( int i = 0; i < scene.size(); i++ ){
    
    //Set-up variables
    Point center(cvRound(scene[i].x), cvRound(scene[i].y));

    //Draw scene
    circle(frame_tmp, center, 5, Scalar(0,0,255), 3, 8, 0 );
  }

  //Display result
  imshow("SURF", frame_tmp);
};

//Find lines
void detectLines(Mat frame, string s){

  vector<Vec4i> lines; //Stores results

  //Preprocessing
  Mat canny_output, frame_gray;
  Canny(frame, canny_output, 50, 150, 3);
  cvtColor(canny_output, frame_gray, CV_GRAY2BGR);

  //Apply transform
  HoughLinesP(canny_output, lines, 1, CV_PI/180, 50, 80, 10 );

  //Compute average line length
  float sum;
  float average;
  for(int i=0;i<lines.size();i++){
    sum+=sqrt((lines[i][0]-lines[i][2])*(lines[i][0]-lines[i][2])+(lines[i][1]-lines[i][3])*(lines[i][1]-lines[i][3]));
  }
  average = sum/lines.size();

  //Compute points where multiple lines "intersect"
  //Elongate lines
  //Check for intersection
  //If intersection#>3/4/5 remove them and repeat

  int count;
  for(int i=0;i<lines.size();i++){
    for(int j=0;j<4;j++){

    }
  }

  for( size_t i = 0; i < lines.size(); i++ ){
    Vec4i l = lines[i];
    line(frame_gray, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);
  }
 
  imshow("detected lines", frame_gray);
};

//Find all contours
void detectContours(Mat frame, string s){
  RNG rng(12345);//random number seed

  //Stores resulting data
  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;

  //Preprocessing
  //Detect edges using canny
  Mat canny_output;
  Canny(frame, canny_output, 100, 100*2, 3);
  //Find contours
  findContours(canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

  //Draw contours
  Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
  vector<vector<Point> > contours_poly( contours.size() );
  for( int i = 0; i< contours.size(); i++ ){
    Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) ); //This is a colour
    drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
  }
  //Display result   
  imshow( "Contours", drawing );
  drawing = frame.clone();
  for( int i = 0; i< contours.size(); i++ ){
    approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
    if(contours_poly[i].size()<6&&contours_poly[i].size()>2){
      line(drawing, contours_poly[i][0], contours_poly[i][1], cvScalar(255,0,0),4);
      line(drawing, contours_poly[i][1], contours_poly[i][2], cvScalar(255,0,0),4);
      line(drawing, contours_poly[i][2], contours_poly[i][0], cvScalar(255,0,0),4);
    }
  }
  //Display result   
  imshow( "Contours2", drawing );
};

//Detect circles using haar transform
void haarCircles(Mat frame , string s){
  vector<Vec3f> circles;//Stores detected circles
  
  //Apply preprocessing
  Mat frame_gray;
  cvtColor( frame, frame_gray, CV_BGR2GRAY );
  equalizeHist( frame_gray, frame_gray );
  GaussianBlur( frame_gray, frame_gray, Size(9, 9), 2, 2 );

  // Apply the Hough Transform to find the circles
  HoughCircles( frame_gray, circles, CV_HOUGH_GRADIENT, 1, frame_gray.rows/8, 200, 50, 0, 0 );

  //Draw the detected features
  Mat frame_tmp =frame.clone(); //make a copy of the original
  
  for( int i = 0; i < circles.size(); i++ ){
    
    //Set-up variables
    Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
    int radius = cvRound(circles[i][2]);
    
    //Draw circles
    circle(frame_tmp, center, radius, Scalar(0,0,255), 3, 8, 0 );
  }

  //Display results
  imshow((string)"dartHC",frame_tmp);
};


//Apply haar features detection
void haarDetect( Mat frame , string s){
  std::vector<Rect> obj; //Stores detected objects
  
  //Apply preprocessing
  Mat frame_gray;
  cvtColor(frame, frame_gray, CV_BGR2GRAY);
  //imshow("Original Gray", frame_gray);
  equalizeHist(frame_gray, frame_gray);
  //imshow("Equalized Gray", frame_gray);

  //Apply cascade detection
  cascade.detectMultiScale( frame_gray, obj, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

  //Draw the detected features
  Mat frame_tmp =frame.clone(); //make a copy of the original
  
  for( int i = 0; i < obj.size(); i++ ){
    //Draw rectangles
    rectangle(frame_tmp, Point(obj[i].x, obj[i].y), Point(obj[i].x + obj[i].width, obj[i].y + obj[i].height), Scalar( 0, 255, 0 ), 2);
  }

  //Display results
  imshow((string)"dartH",frame_tmp);
};


//Apply all detections and display all
void detectAndDisplay( Mat frame , string s){
  //Show Haar object detection
//  haarDetect(frame, s);
  //Haar transform
//  haarCircles(frame, s);
  //Find contours
//  detectContours(frame,s);
  //Detect lines
  detectLines(frame, s);
  //SURF
//  surfDetect(frame, s);
};


int main( int argc, const char** argv )
{ 
  for(int i=0;i<12;i++){
    
    string s  = int_to_string(i); //current image index

    //Read source image
    Mat frame = imread((string)"dart"+s+(string)".jpg", CV_LOAD_IMAGE_COLOR);

    if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
    
    //Show source
    //imshow((string)"dart",frame);
    
    //Apply detection and display results
    detectAndDisplay(frame ,s);
    
    //Hold the windows open
    waitKey();
  }

  return 0;
};