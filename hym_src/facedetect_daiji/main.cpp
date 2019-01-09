
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp> // highguiのヘッダーをインクルード
#include "cascadedetectKoji.cpp"

#include <cctype>
#include <iostream>
#include <iterator>
#include <stdio.h>

using namespace std;
using namespace cv;

//cascade is haar-like,HOG,LBP
/*  haar-like feature
    顔  haarcascade_frontalface_default.xml   haarcascade_frontalface_alt.xml  
    //haarcascade_frontalface_alt2.xml//  haarcascade_frontalface_alt_tree.xml
    横  haarcascade_profileface.xml 
    目  haarcascade_eye.xml  haarcascade_eye_tree_eyeglasses.xml
    haarcascade_mcs_lefteye.xml  haarcascade_mcs_righteye.xml
    haarcascade_lefteye_2splits.xml    haarcascade_righteye_2splits.xml
    haarcascade_mcs_eyepair_big.xml    haarcascade_mcs_eyepair_small.xml
    耳  haarcascade_mcs_leftear.xml haarcascade_mcs_rightear.xml
    口  haarcascade_mcs_mouth.xml
    鼻  haarcascade_mcs_nose.xml
    体  haarcascade_fullbody.xml    haarcascade_mcs_upperbody.xml
    haarcascade_lowerbody.xml    haarcascade_upperbody.xml
*/

const char* cascadeName = "/home/hymb305/dev/opencv-2.4.9/data/haarcascades/haarcascade_frontalface_alt2.xml";
const char* nestedCascadeName = "/home/hymb305/dev/opencv-2.4.9/data/haarcascades/haarcascade_mcs_mouth.xml";
                                                                                 
const static Scalar colors[] = 
  { CV_RGB(0,255,0),CV_RGB(0,128,255),CV_RGB(0,255,255),CV_RGB(0,255,0),CV_RGB(255,128,0),
    CV_RGB(255,255,0),CV_RGB(255,0,0),CV_RGB(255,0,255) };


Mat skinDetect(Mat& In){

  // const int sizes[] = { In.size[0], In.size[1], In.size[2] };
  Mat res,hsv_img;
  res.create(In.rows,In.cols, CV_8UC3);

  cv::cvtColor( In ,hsv_img,CV_BGR2HSV);	//HSVに変換		
  for(int y=0; y<hsv_img.rows;y++){
    for(int x=0; x<hsv_img.cols; x++){
      int a = hsv_img.step*y+(x*3);
      if(hsv_img.data[a] >=3 && hsv_img.data[a] <=38 /*&&hsv_img.data[a+1] >=50 && hsv_img.data[a+2] >= 50*/ ) //HSVでの検出
	{
	  res.data[a] = 0; //肌色部分を青に	
	  res.data[a+1] = 255;
	  res.data[a+2] = 0;
	} else {
	res.data[a] = 0; //肌色部分を青に	
	res.data[a+1] = 0;
	res.data[a+2] = 0;
      }
    }
  }	

  // imshow("hoge",res);


   return res;
}


void detectAndDraw( Mat& img, CascadeClassifier& cascade,
                    CascadeClassifier& nestedCascade,
                    double scale )
{
  int i = 0;
  double t = 0;
  vector<Rect> faces, faces2;
   
  Mat gray, smallImg( cvRound (img.rows/scale), cvRound(img.cols/scale), CV_8UC1 );

  cvtColor( img, gray, CV_BGR2GRAY ); //グレイスケール化
  resize( gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR );
  equalizeHist( smallImg, smallImg ); //ヒストグラム平滑化

  t = (double)cvGetTickCount(); //処理スタート

  cascade.detectMultiScaleKoji( smallImg, faces,
			    1.1, 2, 0
			    //|CV_HAAR_FIND_BIGGEST_OBJECT
			    //|CV_HAAR_DO_ROUGH_SEARCH
			    |CV_HAAR_SCALE_IMAGE
			    ,
			    Size(20, 20) );

   
  for( vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++ )
    {
      Mat smallImgROI;
      vector<Rect> nestedObjects;
      Point center;
      Scalar color = colors[i%8];
      int radius;

      double aspect_ratio = (double)r->width/r->height;

      rectangle( img, cvPoint(cvRound(r->x*scale), cvRound(r->y*scale)),
		 cvPoint(cvRound((r->x + r->width-1)*scale), cvRound((r->y + r->height-1)*scale)),
		 color, 3, 8, 0);
      if( nestedCascade.empty() ){
	cout << "nastedCascade is empty" << endl;
	continue;
      }else// cout << "aruyo" << endl;
      smallImgROI = smallImg(*r);
      imshow("ROI",smallImgROI);
      nestedCascade.detectMultiScale( smallImgROI, nestedObjects,
				      1.1, 2, 0
				      //|CV_HAAR_FIND_BIGGEST_OBJECT
				      //|CV_HAAR_DO_ROUGH_SEARCH
				      //|CV_HAAR_DO_CANNY_PRUNING
				      |CV_HAAR_SCALE_IMAGE
				      ,
				      Size(10, 10) );
      for( vector<Rect>::const_iterator nr = nestedObjects.begin(); nr != nestedObjects.end(); nr++ ){
	//printf("ROI(x,y) = (%d,%d)\n",smallImgROI.cols,smallImgROI.rows);
	//printf("detect(x,y) = (%d,%d)\n",nr->x,nr->y);
	if( nr->y > smallImgROI.rows/2){
	  center.x = cvRound((r->x + nr->x + nr->width*0.5)*scale);
	  center.y = cvRound((r->y + nr->y + nr->height*0.5)*scale);
	  radius = cvRound((nr->width + nr->height)*0.25*scale);
	  circle( img, center, radius, CV_RGB(255,0,0) , 3, 8, 0 );
	}
      }
    }
  cv::imshow( "result", img );
}


int main(int argc, const char* argv[])
{

  Mat frame, frameCopy, image;
 

  CascadeClassifier cascade, nestedCascade;
  cascade.load( cascadeName ); //cascade
  nestedCascade.load( nestedCascadeName ); //nasted

  double scale = atoi(argv[1]); //smalling
  cv::VideoCapture cap(argv[2]); //load movie

  Mat tmpImg;
  cap >> tmpImg;
  printf("(w,h) = (%d,%d)\n", tmpImg.cols, tmpImg.rows); 
  Mat hsv_img;
  while(1)
    {
      cap >> frame; // 1frame
      frame.copyTo(tmpImg);
      //    if (frame.empty()) break;
      // cv::resize(frame,sframe, cv::Size(), 0.5, 0.5); 
      detectAndDraw( frame, cascade, nestedCascade, scale );
        tmpImg = skinDetect(tmpImg);
	/*   cv::cvtColor( tmpImg ,hsv_img,CV_BGR2HSV);	//HSVに変換		
      for(int y=0; y<hsv_img.rows;y++){
	for(int x=0; x<hsv_img.cols; x++){
	  int a = hsv_img.step*y+(x*3);
	  if(hsv_img.data[a] >=0 && hsv_img.data[a] <=38 &&hsv_img.data[a+1] >=50 && hsv_img.data[a+2] >= 50 ) //HSVでの検出
	    {
	      frame.data[a] = 0; //肌色部分を青に	
	      frame.data[a+1] = 255;
	      frame.data[a+2] = 0;
	    } else {
	    frame.data[a] = 0; //肌色部分を青に	
	    frame.data[a+1] = 0;
	    frame.data[a+2] = 0;
	  }
	}
      }	

      imshow("hoge",frame);
      //	  waitKey(0);*/


      if(cv::waitKey(30) >= 0) break;
    }
  return 0;
}
