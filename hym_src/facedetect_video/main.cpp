#include "objdetectKoji.hpp"
//#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp> // highguiのヘッダーをインクルード
#include "cascadedetectKoji.cpp"

#include <cctype>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <math.h>

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

//const char* cascadeName = "/home/hym70/haarcascades/haarcascade_frontalface_alt_tree.xml";
const char* cascadeName = "/home/hym70/haarcascades/haarcascade_frontalface_alt_tree.xml";
const char* nestedCascadeName = "/home/hym70/haarcascades/tmp.xml";
const static Scalar colors[] = 
  { CV_RGB(0,255,0),CV_RGB(0,128,255),CV_RGB(0,255,255),CV_RGB(0,255,0),CV_RGB(255,128,0),
    CV_RGB(255,255,0),CV_RGB(255,0,0),CV_RGB(255,0,255) };

int sfilcnt = 0;
/*
Mat skinDetect(Mat& In){

  Mat res,hsv_img;
  res.create(In.rows,In.cols, CV_8UC3);
  cv::cvtColor( In ,hsv_img,CV_BGR2HSV);	//HSVに変換

  for(int y=0; y<hsv_img.rows;y++){
    for(int x=0; x<hsv_img.cols; x++){
      int a = hsv_img.step*y+(x*3);
      if(hsv_img.data[a] >=0 && hsv_img.data[a] <=50 &&hsv_img.data[a+1] >=50 && hsv_img.data[a+2] >= 50 ){
	  res.data[a] = 255; //肌色部分を白
	  res.data[a+1] = 255;
	  res.data[a+2] = 255;
      } else {
	res.data[a] = 0; //黒
	res.data[a+1] = 0;
	res.data[a+2] = 0;
      }
    }
  }	

   imshow("skin",res);
   return res;
}*/

bool skinAreaFilter(Mat &img){

  float sum=0;
  float ratio=0;
  float area = img.cols * img.rows;
  Mat hsv_img;

  cv::cvtColor( img , hsv_img,CV_BGR2HSV);	//HSVに変
  for(int x=0; x<hsv_img.cols; x++){
    for(int y=0; y<hsv_img.rows; y++){
      int a = hsv_img.step*y+(x*3);
      if(hsv_img.data[a] >=0 && hsv_img.data[a] <=50 /*&&hsv_img.data[a+1] >=50*/ && hsv_img.data[a+2] >= 20 ){
	sum++;
      }
    }
  }

  ratio = sum / area;

  if(ratio>0.3) return true;
  else return false;
}



void detectAndDraw( Mat& img, CascadeClassifier& cascade,
                    CascadeClassifier& nestedCascade,
                    double scale )
{
  int i = 0;
  double t = 0;
  vector<Rect> faces, faces2;
   
  Mat orgSmall( cvRound (img.rows/scale), cvRound(img.cols/scale), CV_8UC3 );
  Mat gray, smallImg( cvRound (img.rows/scale), cvRound(img.cols/scale), CV_8UC1 );

  cvtColor( img, gray, CV_BGR2GRAY ); //グレイスケール化
  resize( gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR );
  resize(img , orgSmall,orgSmall.size(),0,0,INTER_LINEAR );
  equalizeHist( smallImg, smallImg ); //ヒストグラム平滑化

  
 t = (double)cvGetTickCount(); //処理スタート

  cascade.detectMultiScaleKoji( smallImg, faces,
			    1.1, 2, 0
			    //|CV_HAAR_FIND_BIGGEST_OBJECT
			    //|CV_HAAR_DO_ROUGH_SEARCH
			    |CV_HAAR_SCALE_IMAGE
			    ,
			    Size(0, 0) );

  bool skinFlag = false;

  for( vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++ )
    {
      //Mat smallImgROI;
      vector<Rect> nestedObjects;
      Point center;
      //  Scalar color = colors[i%8];
      Scalar color;
      int radius;
      double aspect_ratio = (double)r->width/r->height;

       Mat ROI = orgSmall(*r);
       skinFlag = skinAreaFilter(ROI); //肌色面積フィルター

       //color = CV_RGB(0,255,0);
           if(skinFlag == true) color = CV_RGB(0,255,0); //黄緑
      else color = CV_RGB(255,0,0);
	   // color = CV_RGB(0,255,0);
       //if(skinFlag == true)
      rectangle( img, cvPoint(cvRound(r->x*scale), cvRound(r->y*scale)),
		 cvPoint(cvRound((r->x + r->width-1)*scale), cvRound((r->y + r->height-1)*scale)),
		 color, 3, 8, 0);
     
      if( nestedCascade.empty() ){
	cout << "nastedCascade is empty" << endl;
	continue;
      }// cout << "aruyo" << endl;
      Mat smallImgROI = smallImg(*r);
       

      // imshow("ROI",ROI); 
     
      nestedCascade.detectMultiScaleKoji( smallImgROI, nestedObjects,
				      1.1, 2, 0
				      //|CV_HAAR_FIND_BIGGEST_OBJECT
				      //|CV_HAAR_DO_ROUGH_SEARCH
				      //|CV_HAAR_DO_CANNY_PRUNING
				      |CV_HAAR_SCALE_IMAGE
				      ,
				      Size(0, 0) );
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


bool detectShot(Mat& bframe,Mat& frame){

  float sad=0;
  float rsum,gsum,bsum; rsum = gsum = bsum = 0;
  float threshold = 100;

  for(int y=0; y<frame.rows;y++){
    for(int x=0; x<frame.cols; x++){
      int a = frame.step*y+(x*3);	
      rsum +=  abs( bframe.data[a] - frame.data[a] ); 
      gsum +=  abs( bframe.data[a+1] - frame.data[a+1] );
      bsum +=  abs( bframe.data[a+2] - frame.data[a+2] ); 		  
    }
  }	

  sad = rsum + gsum + bsum;
  sad = sad / (frame.rows * frame.cols);
    
  if(sad > 100) return true;
  else false;
  //  printf("%.1f\n" ,sad);
  // if( threshold < sad ) cout << "detect shot!!!!!!" << endl;
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
  printf("framesize(w,h) = (%d,%d)\n", tmpImg.cols, tmpImg.rows); 


  VideoWriter writer("result.avi",CV_FOURCC_MACRO('X', 'V', 'I', 'D'), 20, Size(tmpImg.cols,tmpImg.rows), true);
  bool sflag = false;

  Mat hsv_img;

  int i=0;
  double t=0;

//      



  while(1)
    {
      //  cout << "frame = " << i << endl;
      
      cap >> frame; // 1frame

      //    detectShot(tmpImg,frame); //shot
      //   tmpImg = skinDetect(frame); //skin
	 //  frame.copyTo(tmpImg);
      //    if (frame.empty()) break;
      // cv::resize(frame,sframe, cv::Size(), 0.5, 0.5); 

       int64 start = cv::getTickCount();
      detectAndDraw( frame, cascade, nestedCascade, scale );

      int64 end = cv::getTickCount();
      double elapsedMsec = (end - start) * 1000 / cv::getTickFrequency();
      std::cout << elapsedMsec << "ms" << std::endl;
      
      int c = waitKey(1);
      if( c == 27) break; 

      
       writer << frame;
      // if(i>500) break;
      
      //  frame.copyTo(tmpImg); 
      i++;
    }
  return 0;
}
