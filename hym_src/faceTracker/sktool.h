#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"


#include <iostream>
#include <ctype.h>
#include <stdio.h>

using namespace cv;
using namespace std;


bool skinAreaFilter(Mat &img){

  float sum=0;
  float ratio=0;
  float area = img.cols * img.rows;
  Mat hsv_img;

  cv::cvtColor( img , hsv_img,CV_BGR2HSV);	//HSVに変換		
  for(int y=0; y<hsv_img.rows;y++){
    for(int x=0; x<hsv_img.cols; x++){
      int a = hsv_img.step*y+(x*3);
      if(hsv_img.data[a] >=0 && hsv_img.data[a] <=50 /*&&hsv_img.data[a+1] >=50*/ && hsv_img.data[a+2] >= 20){
	sum++;
      }
    }
  }

  ratio = sum / area;

  if(ratio>0.4) return true;
  else return false;
}

/*
void face2ndDetect(Rect &faceRec1,Rect &faceRec2){

  int cx,cy;
  cx = faceRec2.cols + 0.5 * faceRec2.width;
  cy = faceRec2.rows + 0.5 * faceRec2.height;




  }*/

//add face while tracking
void addFace( Mat& img, CascadeClassifier& cascade,
		    CascadeClassifier& nestedCascade,
	      double scale ,Rect* faceArea){

  double t = 0;
  vector<Rect> faces, faces2;
  Rect facetmp[20];
  
  Mat orgSmall( cvRound (img.rows/scale), cvRound(img.cols/scale), CV_8UC3 );
  Mat gray, smallImg( cvRound (img.rows/scale), cvRound(img.cols/scale), CV_8UC1 );

  cvtColor( img, gray, CV_BGR2GRAY ); //グレイスケール化
  resize( gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR );
  resize(img , orgSmall,orgSmall.size(),0,0,INTER_LINEAR );
  equalizeHist( smallImg, smallImg ); //ヒストグラム平滑化

  cascade.detectMultiScale( smallImg, faces,
			    1.1, 2, 0
			    //|CV_HAAR_FIND_BIGGEST_OBJECT
			    |CV_HAAR_DO_ROUGH_SEARCH,
			    // |CV_HAAR_SCALE_IMAGE , //defalt!!
			    Size(0, 0) );

  int facecnt=0; //追跡中の探索窓数
 for(int i=0;faceArea[i].area()>1;i++) facecnt++;

   int addNum=0;
   int difcnt=0; //追跡中の探索窓と位置が異なるとカウント

   for( vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++ ){
     difcnt =0;
     int rx1 = scale*r->x; int ry1 = scale*r->y;
     int rx2 = scale*r->x + scale*r->width; int ry2 = scale*r->y + scale*r->height;
     for(int i=0;faceArea[i].area()>1;i++){
       int fx1 = faceArea[i].x; int fx2 = fx1 + faceArea[i].width;
       int fy1 = faceArea[i].y; int fy2 = fy1 + faceArea[i].height;
       if( !(rx1<=fx2 && fx1 <= rx2 && fx1 <= rx2 && ry1 <= fy2 && fy1 <= ry2 ) ){
	 difcnt++; //追跡中の探索窓と被らないように
       }
     }
     if(facecnt == difcnt){ //全ての探索窓と位置が違ったら
       facetmp[addNum] = Rect( scale*r->x,scale*r->y,scale*r->width,scale*r->height);
       addNum++;
     }
   }
  //cout << facetmp[0].area() << endl;
  // facetmp[0] = Rect( 100,200,100,100);

    //faceArea[2] = Rect( 100,200,100,100);

  
 
  
  for(int i=0;i<facecnt;i++){
    if(facetmp[i].area()>1) faceArea[facecnt+i] = facetmp[i];
    }

  //for(int i=0;1<faceArea[i].area();i++)// cout << faceArea[i] << endl;

    //char k = waitKey(0);
    //  cout << detectNum << "faces detect!!" << endl;
}


void detectAndDraw( Mat& img, CascadeClassifier& cascade,
		    CascadeClassifier& nestedCascade,
                    double scale ,Rect* faceArea){
 
  double t = 0;
  vector<Rect> faces, faces2;

  Mat orgSmall( cvRound (img.rows/scale), cvRound(img.cols/scale), CV_8UC3 );
  Mat gray, smallImg( cvRound (img.rows/scale), cvRound(img.cols/scale), CV_8UC1 );

  cvtColor( img, gray, CV_BGR2GRAY ); //グレイスケール化
  resize( gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR );
  resize(img , orgSmall,orgSmall.size(),0,0,INTER_LINEAR );
  equalizeHist( smallImg, smallImg ); //ヒストグラム平滑化

  cascade.detectMultiScale( smallImg, faces,
			    1.1, 2, 0
			    //|CV_HAAR_FIND_BIGGEST_OBJECT
			    |CV_HAAR_DO_ROUGH_SEARCH,
			    // |CV_HAAR_SCALE_IMAGE , //defalt!!
			    Size(0, 0) );
  int i = 0;
  int detectNum=0;
  for(  vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++ , i++ ){
    //if(r==faces.begin())
      faceArea[i] = Rect( scale*faces[0].x,scale*faces[0].y,scale*faces[0].width,scale*faces[0].height);
      
    detectNum++;
  }

  // cout << detectNum << "faces detect!!" << endl;

  // cout << detectNum << endl;

  // cout  << "faceArea" <<  faceArea << endl;

  
  //faceArea = Rect(100,100,50,50);
  //cout << faceArea << endl;
  // faceArea = faces[0];
  // bool skinFlag = false;
  /*
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
    // skinFlag = skinAreaFilter(ROI); //肌色面積フィルター

    // if(skinFlag == true) color = CV_RGB(0,255,0); //黄緑
    // else color = CV_RGB(0,0,255);

    // rectangle( img, cvPoint(cvRound(r->x*scale), cvRound(r->y*scale)),
    //	 cvPoint(cvRound((r->x + r->width-1)*scale), cvRound((r->y + r->height-1)*scale)),
    //	 color, 3, 8, 0);
     
    if( nestedCascade.empty() ){
    cout << "nastedCascade is empty" << endl;
    continue;
    }// cout << "aruyo" << endl;
    Mat smallImgROI = smallImg(*r);
       

    // imshow("ROI",ROI); 
     
    nestedCascade.detectMultiScale( smallImgROI, nestedObjects,
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
    }*/


  // cv::imshow( "result", img );
}


bool detectShot(Mat& bframe,Mat& frame,float threshold){

   float sad=0;
   float rsum,gsum,bsum; rsum = gsum = bsum = 0;
   //float threshold = 70;

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

   //  printf("%.1f\n" ,sad);
   if( threshold < sad ) return true;
   else false;
}

//追跡窓が画面の縁に接触したら追跡失敗
bool trackMiss(Mat &frame,Rect* faceArea){

  if(faceArea[0].x<=0 || faceArea[0].y<=0 || faceArea[0].x+faceArea[0].width >= frame.cols || faceArea[0].y+faceArea[0].height >= frame.rows)
    return true;
  else
    return false;
}
