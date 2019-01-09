#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"


#include <iostream>
#include <ctype.h>
#include <stdio.h>
#include "sktool.h"

using namespace cv;
using namespace std;

//顔  haarcascade_frontalface_default.xml   haarcascade_frontalface_alt.xml  
//haarcascade_frontalface_alt2.xml//  haarcascade_frontalface_alt_tree.xml

const char* cascadeName = "/home/hym70/haarcascades/haarcascade_frontalface_alt_tree.xml";
const char* nestedCascadeName = "/home/hym70/haarcascades/haarcascade_mcs_mouth.xml";
//const static Scalar colors[] = 
// { CV_RGB(0,255,0),CV_RGB(0,128,255),CV_RGB(0,255,255),CV_RGB(0,255,0),CV_RGB(255,128,0),
//  CV_RGB(255,255,0),CV_RGB(255,0,0),CV_RGB(255,0,255) };
const static Scalar colors[] = 
 { CV_RGB(0,255,0),CV_RGB(0,255,200),CV_RGB(100,255,0),CV_RGB(50,255,50),CV_RGB(255,0,255),
   CV_RGB(255,255,0),CV_RGB(255,0,0),CV_RGB(255,0,255),CV_RGB(0,255,0),CV_RGB(0,255,200),CV_RGB(100,255,0),CV_RGB(50,255,50),CV_RGB(255,0,255),
   CV_RGB(255,255,0),CV_RGB(255,0,0),CV_RGB(255,0,255) };

Mat image;
bool backprojMode = false;
bool selectObject = false;
int trackObject = 0;
bool showHist = true;
Point origin;
Rect selection;
int vmin = 10, vmax = 256, smin = 30;
int TRACKINGNUM=0;

const char* keys =
  {
    //"{1|  | 0 | camera number}"
    "{sc | double | 2   | scale num| }"
    "{ad | int    | 1   | add span |}"
    "{th | float    | 100.0 | threshold |}"
  };

#define MAX_OBJ 20

void multicrush(Rect* faceArea){

  int objNum=0;

  for(int i=0;i<MAX_OBJ;i++){
    if(faceArea[i].area()>1) objNum++;
  }

  for(int i=0;i<objNum;i++){
    int rx1 = faceArea[i].x; int ry1 = faceArea[i].y;
    int rx2 = faceArea[i].x + faceArea[i].width; int ry2 = faceArea[i].y + faceArea[i].height;
    for(int j=i+1;j<objNum;j++){
       int fx1 = faceArea[j].x; int fx2 = fx1 + faceArea[j].width;
       int fy1 = faceArea[j].y; int fy2 = fy1 + faceArea[j].height;
       if( (rx1<=fx2 && fx1 <= rx2 && fx1 <= rx2 && ry1 <= fy2 && fy1 <= ry2 ) ){
	 for(int k=0;k<MAX_OBJ;k++) faceArea[k] = Rect(0,0,0,0);
       }
    }
  }
}

const char* facerec[] = {"1", "2"};

int main( int argc, const char** argv )
{
  
  CommandLineParser parser(argc, argv, keys); 

  //input video
  VideoCapture cap(argv[1]); //for movie
  double scale = parser.get<double>("sc"); //Scaling

  //input camera
  // VideoCapture cap; //for Camera
  // int camNum = parser.get<int>("1");
  // cap.open(camNum);

  //output video:Linux = "XVID"，Windows = "MJPG"
  Mat bframe; cap >> bframe; //1frame
   VideoWriter writer("result.avi",CV_FOURCC_MACRO('M', 'J', 'P', 'G'), 25, Size(bframe.cols,bframe.rows), true);

  //histgram
  int hsize = 16; //def:16 //n:128
  float hranges[] = {0,180};
  const float* phranges = hranges;

  //facedetector
  CascadeClassifier cascade, nestedCascade;
  cascade.load( cascadeName ); //cascade
  nestedCascade.load( nestedCascadeName ); //nasted

  //for tracking
  Mat frame,hsv, hue,tmpImg,mask, histimg = Mat::zeros(200, 320, CV_8UC3);
  bool paused = false;
  Rect traWin[20]; int objNum = 20; //トラックウィンドウ
  //  Mat mask[10];
  Mat maskroi[20]; Mat hist[20]; Mat roi[20];
  Mat tmphist[20];
  Mat backproj[20];

  int addcnt=0; int addSpan = parser.get<int>("ad"); //何フレームごとに追加受け付けるか
  int tframe=0; //フレーム数

  namedWindow( "Histogram", 0 );  namedWindow( "CamShift Demo", 0 );
  createTrackbar( "Vmin", "CamShift Demo", &vmin, 256, 0 ); createTrackbar( "Vmax", "CamShift Demo", &vmax, 256, 0 );
  createTrackbar( "Smin", "CamShift Demo", &smin, 256, 0 );

  
  int frameNum = 1;
  for(;;)
    {

      // int64 start = cv::getTickCount(); //スタート

      cap >> frame;
      frame.copyTo(image); //これに処理を加える、逆にOrgはframe
      cvtColor(image, hsv, COLOR_BGR2HSV); //HSV変換

      //---------------------ショット境界検出---------------------------------//
      if( detectShot(frame,bframe,parser.get<float>("th")) == true){ //ショット境界で追跡リセット
	for(int i=0;i<objNum;i++)traWin[i] = Rect(0,0,0,0);
        printf("shotPoint %d\n",frameNum);
      }

      //-------------------------顔検出--------------------------------------//
      if(traWin[0].area()<=1){
	detectAndDraw( image, cascade,nestedCascade,scale,traWin); //顔検出
	tframe = 1;
      }



      //-------------------------顔追跡--------------------------------------//
      //-------------------登場人物追加を実装段階-----------------------------//
      if(traWin[0].area() > 1 )	{ //検出されたら追跡

	int _vmin = vmin, _vmax = vmax;
	inRange(hsv, Scalar(0, smin, MIN(_vmin,_vmax)),Scalar(180, 256, MAX(_vmin, _vmax)), mask);
	int ch[] = {0, 0};
	hue.create(hsv.size(), hsv.depth());
	mixChannels(&hsv, 1, &hue, 1, ch, 1);

	//----------------------人物追加-------------------------------------//
	if(addcnt > addSpan){
	  addFace( image, cascade,nestedCascade,scale,traWin);
	  addcnt = 0;
	}
	addcnt++;



	for(int i=0;i<objNum;i++){

	  if(traWin[i].area()>1){ //追跡窓の値のあるものだけ
	    roi[i] =  Mat(hue, traWin[i]);
	    maskroi[i] =  Mat(mask, traWin[i]);
	    calcHist(&roi[i], 1, 0, maskroi[i], hist[i], 1, &hsize, &phranges);
	    normalize(hist[i], hist[i], 0, 255, CV_MINMAX);
	    calcBackProject(&hue, 1, 0, hist[i], backproj[i], &phranges);
	    //おそらくエラーが出る原因//backproj &= mask;
	    //imshow("backproj",backproj);
	    //meanShift
	    //if(traWin[0].size()>1)
	    meanShift(backproj[i], traWin[i], TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 ));
	  }

	}
	//------------------------追跡失敗------------------------------------//
	if( trackMiss(frame,traWin) == true){ //ショット境界で追跡リセット
	  for(int i=0;i<objNum;i++) traWin[i] = Rect(0,0,0,0);
	  printf("trackMiss %d\n",frameNum);
	}

	//------------------------追跡窓がぶつかったら--------------------------//
	multicrush(traWin);

	//camShift
	//RotatedRect trackBox = CamShift(backproj, trackWindow, TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 ));

	//尤度画像みたいなもの
	//	if( backprojMode ) cvtColor( backproj[0], image, COLOR_GRAY2BGR );

	//ellipse( image, trackBox, Scalar(0,0,255), 3, CV_AA ); //for camShift
	//traWin[0] =  Rect(100,100,100,100);
	

	for(int i=0;i<objNum;i++){
	   if(traWin[i].area()>1){ //追跡窓の値のあるものだけ
	     Rect resRect = Rect(traWin[i].x,traWin[i].y+traWin[i].height*0.2,traWin[i].width,traWin[i].height);
	     Mat roiImg(image,resRect);
	     //colors[i] = CV_RGB(0,255,0);
	     if(skinAreaFilter(roiImg) == true) rectangle( image, resRect, CV_RGB(0,255,0)/*colors[i]*/, 3, CV_AA ); //for meanShift
	   }
	}

      }


      ////////////////////////////////////////////////////////////////////////////////////////////
      ////////////////////////////////////////////////////////////////////////////////////////////
  
      imshow( "faceDetect Result", image );
      //imshow( "Histogram", histimg );
 
      frame.copyTo(bframe); //bフレーム受け渡し

      //キー入力
      char c = (char)waitKey(20);
      if( c == 27 )
	break;
      switch(c)
	{
	case 'b':
	  backprojMode = !backprojMode;
	  break;
	case 'c':
	  trackObject = 0;
	  histimg = Scalar::all(0);
	  for(int i=0;i<objNum;i++)traWin[i] = Rect(0,0,0,0);
	  break;
	case 'h':
	  showHist = !showHist;
	  if( !showHist )
	    destroyWindow( "Histogram" );
	  else
	    namedWindow( "Histogram", 1 );
	  break;
	case 'p':
	  paused = !paused;
	  break;
	default:
	  ;
	}

        writer << image;
	//	frameNum++;

	// Timer /////////////////////////////////////////////////////////////////
	//int64 end = cv::getTickCount();
	//double elapsedMsec = (end - start) * 1000 / cv::getTickFrequency();
        //std::cout << elapsedMsec << "ms" << std::endl;
	///////////////////////////////////////////////////////////////////////////
    }

  return 0;
}
