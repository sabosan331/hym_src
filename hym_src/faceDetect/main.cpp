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
const static Scalar colors[] = 
  { CV_RGB(0,255,0),CV_RGB(0,128,255),CV_RGB(0,255,255),CV_RGB(0,255,0),CV_RGB(255,128,0),
    CV_RGB(255,255,0),CV_RGB(255,0,0),CV_RGB(255,0,255) };

Mat image;
bool backprojMode = false;
int trackObject = 0;
bool showHist = true;
int vmin = 10, vmax = 256, smin = 30;

const char* keys =
  {
    //"{1|  | 0 | camera number}"
    "{sc | double | 2   | scale num| }"
    "{ad | int    | 1   | add span |}"
    "{th | float    | 100.0 | threshold |}"
  };

const char* facerec[] = {"1", "2"};

int main( int argc, const char** argv )
{
  
  CommandLineParser parser(argc, argv, keys); 

  //input video
  VideoCapture cap(argv[1]); //for movie
  double scale = parser.get<double>("sc"); //Scaling

  //output video:Linux = "XVID"，Windows = "MJPG"
  Mat bframe; cap >> bframe; //1frame
  VideoWriter writer("result.avi",CV_FOURCC_MACRO('M', 'J', 'P', 'G'), 25, Size(bframe.cols,bframe.rows), true);

  //histgram
  int hsize = 8; //def:16 //n:128
  float hranges[] = {0,180};
  const float* phranges = hranges;

  //facedetector
  CascadeClassifier cascade, nestedCascade;
  cascade.load( cascadeName ); //cascade
  nestedCascade.load( nestedCascadeName ); //nasted

  //for tracking
  Mat frame,hsv, hue,tmpImg,mask, histimg = Mat::zeros(200, 320, CV_8UC3);
  bool paused = false;
  Rect traWin[10]; int objNum = 10; //トラックウィンドウ10用意,動的確保はいつかやる
  Mat maskroi[10]; Mat hist[10]; Mat roi[10];
  Mat tmphist[10];
  Mat backproj[10];
  int addcnt=0; int addSpan = parser.get<int>("ad"); //何フレームごとに追加受け付けるか

  for(;;)
    {

      // int64 start = cv::getTickCount(); //スタート

      cap >> frame;
      frame.copyTo(image); //これに処理を加える、逆にOrgはframe
      cvtColor(image, hsv, COLOR_BGR2HSV); //HSV変換

      //---------------------ショット境界検出---------------------------------//
      if( detectShot(frame,bframe,parser.get<float>("th")) == true){ //ショット境界で追跡リセット
	for(int i=0;i<objNum;i++)traWin[i] = Rect(0,0,0,0);
        //printf("shotPoint %d\n",frameNum);
      }

      //-------------------------顔検出--------------------------------------//
      if(traWin[0].area()<=1){
	detectAndDraw( image, cascade,nestedCascade,scale,traWin); //顔検出
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
	    meanShift(backproj[i], traWin[i], TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 ));
	  }

	}
	//------------------------追跡失敗------------------------------------//
	if( trackMiss(frame,traWin) == true){ //ショット境界で追跡リセット
	  for(int i=0;i<objNum;i++) traWin[i] = Rect(0,0,0,0);
	  //  printf("trackMiss %d\n",frameNum);
	}

	//------------------------出力結果-------------------------------------//
	for(int i=0;i<objNum;i++){
	   if(traWin[i].area()>1){ //追跡窓の値のあるものだけ
	     Rect resRect = Rect(traWin[i].x,traWin[i].y+traWin[i].height*0.2,traWin[i].width,traWin[i].height);
	     rectangle( image, resRect, colors[i], 3, CV_AA ); //for meanShift
	   }
	}

      }

      ////////////////////////////////////////////////////////////////////////////////////////////
      ////////////////////////////////////////////////////////////////////////////////////////////
      imshow( "faceDetect Result", image );
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
	case 'e':
	  goto end;

	}

        writer << image;

	// Timer /////////////////////////////////////////////////////////////////
	//int64 end = cv::getTickCount();
	//double elapsedMsec = (end - start) * 1000 / cv::getTickFrequency();
        //std::cout << elapsedMsec << "ms" << std::endl;
	///////////////////////////////////////////////////////////////////////////
    }

 end:
  cout << "End  " << endl;
  return 0;
}
