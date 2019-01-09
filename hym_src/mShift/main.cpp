#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include <iostream>
#include <ctype.h>
#include <stdio.h>

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
bool selectObject = false;
int trackObject = 0;
bool showHist = true;
Point origin;
Rect selection;
int vmin = 10, vmax = 256, smin = 30;

int TRACKINGNUM=0;


/*
bool skinAreaFilter(Mat &img){

  float sum=0;
  float ratio=0;
  float area = img.cols * img.rows;
  Mat hsv_img;

  cv::cvtColor( img , hsv_img,CV_BGR2HSV);	//HSVに変換		
  for(int y=0; y<hsv_img.rows;y++){
    for(int x=0; x<hsv_img.cols; x++){
      int a = hsv_img.step*y+(x*3);
      if(hsv_img.data[a] >=0 && hsv_img.data[a] <=50 &&hsv_img.data[a+1] >=50 && hsv_img.data[a+2] >= 50){
	sum++;
      }
    }
  }
 
  ratio = sum / area;

  if(ratio>0.5) return true;
  else return false;
  }*/

/*
void face2ndDetect(Rect &faceRec1,Rect &faceRec2){

  int cx,cy;
  cx = faceRec2.cols + 0.5 * faceRec2.width;
  cy = faceRec2.rows + 0.5 * faceRec2.height;




  }*/

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

  cout << detectNum << "faces detect!!" << endl;

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


bool detectShot(Mat& bframe,Mat& frame){

   float sad=0;
   float rsum,gsum,bsum; rsum = gsum = bsum = 0;
   float threshold = 70;

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

const char* keys =
  {
    //"{1|  | 0 | camera number}"
    "{s|double|scale num| }"
  };

const char* facerec[] = {"1", "2"};

int main( int argc, const char** argv )
{

  CommandLineParser parser(argc, argv, keys);
  //VideoCapture cap;  int camNum = parser.get<int>("1");   cap.open(camNum); //for Web Camera
  VideoCapture cap(argv[1]); //for movie
  double scale = parser.get<double>("s"); //
  //Rect trackWindow;
 
  int hsize = 128; //def:16
  float hranges[] = {0,180};
  const float* phranges = hranges;
  Mat frame,bframe, hsv, hue,tmpImg,mask, histimg = Mat::zeros(200, 320, CV_8UC3), backproj;
  bool paused = false;
  CascadeClassifier cascade, nestedCascade;
  cascade.load( cascadeName ); //cascade
  nestedCascade.load( nestedCascadeName ); //nasted
  VideoWriter writer("result.avi",CV_FOURCC_MACRO('M', 'J', 'P', 'G'), 20, Size(bframe.cols,bframe.rows), true);
  //XVID　Linux
  //MJPG  Windows

  //for tracking
  Rect traWin[10]; int objNum = 1;
  //  Mat mask[10];
 Mat maskroi[10]; Mat hist[10]; Mat roi[10];
 Mat tmphist[10];

 int tframe=0;


  namedWindow( "Histogram", 0 );  namedWindow( "CamShift Demo", 0 );
  createTrackbar( "Vmin", "CamShift Demo", &vmin, 256, 0 ); createTrackbar( "Vmax", "CamShift Demo", &vmax, 256, 0 );
  createTrackbar( "Smin", "CamShift Demo", &smin, 256, 0 );

  cap >> bframe; //1frame
  int frameNum = 1;
  for(;;)
    {
      cap >> frame;
      frame.copyTo(image); //これに処理を加える、逆にOrgはframe
      cvtColor(image, hsv, COLOR_BGR2HSV); //HSV変換

      //---------------------ショット境界検出---------------------------------//
      if( detectShot(frame,bframe) == true){ //ショット境界で追跡リセット
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

	//detect2n( image, cascade,nestedCascade,scale,traWin[1]); //2つ目以降
	//	check2nd(traWin[0],traWin[1]);
	//左ボタンが上がったらtrackobject = -1
	//ここを顔検出が起きたらに変える
	if(traWin[0].area() > 1 ){// trackObject < 0

	  roi[0] =  Mat(hue, traWin[0]);
	  maskroi[0] =  Mat(mask, traWin[0]);
	  calcHist(&roi[0], 1, 0, maskroi[0], hist[0], 1, &hsize, &phranges);
	  normalize(hist[0], hist[0], 0, 255, CV_MINMAX);

	  if(tframe == 1) tmphist[0] = hist[0];
	  else hist[0] = tmphist[0];
	  tframe++;


	  //trackObject = 1;

	  //	  histimg = Scalar::all(0);
	  //int binW = histimg.cols / hsize;
	  //	  Mat buf(1, hsize, CV_8UC3);
	  //	  for( int i = 0; i < hsize; i++ )
	  //	    buf.at<Vec3b>(i) = Vec3b(saturate_cast<uchar>(i*180./hsize), 255, 255);

	  //	  cvtColor(buf, buf, CV_HSV2BGR);

	}

	


	calcBackProject(&hue, 1, 0, hist[0], backproj, &phranges);
	backproj &= mask;
	//imshow("backproj",backproj);
	//meanShift
	//if(traWin[0].size()>1)
	meanShift(backproj, traWin[0], TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 ));

	//camShift
	//RotatedRect trackBox = CamShift(backproj, trackWindow, TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 ));

	//尤度画像みたいなもの
	if( backprojMode ) cvtColor( backproj, image, COLOR_GRAY2BGR );

	//ellipse( image, trackBox, Scalar(0,0,255), 3, CV_AA ); //for camShift
	//traWin[0] =  Rect(100,100,100,100);

	for(int i=0;i<objNum;i++){
	  Rect resRect = Rect(traWin[i].x,traWin[i].y+traWin[i].height*0.2,traWin[i].width,traWin[i].height);
	  rectangle( image, resRect, colors[i], 3, CV_AA ); //for meanShift
	  putText(image, facerec[i], cv::Point(resRect.x+resRect.width*0.45,resRect.y+resRect.height*0.5),cv::FONT_HERSHEY_SIMPLEX , 10*resRect.width/image.cols, cv::Scalar(0,255,0), 2, CV_AA);
	}
      }

      ////////////////////////////////////////////////////////////////////////////////////////////
      ////////////////////////////////////////////////////////////////////////////////////////////
  
      imshow( "faceDetect Result", image );
      //imshow( "Histogram", histimg );
 
      frame.copyTo(bframe); //bフレーム受け渡し

      //キー入力
      char c = (char)waitKey(10);
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
	  traWin[0] = Rect(0,0,0,0);
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
      frameNum++;
    }

  return 0;
}
