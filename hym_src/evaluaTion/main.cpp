#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include <iostream>
#include <ctype.h>
#include <stdio.h>

using namespace cv;
using namespace std;

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
    "{s|double|scale num| }"
  };


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


int main( int argc, const char** argv ){

  CommandLineParser parser(argc, argv, keys); 

  //input video
  VideoCapture cap(argv[1]); //for movie
  double scale = parser.get<double>("s"); //Scaling

  Mat image,frame,bframe;
  int frmcnt=1;
  int shotcnt =0;
  int shot[200];
  int grossfrm = 0;

  for(int i=0;i<200;i++){
    shot[i] = 0;
  }

  cap >> bframe;


  while(1){


      cap >> frame;
      if(frame.empty() == true) goto end;

      frame.copyTo(image); //これに処理を加える、逆にOrgはframe

      imshow( "result", image );

      if(detectShot(bframe,frame) == true){
	//	shot[shotcnt] = frmcnt;
	frmcnt = 1;
	shotcnt++;
	cout << "---------------detectShot!!------------------" << endl;
      }

      //キー入力
      char c = (char)waitKey(0);
      /*  if( c == 27 )
	  break;*/
      switch(c){
      case 'r':
	frmcnt = 0;
	break;
      case 'e':
	frmcnt = 0;
	break;
	//	goto end;
      }

      frmcnt++;
      grossfrm++;
      cout << "frameNum: " << frmcnt << "," << "gross :" << grossfrm << endl; 

      frame.copyTo(bframe);
  }
 end:


  /*
  for(int i = 0; i< 200; i++){
    cout << i << ":" << shot[i] << endl;
  }

  cout << "ショット数: " << frmcnt << endl;*/

  return 0;
}
