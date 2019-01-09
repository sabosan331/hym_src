# 下記の2つは自分の環境に合わせて書き換える
SET(OpenCV_DIR "/home/hymb305/dev/opencv-2.4.9/build") # OpenCVConfig.cmake があるディレクトリ
INCLUDE_DIRECTORIES("/home/hymb305/dev/opencv-2.4.9/build/include") # 上記ディレクトリの下のincludeディレクトリ

CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT( samp1 )
FIND_PACKAGE( OpenCV REQUIRED )
ADD_EXECUTABLE( facedetect main.cpp )
TARGET_LINK_LIBRARIES( facedetect ${OpenCV_LIBS})