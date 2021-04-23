###################
# LinkXNect.cmake #
###################

SET(CMAKE_FIND_LIBRARY_SUFFIXES ".dll.a" ".a" ".lib")

SET(Caffe_DEPENDENCIES_LIB_DIR "${Caffe_DEPENDENCIES_DIR}/lib")
SET(Caffe_DEPENDENCIES_VC14_LIB_DIR "${Caffe_DEPENDENCIES_DIR}/x64/vc14/lib")
SET(Caffe_LIB_DIR "${Caffe_ROOT_DIR}/build/lib")

FIND_LIBRARY(boost_python_LIBRARY boost_python-vc140-mt-1_61 HINTS ${Caffe_DEPENDENCIES_LIB_DIR})
FIND_LIBRARY(boost_system_LIBRARY boost_system-vc140-mt-1_61 HINTS ${Caffe_DEPENDENCIES_LIB_DIR})
FIND_LIBRARY(boost_thread_LIBRARY boost_thread-vc140-mt-1_61 HINTS ${Caffe_DEPENDENCIES_LIB_DIR})
FIND_LIBRARY(caffe_LIBRARY caffe HINTS ${Caffe_LIB_DIR})
FIND_LIBRARY(caffehdf5_LIBRARY caffehdf5 HINTS ${Caffe_DEPENDENCIES_LIB_DIR})
FIND_LIBRARY(caffehdf5_hl_LIBRARY caffehdf5_hl HINTS ${Caffe_DEPENDENCIES_LIB_DIR})
FIND_LIBRARY(caffezlib_LIBRARY caffezlib HINTS ${Caffe_DEPENDENCIES_LIB_DIR})
FIND_LIBRARY(caffeproto_LIBRARY caffeproto HINTS ${Caffe_LIB_DIR})
FIND_LIBRARY(gflags_LIBRARY gflags HINTS ${Caffe_DEPENDENCIES_LIB_DIR})
FIND_LIBRARY(glog_LIBRARY glog HINTS ${Caffe_DEPENDENCIES_LIB_DIR})
FIND_LIBRARY(libopenblas_LIBRARY NAMES libopenblas HINTS ${Caffe_DEPENDENCIES_LIB_DIR})
FIND_LIBRARY(libprotobuf_LIBRARY libprotobuf HINTS ${Caffe_DEPENDENCIES_LIB_DIR})
FIND_LIBRARY(mongoose_LIBRARY mongoose HINTS "${XNect_ROOT_DIR}/extern/mongoose/mongoose/lib/vc140")
FIND_LIBRARY(OpenCV_CALIB3D_LIBRARY opencv_calib3d310 HINTS ${Caffe_DEPENDENCIES_VC14_LIB_DIR})
FIND_LIBRARY(OpenCV_CORE_LIBRARY opencv_core310 HINTS ${Caffe_DEPENDENCIES_VC14_LIB_DIR})
FIND_LIBRARY(OpenCV_HIGHGUI_LIBRARY opencv_highgui310 HINTS ${Caffe_DEPENDENCIES_VC14_LIB_DIR})
FIND_LIBRARY(OpenCV_IMGCODECS_LIBRARY opencv_imgcodecs310 HINTS ${Caffe_DEPENDENCIES_VC14_LIB_DIR})
FIND_LIBRARY(OpenCV_IMGPROC_LIBRARY opencv_imgproc310 HINTS ${Caffe_DEPENDENCIES_VC14_LIB_DIR})
FIND_LIBRARY(OpenCV_VIDEOSTAB_LIBRARY opencv_videostab310 HINTS ${Caffe_DEPENDENCIES_VC14_LIB_DIR})
FIND_LIBRARY(XNect_LIBRARY XNECT HINTS "${XNect_ROOT_DIR}/extern/xnect/lib/vc140")

TARGET_LINK_LIBRARIES(${targetname}
${boost_python_LIBRARY}
${boost_system_LIBRARY}
${boost_thread_LIBRARY}
${caffe_LIBRARY}
${caffehdf5_LIBRARY}
${caffehdf5_hl_LIBRARY}
${caffezlib_LIBRARY}
${caffeproto_LIBRARY}
${gflags_LIBRARY}
${glog_LIBRARY}
${libopenblas_LIBRARY}
${libprotobuf_LIBRARY}
${mongoose_LIBRARY}
${OpenCV_CALIB3D_LIBRARY}
${OpenCV_CORE_LIBRARY}
${OpenCV_HIGHGUI_LIBRARY}
${OpenCV_IMGCODECS_LIBRARY}
${OpenCV_IMGPROC_LIBRARY}
#${OpenCV_VIDEOSTAB_LIBRARY}
${XNect_LIBRARY}
)

#C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\lib\x64\cudart_static.lib
#D:\caffe\build\lib\caffe.lib
#${Caffe_DEPENDENCIES_DIR}\x64\vc14\lib\opencv_videostab310.lib
#${Caffe_DEPENDENCIES_DIR}\x64\vc14\lib\opencv_superres310.lib
#${Caffe_DEPENDENCIES_DIR}\x64\vc14\lib\opencv_stitching310.lib
#${Caffe_DEPENDENCIES_DIR}\x64\vc14\lib\opencv_shape310.lib
#${Caffe_DEPENDENCIES_DIR}\x64\vc14\lib\opencv_photo310.lib
#${Caffe_DEPENDENCIES_DIR}\x64\vc14\lib\opencv_objdetect310.lib
#${Caffe_DEPENDENCIES_DIR}\x64\vc14\lib\opencv_calib3d310.lib
#xnect.lib
#mongoose.lib
#D:\caffe\build\lib\caffeproto.lib
#${Caffe_DEPENDENCIES_DIR}\lib\boost_system-vc140-mt-1_61.lib
#${Caffe_DEPENDENCIES_DIR}\lib\boost_thread-vc140-mt-1_61.lib
#${Caffe_DEPENDENCIES_DIR}\lib\boost_filesystem-vc140-mt-1_61.lib
#${Caffe_DEPENDENCIES_DIR}\lib\boost_chrono-vc140-mt-1_61.lib
#${Caffe_DEPENDENCIES_DIR}\lib\boost_date_time-vc140-mt-1_61.lib
#${Caffe_DEPENDENCIES_DIR}\lib\boost_atomic-vc140-mt-1_61.lib
#${Caffe_DEPENDENCIES_DIR}\lib\glog.lib
#${Caffe_DEPENDENCIES_DIR}\Lib\gflags.lib
#${Caffe_DEPENDENCIES_DIR}\lib\libprotobuf.lib
#${Caffe_DEPENDENCIES_DIR}\lib\caffehdf5_hl.lib
#${Caffe_DEPENDENCIES_DIR}\lib\caffehdf5.lib
#${Caffe_DEPENDENCIES_DIR}\cmake\..\lib\caffezlib.lib
#${Caffe_DEPENDENCIES_DIR}\lib\lmdb.lib
#${Caffe_DEPENDENCIES_DIR}\lib\leveldb.lib
#${Caffe_DEPENDENCIES_DIR}\cmake\..\lib\boost_date_time-vc140-mt-1_61.lib
#${Caffe_DEPENDENCIES_DIR}\cmake\..\lib\boost_filesystem-vc140-mt-1_61.lib
#${Caffe_DEPENDENCIES_DIR}\cmake\..\lib\boost_system-vc140-mt-1_61.lib
#${Caffe_DEPENDENCIES_DIR}\lib\snappy_static.lib
#${Caffe_DEPENDENCIES_DIR}\lib\caffezlib.lib
#C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\lib\x64\cudart.lib
#C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\lib\x64\curand.lib
#C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\lib\x64\cublas.lib
#${Caffe_DEPENDENCIES_DIR}\lib\libopenblas.dll.a
#C:\ProgramData\Anaconda3\envs\caffe\libs\python35.lib
#${Caffe_DEPENDENCIES_DIR}\lib\boost_python-vc140-mt-1_61.lib
#${Caffe_DEPENDENCIES_DIR}\x64\vc14\lib\opencv_features2d310.lib
#${Caffe_DEPENDENCIES_DIR}\x64\vc14\lib\opencv_ml310.lib
#${Caffe_DEPENDENCIES_DIR}\x64\vc14\lib\opencv_highgui310.lib
#${Caffe_DEPENDENCIES_DIR}\x64\vc14\lib\opencv_videoio310.lib
#${Caffe_DEPENDENCIES_DIR}\x64\vc14\lib\opencv_imgcodecs310.lib
#${Caffe_DEPENDENCIES_DIR}\x64\vc14\lib\opencv_flann310.lib
#${Caffe_DEPENDENCIES_DIR}\x64\vc14\lib\opencv_video310.lib
#${Caffe_DEPENDENCIES_DIR}\x64\vc14\lib\opencv_imgproc310.lib
#${Caffe_DEPENDENCIES_DIR}\x64\vc14\lib\opencv_core310.lib
