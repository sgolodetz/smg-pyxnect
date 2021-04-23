##################
# UseXNect.cmake #
##################

SET(Caffe_ROOT_DIR "D:/caffe" CACHE PATH "The Caffe root directory")

SET(Caffe_DEPENDENCIES_DIR
"$ENV{HOMEDRIVE}$ENV{HOMEPATH}/.caffe/dependencies/libraries_v140_x64_py35_1.1.0/libraries"
CACHE PATH "The Caffe dependencies directory"
)

SET(XNect_ROOT_DIR "D:/xnect" CACHE PATH "The XNect root directory")

SET(XNect_INCLUDE_DIRS
"${Caffe_DEPENDENCIES_DIR}/include"
"${Caffe_DEPENDENCIES_DIR}/include/boost-1_61"
"${Caffe_DEPENDENCIES_DIR}/include/opencv"
"${Caffe_ROOT_DIR}/build/include"
"${Caffe_ROOT_DIR}/include"
"${XNect_ROOT_DIR}/extern/eigen"
"${XNect_ROOT_DIR}/extern/eigen/unsupported"
"${XNect_ROOT_DIR}/extern/mongoose"
"${XNect_ROOT_DIR}/extern/mongoose/mongoose"
"${XNect_ROOT_DIR}/extern/rtpose"
"${XNect_ROOT_DIR}/extern/xnect/include"
)

INCLUDE_DIRECTORIES(${XNect_INCLUDE_DIRS})

ADD_DEFINITIONS(-DBOOST_ALL_NO_LIB)
