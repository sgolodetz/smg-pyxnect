#################
# UseCUDA.cmake #
#################

FIND_PACKAGE(CUDA QUIET)

OPTION(WITH_CUDA "Build with CUDA support?" ${CUDA_FOUND})

IF(WITH_CUDA)
  SET(CUDA_SEPARABLE_COMPILATION ON CACHE BOOL "" FORCE)

  # Auto-detect the CUDA compute capability.
  SET(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
  IF(NOT DEFINED CUDA_COMPUTE_CAPABILITY)
    INCLUDE("${CMAKE_MODULE_PATH}/CUDACheckCompute.cmake")
  ENDIF()

  # Set the compute capability flags.
  FOREACH(compute_capability ${CUDA_COMPUTE_CAPABILITY})
    LIST(APPEND CUDA_NVCC_FLAGS --generate-code;arch=compute_${compute_capability},code=compute_${compute_capability})
    LIST(APPEND CUDA_NVCC_FLAGS --generate-code;arch=compute_${compute_capability},code=sm_${compute_capability})
  ENDFOREACH()

  # If on Windows, make it possible to enable GPU debug information.
  IF(MSVC_IDE)
    OPTION(ENABLE_CUDA_DEBUGGING "Enable CUDA debugging?" OFF)
    IF(ENABLE_CUDA_DEBUGGING)
      SET(CUDA_NVCC_FLAGS -G; ${CUDA_NVCC_FLAGS})
    ENDIF()
  ENDIF()

  # If on Mac OS X 10.9 (Mavericks), make sure everything compiles and links using the correct C++ Standard Library.
  IF(${CMAKE_SYSTEM} MATCHES "Darwin-13.")
    SET(CUDA_HOST_COMPILER /usr/bin/clang)
    SET(CUDA_NVCC_FLAGS -Xcompiler -stdlib=libstdc++; -Xlinker -stdlib=libstdc++; ${CUDA_NVCC_FLAGS})
  ENDIF()

  # If on Linux:
  IF(${CMAKE_SYSTEM} MATCHES "Linux")
    # Make sure that C++11 support is enabled when compiling with nvcc. From CMake 3.5 onwards,
    # the host flag -std=c++11 is automatically propagated to nvcc. Manually setting it prevents
    # the project from building.
    IF(${CMAKE_VERSION} VERSION_LESS 3.5)
      SET(CUDA_NVCC_FLAGS -std=c++11; ${CUDA_NVCC_FLAGS})
    ENDIF()
  ENDIF()

  # Disable some annoying nvcc warnings.
  IF(MSVC_IDE)
    SET(CUDA_NVCC_FLAGS --Wno-deprecated-declarations ; ${CUDA_NVCC_FLAGS})
    SET(CUDA_NVCC_FLAGS -Xcudafe "--diag_suppress=bad_friend_decl" ; -Xcudafe "--diag_suppress=base_class_has_different_dll_interface" ; -Xcudafe "--diag_suppress=code_is_unreachable" ; -Xcudafe "--diag_suppress=dll_interface_conflict_dllexport_assumed" ; -Xcudafe "--diag_suppress=field_without_dll_interface" ; -Xcudafe "--diag_suppress=overloaded_function_linkage" ; -Xcudafe "--diag_suppress=useless_type_qualifier_on_return_type" ; ${CUDA_NVCC_FLAGS})
  ELSE()
    SET(CUDA_NVCC_FLAGS -Xcudafe "--diag_suppress=cc_clobber_ignored" ; -Xcudafe "--diag_suppress=set_but_not_used" ; ${CUDA_NVCC_FLAGS})
  ENDIF()

  INCLUDE_DIRECTORIES(${CUDA_TOOLKIT_INCLUDE})
  ADD_DEFINITIONS(-DWITH_CUDA)
ENDIF()
