##################
# LinkCUDA.cmake #
##################

IF(WITH_CUDA AND NOT APPLE)
  TARGET_LINK_LIBRARIES(${targetname}
    ${CUDA_CUDA_LIBRARY}
    ${CUDA_cublas_LIBRARY}
    ${CUDA_cudart_static_LIBRARY}
    ${CUDA_curand_LIBRARY}
  )
ENDIF()
