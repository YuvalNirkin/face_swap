# Source
set(SRC
	basel_3dmm.cpp
	cnn_3dmm.cpp
	cnn_3dmm_expr.cpp
	face_renderer.cpp
	face_swap.cpp
	utilities.cpp	
)
set(HDR
	face_swap/basel_3dmm.h
	face_swap/cnn_3dmm.h
	face_swap/cnn_3dmm_expr.h
	face_swap/face_renderer.h
	face_swap/face_swap.h
	face_swap/utilities.h
)

# Target
add_library(face_swap ${SRC} ${HDR})
target_include_directories(face_swap PUBLIC
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
	${Boost_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS}
	${FIND_FACE_LANDMARKS_INCLUDE_DIRS}
	${face_segmentation_INCLUDE_DIRS}
	${Caffe_INCLUDE_DIRS}
	${HDF5_INCLUDE_DIRS}
)
target_link_libraries(face_swap PUBLIC
	cv_gl
	iris_sfs
	${Boost_LIBRARIES}
	${OpenCV_LIBS}
	${FIND_FACE_LANDMARKS_LIBRARIES}
	${face_segmentation_LIBRARIES}
	${Caffe_LIBRARIES}
	${HDF5_LIBRARIES}
)

# Installations
install(TARGETS face_swap
	 EXPORT face_swap-targets
     RUNTIME DESTINATION bin COMPONENT dev
     LIBRARY DESTINATION lib COMPONENT dev
     ARCHIVE DESTINATION lib COMPONENT dev)
install(FILES ${HDR} DESTINATION include/face_swap)
