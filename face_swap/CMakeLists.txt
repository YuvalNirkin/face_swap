# Source
set(SRC
	basel_3dmm.cpp
	cnn_3dmm.cpp
	cnn_3dmm_expr.cpp
	face_seg.cpp
	face_swap_engine_impl.cpp
	face_swap_c_interface.cpp
	render_utilities.cpp
	utilities.cpp
	face_detection_landmarks.cpp
	landmarks_utilities.cpp
	segmentation_utilities.cpp
)
set(HDR
	face_swap/basel_3dmm.h
	face_swap/cnn_3dmm.h
	face_swap/cnn_3dmm_expr.h
	face_swap/face_seg.h
	face_swap/face_swap_engine.h
	face_swap/face_swap_engine_impl.h
	face_swap/face_swap_c_interface.h
	face_swap/render_utilities.h
	face_swap/utilities.h
	face_swap/face_detection_landmarks.h
	face_swap/landmarks_utilities.h
	face_swap/segmentation_utilities.h
)

if(PROTOBUF_FOUND)
	set(PROTO_FILES face_data.proto)
	protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILES})
	set(SRC ${SRC} ${PROTO_SRCS} ${PROTO_HDRS} ${PROTO_FILES})
	add_definitions(-DWITH_PROTOBUF)
endif()

# Target
add_library(face_swap ${LIB_TYPE} ${SRC} ${HDR})
target_include_directories(face_swap PUBLIC
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
	$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
	${OpenCV_INCLUDE_DIRS}
	${Caffe_INCLUDE_DIRS}
	${HDF5_INCLUDE_DIRS}
)
target_link_libraries(face_swap PUBLIC
	iris_sfs
	${OpenCV_LIBS}
	dlib::dlib
	${Caffe_LIBRARIES}
	${HDF5_LIBRARIES}
)
if(PROTOBUF_FOUND)
	target_include_directories(face_swap PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
	target_include_directories(face_swap PUBLIC ${PROTOBUF_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
	target_link_libraries(face_swap PUBLIC ${PROTOBUF_LIBRARIES} ${Boost_LIBRARIES})
endif()

# Add export header
include (GenerateExportHeader)
if(UNIX)
	set(USE_COMPILER_HIDDEN_VISIBILITY OFF)
endif(UNIX)
generate_export_header(face_swap EXPORT_FILE_NAME ${PROJECT_BINARY_DIR}/face_swap/face_swap_export.h)
install(FILES ${PROJECT_BINARY_DIR}/face_swap/face_swap_export.h DESTINATION include/face_swap)


# Installations
install(TARGETS face_swap
	 EXPORT face_swap-targets
     RUNTIME DESTINATION bin COMPONENT dev
     LIBRARY DESTINATION lib COMPONENT dev
     ARCHIVE DESTINATION lib COMPONENT dev)
install(FILES ${HDR} DESTINATION include/face_swap)
