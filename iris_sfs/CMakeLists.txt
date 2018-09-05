# Source
SET(IRIS_SFS_SOURCE
	utility.cpp
    epnp.cpp
    BaselFace.cpp
    BaselFaceEstimator.cpp
    FaceServices2.cpp
)

SET(IRIS_SFS_HEADERS
    utility.h	
    epnp.h
    BaselFace.h
    BaselFaceEstimator.h
    FaceServices2.h
)

# Target
add_library(iris_sfs STATIC ${IRIS_SFS_SOURCE} ${IRIS_SFS_HEADERS})
target_include_directories(iris_sfs PUBLIC 
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
	${OpenCV_INCLUDE_DIRS}
	${EIGEN3_INCLUDE_DIR}
)
target_link_libraries(iris_sfs PUBLIC
	${OpenCV_LIBS}
)

# Installations
install(TARGETS iris_sfs
	 EXPORT face_swap-targets
     RUNTIME DESTINATION bin COMPONENT dev
     LIBRARY DESTINATION lib COMPONENT dev
     ARCHIVE DESTINATION lib COMPONENT dev)
#install(FILES ${IRIS_SFS_HEADERS} DESTINATION include)
