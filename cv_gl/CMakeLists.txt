# Source
SET(CVGL_SOURCE
    FBRender.cpp
)

SET(CVGL_HEADERS
    cvgl.h
    FBRender.h	
    Material.h
    SimpleMesh.h
    Texture.h	
)

# Target
add_library(cv_gl ${CVGL_SOURCE} ${CVGL_HEADERS})
target_include_directories(cv_gl PUBLIC
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
	${OpenCV_INCLUDE_DIRS}
	${OPENGL_INCLUDE_DIR}
	${GLEW_INCLUDE_DIRS}
)
target_link_libraries(cv_gl PUBLIC
	${OpenCV_LIBS}
	${OPENGL_LIBRARIES}
	${GLEW_LIBRARIES}
)

# Installations
install(TARGETS cv_gl
	 EXPORT face_swap-targets
     RUNTIME DESTINATION bin COMPONENT dev
     LIBRARY DESTINATION lib COMPONENT dev
     ARCHIVE DESTINATION lib COMPONENT dev)
#install(FILES ${CVGL_HEADERS} DESTINATION include)
