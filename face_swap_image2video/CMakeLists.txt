# Target
add_executable(face_swap_image2video face_swap_image2video.cpp)
target_include_directories(face_swap_image2video PRIVATE 
	${Boost_INCLUDE_DIRS}
)
target_link_libraries(face_swap_image2video PRIVATE
	face_swap
	${Boost_LIBRARIES}
)

# Installations
install(TARGETS face_swap_image2video EXPORT face_swap-targets DESTINATION bin COMPONENT app)
install(FILES face_swap_image2video.cfg DESTINATION bin COMPONENT app)
