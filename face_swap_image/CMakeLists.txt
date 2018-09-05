# Target
add_executable(face_swap_image face_swap_image.cpp)
target_include_directories(face_swap_image PRIVATE 
	${Boost_INCLUDE_DIRS}
)
target_link_libraries(face_swap_image PRIVATE
	face_swap
	${Boost_LIBRARIES}
)

# Installations
install(TARGETS face_swap_image EXPORT face_swap-targets DESTINATION bin COMPONENT app)
install(FILES face_swap_image.cfg DESTINATION bin COMPONENT app)
