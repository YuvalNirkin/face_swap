# Target
add_executable(face_swap_single2many face_swap_single2many.cpp)
target_include_directories(face_swap_single2many PRIVATE 
	${Boost_INCLUDE_DIRS}
)
target_link_libraries(face_swap_single2many PRIVATE
	face_swap
	${Boost_LIBRARIES}
)

# Installations
install(TARGETS face_swap_single2many EXPORT face_swap-targets DESTINATION bin COMPONENT app)
install(FILES face_swap_single2many.cfg DESTINATION bin COMPONENT app)
