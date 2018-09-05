# Target
add_executable(face_swap_batch face_swap_batch.cpp)
target_include_directories(face_swap_batch PRIVATE 
	${Boost_INCLUDE_DIRS}
)
target_link_libraries(face_swap_batch PRIVATE
	face_swap
	${Boost_LIBRARIES}
)

# Installations
install(TARGETS face_swap_batch EXPORT face_swap-targets DESTINATION bin COMPONENT app)
install(FILES face_swap_batch.cfg DESTINATION bin COMPONENT app)
