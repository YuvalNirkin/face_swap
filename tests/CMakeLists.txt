# Find all test cpp source files
file(GLOB TEST_CPP_SRC RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

# For each test cpp source file	
foreach(TEST_FILE_NAME ${TEST_CPP_SRC})
    message(STATUS "Processing test file: ${TEST_FILE_NAME}")

    # Generate target name
    string(REGEX REPLACE ".cpp\$" "" TEST_NAME "${TEST_FILE_NAME}")
    message(STATUS "Adding test target: ${TEST_NAME}")
	
	# Target
	add_executable(${TEST_NAME} ${TEST_FILE_NAME})
	target_include_directories(${TEST_NAME} PRIVATE 
		${Boost_INCLUDE_DIRS}
	)
	target_link_libraries(${TEST_NAME} PRIVATE
		face_swap
		${Boost_LIBRARIES}
	)
	set_property(TARGET ${TEST_NAME} PROPERTY FOLDER "tests")

	# Installations
	install(TARGETS ${TEST_NAME} EXPORT face_swap-targets DESTINATION tests COMPONENT tests)

endforeach(TEST_FILE_NAME)