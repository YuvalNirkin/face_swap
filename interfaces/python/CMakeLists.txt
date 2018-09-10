# Load the package that SWIG uses to generate Python
find_package(PythonLibs REQUIRED)
find_package(Boost REQUIRED python numpy)

# Target
add_library(face_swap_py SHARED face_swap_py.cpp)
target_include_directories(face_swap_py PUBLIC
	${PYTHON_INCLUDE_DIRS}
	${Boost_INCLUDE_DIRS}
)
target_link_libraries(face_swap_py
	face_swap
	${PYTHON_LIBRARIES}
	${Boost_LIBRARIES}
)
set_target_properties(face_swap_py PROPERTIES
		PREFIX ""
	)
if(WIN32)
	set_target_properties(face_swap_py PROPERTIES
		SUFFIX ".pyd"
	)
endif()
if(MSVC)
  set_target_properties(face_swap_py
      PROPERTIES
      ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  )
else()
  set_target_properties(face_swap_py
      PROPERTIES
      ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
      )
endif()
set_target_properties(face_swap_py PROPERTIES DEBUG_POSTFIX "")

# Installations
install(TARGETS face_swap_py
	 EXPORT face_swap-targets
     RUNTIME DESTINATION interfaces/python COMPONENT python
     LIBRARY DESTINATION interfaces/python COMPONENT python
     ARCHIVE DESTINATION interfaces/python COMPONENT python)