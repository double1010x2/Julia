#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "imgui-cpp" for configuration "Release"
set_property(TARGET imgui-cpp APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(imgui-cpp PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libimgui-cpp.so"
  IMPORTED_SONAME_RELEASE "libimgui-cpp.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS imgui-cpp )
list(APPEND _IMPORT_CHECK_FILES_FOR_imgui-cpp "${_IMPORT_PREFIX}/lib/libimgui-cpp.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
