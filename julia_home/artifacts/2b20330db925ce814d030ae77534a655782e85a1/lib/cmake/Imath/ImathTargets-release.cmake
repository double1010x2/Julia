#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "Imath::Imath" for configuration "Release"
set_property(TARGET Imath::Imath APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(Imath::Imath PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libImath-3_1.so.29.1.0"
  IMPORTED_SONAME_RELEASE "libImath-3_1.so.29"
  )

list(APPEND _IMPORT_CHECK_TARGETS Imath::Imath )
list(APPEND _IMPORT_CHECK_FILES_FOR_Imath::Imath "${_IMPORT_PREFIX}/lib/libImath-3_1.so.29.1.0" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
