#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "GR::GKS" for configuration "Release"
set_property(TARGET GR::GKS APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(GR::GKS PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libGKS.so"
  IMPORTED_SONAME_RELEASE "libGKS.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS GR::GKS )
list(APPEND _IMPORT_CHECK_FILES_FOR_GR::GKS "${_IMPORT_PREFIX}/lib/libGKS.so" )

# Import target "GR::GR" for configuration "Release"
set_property(TARGET GR::GR APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(GR::GR PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libGR.so"
  IMPORTED_SONAME_RELEASE "libGR.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS GR::GR )
list(APPEND _IMPORT_CHECK_FILES_FOR_GR::GR "${_IMPORT_PREFIX}/lib/libGR.so" )

# Import target "GR::GR3" for configuration "Release"
set_property(TARGET GR::GR3 APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(GR::GR3 PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "GR::GR"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libGR3.so"
  IMPORTED_SONAME_RELEASE "libGR3.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS GR::GR3 )
list(APPEND _IMPORT_CHECK_FILES_FOR_GR::GR3 "${_IMPORT_PREFIX}/lib/libGR3.so" )

# Import target "GR::GRM" for configuration "Release"
set_property(TARGET GR::GRM APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(GR::GRM PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "GR::GR;GR::GR3"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libGRM.so"
  IMPORTED_SONAME_RELEASE "libGRM.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS GR::GRM )
list(APPEND _IMPORT_CHECK_FILES_FOR_GR::GRM "${_IMPORT_PREFIX}/lib/libGRM.so" )

# Import target "GR::qt5gr" for configuration "Release"
set_property(TARGET GR::qt5gr APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(GR::qt5gr PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libqt5gr.so"
  IMPORTED_SONAME_RELEASE "libqt5gr.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS GR::qt5gr )
list(APPEND _IMPORT_CHECK_FILES_FOR_GR::qt5gr "${_IMPORT_PREFIX}/lib/libqt5gr.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
