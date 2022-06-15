cmake_minimum_required(VERSION 3.6)

project(make_larclip_data CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/")

set( larlite_DIR $ENV{LARLITE_LIBDIR}/cmake/larlite )
set( larcv_DIR $ENV{LARCV_LIBDIR}/cmake/larcv )
find_package( ROOT REQUIRED )
find_package( larlite REQUIRED )
find_package( larcv REQUIRED )

#add_executable
add_executable(make_larclip_data make_larclip_data.cxx)

# To set the C++ standard to be used, only one of the following two lines should
# be un-commented.
#set_property(TARGET demo PROPERTY CXX_STANDARD 14)
set_property(TARGET make_larclip_data PROPERTY CXX_STANDARD 17)

include_directories( ${LARLITE_INCLUDE_DIR} )
include_directories( ${LARCV_INCLUDE_DIR} )
include_directories( ${ROOT_INCLUDE_DIRS} )

set( LINK_LIBS larlite_LArUtil larlite_DataFormat larlite_Base LArCVCoreDataFormat LArCVCoreBase ${ROOT_LIBRARIES} )

target_link_libraries(make_larclip_data ${LINK_LIBS})

