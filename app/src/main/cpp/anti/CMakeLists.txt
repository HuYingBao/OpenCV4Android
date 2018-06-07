include_directories(../lib_vlfeat/)

find_package(OpenCV REQUIRED)
find_package(Threads)
add_executable(result_example daxpy.c ddot.c dnrm2.c 
dscal.c linear.cpp live_demo.cpp tron.cpp 
 )
 target_link_libraries(result_example ${OpenCV_LIBS} vlfeat_slic pthread)
