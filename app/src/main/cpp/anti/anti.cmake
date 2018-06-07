get_filename_component(ANTI_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)

add_definitions(-D__CMAKE__)

set(ANTI_INC ${ANTI_CMAKE_DIR})

set(ANTI_SRC
        ${ANTI_CMAKE_DIR}/aib.c
        ${ANTI_CMAKE_DIR}/array.c
        ${ANTI_CMAKE_DIR}/covdet.c
        ${ANTI_CMAKE_DIR}/dsift.c
        ${ANTI_CMAKE_DIR}/fisher.c
        ${ANTI_CMAKE_DIR}/generic.c
        ${ANTI_CMAKE_DIR}/getopt_long.c
        ${ANTI_CMAKE_DIR}/gmm.c
        ${ANTI_CMAKE_DIR}/hikmeans.c
        ${ANTI_CMAKE_DIR}/hog.c
        ${ANTI_CMAKE_DIR}/homkermap.c
        ${ANTI_CMAKE_DIR}/host.c
        ${ANTI_CMAKE_DIR}/ikmeans.c
        ${ANTI_CMAKE_DIR}/imopv_sse2.c
        ${ANTI_CMAKE_DIR}/imopv.c
        ${ANTI_CMAKE_DIR}/kdtree.c
        ${ANTI_CMAKE_DIR}/kmeans.c
        ${ANTI_CMAKE_DIR}/lbp.c
        ${ANTI_CMAKE_DIR}/liop.c
        ${ANTI_CMAKE_DIR}/mathop_avx.c
        ${ANTI_CMAKE_DIR}/mathop_sse2.c
        ${ANTI_CMAKE_DIR}/mathop.c
        ${ANTI_CMAKE_DIR}/mser.c
        ${ANTI_CMAKE_DIR}/pgm.c
        ${ANTI_CMAKE_DIR}/quickshift.c
        ${ANTI_CMAKE_DIR}/random.c
        ${ANTI_CMAKE_DIR}/rodrigues.c
        ${ANTI_CMAKE_DIR}/scalespace.c
        ${ANTI_CMAKE_DIR}/sift.c
        ${ANTI_CMAKE_DIR}/slic.c
        ${ANTI_CMAKE_DIR}/stringop.c
        ${ANTI_CMAKE_DIR}/svm.c
        ${ANTI_CMAKE_DIR}/svmdataset.c
        ${ANTI_CMAKE_DIR}/vlad.c)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DVL_DISABLE_SSE2 -DVL_DISABLE_AVX")
