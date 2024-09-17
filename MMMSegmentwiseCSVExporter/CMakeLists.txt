set(TARGET_NAME "MMMSegmentwiseCSVExporter")

if(NOT(MMMCore_FOUND AND MMMTools_FOUND))
    message(STATUS "Required libraries not found for ${TARGET_NAME}")
    return()
endif()

set(LIBRARIES
    MMMCore
    MMMMotionPrimitiveLib
    MMMViewerLib
    stdc++fs
)

set(SOURCES
    MMMSegmentwiseCSVExporter.cpp
    ../../common/HandleMotionsWithoutModel.cpp
)

set(HEADERS
    MMMSegmentwiseCSVExporterConfiguration.h
    ../../common/HandleMotionsWithoutModel.h
)

add_executable(${TARGET_NAME} ${SOURCES} ${HEADERS})
target_link_libraries(${TARGET_NAME} PRIVATE ${LIBRARIES})

