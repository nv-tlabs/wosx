cmake_minimum_required(VERSION 3.15...3.27)
project(wosx)

option(WOSX_ENABLE_GPU_SUPPORT "Enable GPU support" ON)
option(WOSX_BUILD_DEMO_APPS "Build demo apps" ON)
option(WOSX_BUILD_BINDINGS "Build bindings" OFF)

################################################################################
if(WOSX_BUILD_BINDINGS)
    if (CMAKE_VERSION VERSION_LESS 3.18)
        set(DEV_MODULE Development)
    else()
        set(DEV_MODULE Development.Module)
    endif()
    find_package(Python 3.9
        REQUIRED COMPONENTS Interpreter ${DEV_MODULE}
        OPTIONAL_COMPONENTS Development.SABIModule)
endif()

# submodule check
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

if(CMAKE_BUILD_TYPE MATCHES RELEASE)
    add_definitions(-DNDEBUG)
endif()

################################################################################
# add dependencies
get_directory_property(hasParent PARENT_DIRECTORY)

set(TBB_TEST          OFF CACHE BOOL " " FORCE)
set(TBB_INSTALL       OFF CACHE BOOL " " FORCE)
if(APPLE)
    set(TCM_BUILD OFF CACHE BOOL "Disable unsupported oneTBB TCM on macOS" FORCE)
endif()
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/deps/tbb)

if(hasParent)
    set(WOSX_TBB_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/deps/tbb/include PARENT_SCOPE)
    set(WOSX_NANOFLANN_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/deps/nanoflann/include PARENT_SCOPE)
else()
    set(WOSX_TBB_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/deps/tbb/include)
    set(WOSX_NANOFLANN_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/deps/nanoflann/include)
endif()

set(FCPW_ENABLE_GPU_SUPPORT ${WOSX_ENABLE_GPU_SUPPORT} CACHE BOOL "Enable GPU support" FORCE)
set(FCPW_USE_ENOKI OFF CACHE BOOL "Enable CPU vectorization" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/deps/fcpw)

list(APPEND WOSX_DEPS_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/deps/pcg32)
list(APPEND WOSX_DEPS_INCLUDES ${WOSX_TBB_INCLUDES})
list(APPEND WOSX_DEPS_INCLUDES ${WOSX_NANOFLANN_INCLUDES})
list(APPEND WOSX_DEPS_INCLUDES ${FCPW_EIGEN_INCLUDES})
if(WOSX_ENABLE_GPU_SUPPORT)
    list(APPEND WOSX_DEPS_INCLUDES ${FCPW_SLANG_RHI_INCLUDES})
endif()

################################################################################
# build library
add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE $<BUILD_INTERFACE:${${PROJECT_NAME}_SOURCE_DIR}/include> ${WOSX_DEPS_INCLUDES})
target_link_libraries(${PROJECT_NAME} INTERFACE fcpw TBB::tbb)
if(WOSX_ENABLE_GPU_SUPPORT)
    target_compile_definitions(${PROJECT_NAME} INTERFACE WOSX_USE_GPU)
    target_link_libraries(${PROJECT_NAME} INTERFACE ${FCPW_SLANG_RHI_LIBRARY})
endif()
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)

################################################################################
# build demo apps
if(WOSX_BUILD_DEMO_APPS)
    add_subdirectory(demo_apps)
endif()

# build Python bindings
if(WOSX_BUILD_BINDINGS)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/deps/nanobind)
    add_subdirectory(python)
endif()
