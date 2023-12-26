cmake_minimum_required(VERSION 3.22)

project(lvk_android_native)

set(LVK_DEPLOY_DEPS ON)
set(LVK_WITH_GLFW OFF)
set(LVK_WITH_SAMPLES ON)
set(LVK_WITH_TRACY OFF)
set(LVK_WITH_WAYLAND OFF)
set(LVK_WITH_IMPLOT ON)
add_subdirectory(LVK)

include_directories("LVK/third-party/deps/src/glm")
include_directories("LVK/third-party/deps/src/stb")

add_library(${CMAKE_PROJECT_NAME} SHARED
        NativeActivity.cpp
        ${ANDROID_NDK}/sources/android/native_app_glue/android_native_app_glue.c)
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
        ${ANDROID_NDK}/sources/android/native_app_glue)
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
        LVK/third-party/deps/src/3D-Graphics-Rendering-Cookbook)
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
        LVK/third-party/deps/src/ktx-software/include)
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
        LVK/third-party/deps/src/taskflow)

set(KTX_FEATURE_DOC            OFF CACHE BOOL "")
set(KTX_FEATURE_GL_UPLOAD      OFF CACHE BOOL "")
set(KTX_FEATURE_JNI            OFF CACHE BOOL "")
set(KTX_FEATURE_KTX1            ON CACHE BOOL "")
set(KTX_FEATURE_KTX2            ON CACHE BOOL "")
set(KTX_FEATURE_LOADTEST_APPS  OFF CACHE BOOL "")
set(KTX_FEATURE_STATIC_LIBRARY  ON CACHE BOOL "")
set(KTX_FEATURE_TESTS          OFF CACHE BOOL "")
set(KTX_FEATURE_TOOLS          OFF CACHE BOOL "")
set(KTX_FEATURE_VK_UPLOAD      OFF CACHE BOOL "")
# cmake-format: on
add_subdirectory(LVK/third-party/deps/src/ktx-software)

target_link_libraries(${CMAKE_PROJECT_NAME}
  android
  log
  LVKLibrary
  ktx)