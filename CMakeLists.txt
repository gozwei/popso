cmake_minimum_required(VERSION 3.20)
project(popso LANGUAGES CXX VERSION 0.2.1)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(POPSO_NATIVE "Optimize for the host CPU (-march=native)" OFF)
option(POPSO_SANITIZE "Build with address/undefined sanitizers" OFF)
option(POPSO_BUILD_TESTS "Build unit tests" ON)
option(POPSO_BUILD_SHARED "Build the shared library popso with the C API" ON)
option(POPSO_WERROR "Treat compiler warnings as errors (CI)" OFF)

find_package(Threads REQUIRED)

if(MSVC)
  # Same warning set as gcc/clang: implicit narrowing conversions (C4244, C4267; -Wconversion is
  # not enabled either) and unused parameters (C4100) are not reported.
  set(MILP_WARNINGS /W4 /permissive- /wd4100 /wd4244 /wd4267)
  if(POPSO_WERROR)
    list(APPEND MILP_WARNINGS /WX)
  endif()
else()
  set(MILP_WARNINGS -Wall -Wextra -Wpedantic -Wshadow -Wno-unused-parameter)
  if(POPSO_WERROR)
    list(APPEND MILP_WARNINGS -Werror)
  endif()
endif()

file(GLOB_RECURSE MILP_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/milp/*.cpp)
add_library(milp_lib STATIC ${MILP_SOURCES})
target_include_directories(milp_lib PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_compile_options(milp_lib PRIVATE ${MILP_WARNINGS})
target_link_libraries(milp_lib PUBLIC Threads::Threads)
# PIC so that the shared library (src/api) can link it; hidden so that only the C API is exported.
set_target_properties(milp_lib PROPERTIES POSITION_INDEPENDENT_CODE ON CXX_VISIBILITY_PRESET hidden
                      VISIBILITY_INLINES_HIDDEN ON)
if(WIN32)
  target_link_libraries(milp_lib PUBLIC psapi)
  # std::getenv / std::fopen are fine here; keep windows.h from defining min/max macros.
  target_compile_definitions(milp_lib PUBLIC _CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
endif()
if(MSVC)
  # Sources are plain ASCII but say so; /bigobj for the large search translation unit.
  target_compile_options(milp_lib PUBLIC /utf-8 /bigobj)
endif()

if(POPSO_NATIVE AND NOT MSVC)
  target_compile_options(milp_lib PUBLIC -march=native)
endif()
if(POPSO_SANITIZE AND NOT MSVC)
  target_compile_options(milp_lib PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(milp_lib PUBLIC -fsanitize=address,undefined)
endif()

add_executable(popso apps/popso_main.cpp)
target_link_libraries(popso PRIVATE milp_lib)
target_compile_definitions(popso PRIVATE POPSO_VERSION="${PROJECT_VERSION}")
target_compile_options(popso PRIVATE ${MILP_WARNINGS})

if(POPSO_BUILD_TESTS)
  enable_testing()
  file(GLOB MILP_TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/tests/*.cpp)
  add_executable(popso_tests ${MILP_TEST_SOURCES})
  target_link_libraries(popso_tests PRIVATE milp_lib)
  target_compile_definitions(popso_tests PRIVATE MILP_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
  add_test(NAME unit COMMAND popso_tests)
endif()

if(POPSO_BUILD_SHARED)
  add_subdirectory(src/api)
endif()
