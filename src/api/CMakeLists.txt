# Shared library `popso` with the C API (include/popso.h); its tests (C and Python).
# Included from the top-level CMakeLists.txt when POPSO_BUILD_SHARED is ON.

add_library(popso_shared SHARED ${CMAKE_CURRENT_SOURCE_DIR}/popso_c.cpp)
target_include_directories(popso_shared PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(popso_shared PRIVATE milp_lib)
target_compile_definitions(popso_shared PRIVATE POPSO_BUILDING_LIBRARY POPSO_VERSION="${PROJECT_VERSION}")
target_compile_options(popso_shared PRIVATE ${MILP_WARNINGS})
# libpopso.so / libpopso.dylib / popso.dll next to the popso executable; only the C API is exported.
set_target_properties(popso_shared PROPERTIES
  OUTPUT_NAME popso
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
if(POPSO_SANITIZE AND NOT MSVC)
  # milp_lib's sanitizer flags reach this library's own link only (milp_lib is linked PRIVATE).
  # On Linux a shared library does not carry the sanitizer runtimes, the executable loading it
  # must: its consumers link with the same flags.
  target_link_options(popso_shared INTERFACE -fsanitize=address,undefined)
endif()

if(POPSO_BUILD_TESTS)
  # The C test is compiled as C99, which also checks that the header is plain C.
  enable_language(C)
  add_executable(popso_c_api_test ${CMAKE_SOURCE_DIR}/tests/c_api/test_c_api.c)
  set_target_properties(popso_c_api_test PROPERTIES
    C_STANDARD 99
    C_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  target_link_libraries(popso_c_api_test PRIVATE popso_shared)
  target_compile_definitions(popso_c_api_test PRIVATE POPSO_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
                             POPSO_BINARY_DIR="${CMAKE_BINARY_DIR}")
  if(POPSO_SANITIZE AND NOT MSVC)
    target_compile_options(popso_c_api_test PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    # Link with the C++ driver: the UBSan checks of the C++ library need the C++ part of its
    # runtime, which the C driver leaves out.
    set_target_properties(popso_c_api_test PROPERTIES LINKER_LANGUAGE CXX)
  endif()
  add_test(NAME c_api COMMAND popso_c_api_test)

  # Python bindings (ctypes; no build step), tested against this build's library.
  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_Interpreter_FOUND)
    add_test(NAME python
             COMMAND ${Python3_EXECUTABLE} -m unittest discover -s ${CMAKE_SOURCE_DIR}/python/tests
             WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    set_tests_properties(python PROPERTIES
      ENVIRONMENT "POPSO_LIBRARY=$<TARGET_FILE:popso_shared>")
  endif()
endif()
