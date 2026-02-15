cmake_minimum_required(VERSION 3.16)
project(umcp_accel VERSION 1.0.0 LANGUAGES CXX)

# ─── C++17 required ───────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ─── Compiler optimizations ──────────────────────────────────────────
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-O3 -Wall -Wextra -Wpedantic)
    # Enable architecture-specific SIMD
    if(NOT DEFINED UMCP_PORTABLE)
        add_compile_options(-march=native)
    endif()
elseif(MSVC)
    add_compile_options(/O2 /W4)
endif()

# ─── Static library (core C++ implementation) ────────────────────────
add_library(umcp_core STATIC
    src/kernel.cpp
    src/seam.cpp
    src/integrity.cpp
)
target_include_directories(umcp_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ─── Optional OpenSSL for SHA-256 ────────────────────────────────────
find_package(OpenSSL QUIET)
if(OpenSSL_FOUND)
    target_link_libraries(umcp_core PUBLIC OpenSSL::Crypto)
    target_compile_definitions(umcp_core PUBLIC UMCP_HAS_OPENSSL)
    message(STATUS "OpenSSL found — using hardware-accelerated SHA-256")
else()
    message(STATUS "OpenSSL not found — using built-in SHA-256")
endif()

# ─── pybind11 Python extension module ────────────────────────────────
option(UMCP_BUILD_PYTHON "Build Python bindings" ON)
if(UMCP_BUILD_PYTHON)
    # Try to find pybind11 via pip-installed location or system
    find_package(pybind11 QUIET)
    if(NOT pybind11_FOUND)
        # Fallback: fetch from GitHub
        include(FetchContent)
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG        v2.12.0
        )
        FetchContent_MakeAvailable(pybind11)
    endif()

    pybind11_add_module(umcp_accel bindings/py_umcp.cpp)
    target_link_libraries(umcp_accel PRIVATE umcp_core)

    # Install to the umcp package directory
    install(TARGETS umcp_accel
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/umcp
    )
endif()

# ─── C++ tests (Catch2) ─────────────────────────────────────────────
option(UMCP_BUILD_TESTS "Build C++ unit tests" ON)
if(UMCP_BUILD_TESTS)
    enable_testing()

    # Fetch Catch2 if not found
    find_package(Catch2 3 QUIET)
    if(NOT Catch2_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            Catch2
            GIT_REPOSITORY https://github.com/catchorg/Catch2.git
            GIT_TAG        v3.5.2
        )
        FetchContent_MakeAvailable(Catch2)
    endif()

    add_executable(test_umcp_kernel tests/test_kernel.cpp)
    target_link_libraries(test_umcp_kernel PRIVATE umcp_core Catch2::Catch2WithMain)

    include(CTest)
    include(Catch)
    catch_discover_tests(test_umcp_kernel)
endif()

# ─── Summary ────────────────────────────────────────────────────────
message(STATUS "")
message(STATUS "UMCP C++ Accelerator Configuration:")
message(STATUS "  C++ Standard:     ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "  Python Bindings:  ${UMCP_BUILD_PYTHON}")
message(STATUS "  C++ Tests:        ${UMCP_BUILD_TESTS}")
message(STATUS "  OpenSSL SHA-256:  ${OpenSSL_FOUND}")
message(STATUS "")
