cmake_minimum_required(VERSION 3.15)
project(monte_carlo_option_pricer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

find_package(Threads REQUIRED)

add_executable(monte_carlo_pricer src/main.cpp)
target_include_directories(monte_carlo_pricer PRIVATE include)
target_link_libraries(monte_carlo_pricer PRIVATE Threads::Threads)
