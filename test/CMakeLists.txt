cmake_minimum_required(VERSION 3.20)

link_libraries(${PROJECT_NAME}::${PROJECT_NAME} GTest::gtest_main)

add_executable(test_covariance_setting test_covariance_setting.cpp)
gtest_discover_tests(test_covariance_setting)

add_executable(test_ornstein_uhlenbeck test_ornstein_uhlenbeck.cpp cov_fnc.cpp)
gtest_discover_tests(test_ornstein_uhlenbeck)

add_executable(test_matern32 test_matern32.cpp cov_fnc.cpp)
gtest_discover_tests(test_matern32)
