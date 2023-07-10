#include <gtest/gtest.h>

#include <iostream>

#include "erl_common/test_helper.hpp"
#include "erl_covariance/covariance.hpp"
using namespace erl::covariance;

TEST(CovarianceSettingTest, FromYamlString) {
    Covariance::Setting setting;
    setting.weights = Eigen::Vector2d{1., 2.};
    ASSERT_EQ(setting.type, Covariance::Type::kUnknown);
    ASSERT_EQ(setting.alpha, 1.);
    ASSERT_EQ(setting.scale, 1.);
    ASSERT_EQ(setting.scale_mix, 1.);
    ASSERT_EQ(setting.weights.size(), 2);
    erl::common::GtestAssertSequenceEqual(setting.weights, Eigen::Vector2d{1., 2.});

    std::cout << setting.AsYamlString() << std::endl;

    setting.FromYamlString(R"(
type: kMatern32
alpha: 2.0
scale: 3.0
parallel: true
scale_mix: 0.5
weights: [1.0, 2.0, 3.0, 4.0]
)");

    std::cout << setting.AsYamlString() << std::endl;
    ASSERT_EQ(setting.type, Covariance::Type::kMatern32);
    ASSERT_EQ(setting.alpha, 2.);
    ASSERT_EQ(setting.scale, 3.);
    ASSERT_EQ(setting.scale_mix, 0.5);
    ASSERT_EQ(setting.weights.size(), 4);
    erl::common::GtestAssertSequenceEqual(setting.weights, Eigen::Vector4d{1., 2., 3., 4.});
}
