#include "erl_common/test_helper.hpp"
#include "erl_covariance/covariance.hpp"

#include <iostream>
using namespace erl::covariance;

TEST(CovarianceSettingTest, FromYamlString) {
    Covariance<double>::Setting setting;
    ASSERT_EQ(setting.x_dim, 2);
    ASSERT_EQ(setting.scale, 1.0);
    ASSERT_EQ(setting.scale_mix, 1.0);

    std::cout << setting << std::endl;

    ASSERT_TRUE(setting.FromYamlString(R"(
type: kMatern32
x_dim: 2
scale: 3.0
scale_mix: 0.5
)"));

    std::cout << setting << std::endl;
    ASSERT_EQ(setting.x_dim, 2);
    ASSERT_EQ(setting.scale, 3.0);
    ASSERT_EQ(setting.scale_mix, 0.5);
}
