#include "erl_common/test_helper.hpp"
#include "erl_covariance/covariance.hpp"

#include <iostream>
using namespace erl::covariance;

TEST(CovarianceSettingTest, FromYamlString) {
    Covariance<double>::Setting setting;
    setting.weights = Eigen::Vector2d{1., 2.};
    ASSERT_EQ(setting.x_dim, 2);
    ASSERT_EQ(setting.alpha, 1.);
    ASSERT_EQ(setting.scale, 1.);
    ASSERT_EQ(setting.scale_mix, 1.);
    ASSERT_EQ(setting.weights.size(), 2);
    {
        Eigen::Vector2d gt_weights{1., 2.};
        ASSERT_EIGEN_VECTOR_EQUAL("weights", setting.weights, gt_weights);
    }

    std::cout << setting << std::endl;

    ASSERT_TRUE(setting.FromYamlString(R"(
type: kMatern32
x_dim: 2
parallel: true
alpha: 2.0
scale: 3.0
scale_mix: 0.5
weights: [1.0, 2.0, 3.0, 4.0]
)"));

    std::cout << setting << std::endl;
    ASSERT_EQ(setting.x_dim, 2);
    ASSERT_EQ(setting.alpha, 2.);
    ASSERT_EQ(setting.scale, 3.);
    ASSERT_EQ(setting.scale_mix, 0.5);
    ASSERT_EQ(setting.weights.size(), 4);
    {
        Eigen::Vector4d gt_weights{1., 2., 3., 4.};
        ASSERT_EIGEN_VECTOR_EQUAL("weights", setting.weights, gt_weights);
    }
}
