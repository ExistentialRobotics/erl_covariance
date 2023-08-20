#include "../cov_fnc.h"
#include "../cov_fnc.cpp"
#include "erl_common/eigen.hpp"
#include "erl_common/test_helper.hpp"
#include "erl_covariance/matern32.hpp"

constexpr double kMatern32Scale = 2.;
constexpr double kMatern32Alpha = 1.;
constexpr double kNoiseParam = 0.01;
using namespace erl;

struct Config {
    int d = 2;
    int n = 256;
    int m = 512;
    Eigen::MatrixXd mat_x_1;
    Eigen::MatrixXd mat_x_2;
    Eigen::VectorXd vec_sigma_x;
    Eigen::VectorXb grad_flag_1;
    std::vector<double> grad_flag_2;
    Eigen::VectorXd sigma_grad;
    Eigen::VectorXd sigma_y;

    std::shared_ptr<covariance::Matern32<2>::Setting> matern_32_setting;

    Config()
        : matern_32_setting(std::make_shared<covariance::Matern32<2>::Setting>()) {

        mat_x_1 = Eigen::MatrixXd::Random(d, n);
        mat_x_2 = Eigen::MatrixXd::Random(d, m);
        mat_x_1.array() *= 10;
        mat_x_2.array() *= 10;
        vec_sigma_x = Eigen::VectorXd::Random(n);

        matern_32_setting->type = covariance::Covariance::Type::kMatern32;
        matern_32_setting->alpha = kMatern32Alpha;
        matern_32_setting->scale = kMatern32Scale;

        grad_flag_1 = Eigen::VectorXb::Random(n);
        grad_flag_2.reserve(n);
        for (bool &flag : grad_flag_1) {
            grad_flag_2.push_back(flag ? 1. : 0.);
        }

        sigma_grad = Eigen::VectorXd::Random(n);
        sigma_y = Eigen::VectorXd::Constant(n, kNoiseParam);
    }
};

struct TestEnvironment : public ::testing::Environment {
public:
    static Config
    GetConfig() {
        static Config config;
        return config;
    }

    void
    SetUp() override {
        GetConfig();
    }
};

TEST(Matern32Test, ComputeKtrainWithGradient) {
    const auto &kConfig = TestEnvironment::GetConfig();
    auto matern_32 = covariance::Matern32<2>::Create(kConfig.matern_32_setting);

    auto size = covariance::Matern32<2>::GetMinimumKtrainSize(kConfig.n, kConfig.grad_flag_1.cast<long>().sum(), kConfig.d);
    Eigen::MatrixXd ans(size.first, size.second), gt;
    common::ReportTime<std::chrono::microseconds>("gt", 10, false, [&]() {
        gt = Matern32SparseDeriv1(kConfig.mat_x_1, kConfig.grad_flag_2, kMatern32Scale, kConfig.vec_sigma_x, kNoiseParam, kConfig.sigma_grad);
    });
    common::ReportTime<std::chrono::microseconds>("ans", 10, false, [&]() {
        (void) matern_32->ComputeKtrainWithGradient(ans, kConfig.mat_x_1, kConfig.grad_flag_1, kConfig.vec_sigma_x, kConfig.sigma_y, kConfig.sigma_grad);
    });

    EXPECT_EQ(ans.rows(), gt.rows());
    EXPECT_EQ(ans.cols(), gt.cols());
    EXPECT_TRUE(ans.isApprox(gt, 1e-10));
}

TEST(Matern32Test, ComputeKtestWithGradient) {
    const auto &kConfig = TestEnvironment::GetConfig();
    auto matern_32 = covariance::Matern32<2>::Create(kConfig.matern_32_setting);

    auto size = covariance::Matern32<2>::GetMinimumKtestSize(kConfig.n, kConfig.grad_flag_1.cast<long>().sum(), kConfig.d, kConfig.m);
    Eigen::MatrixXd ans(size.first, size.second), gt;
    common::ReportTime<std::chrono::microseconds>("gt", 10, false, [&]() {
        gt = Matern32SparseDeriv1(kConfig.mat_x_1, kConfig.grad_flag_2, kConfig.mat_x_2, kMatern32Scale);
    });
    common::ReportTime<std::chrono::microseconds>("ans", 10, false, [&]() {
        (void) matern_32->ComputeKtestWithGradient(ans, kConfig.mat_x_1, kConfig.grad_flag_1, kConfig.mat_x_2);
    });
    EXPECT_EQ(ans.rows(), gt.rows());
    EXPECT_EQ(ans.cols(), gt.cols());
    EXPECT_TRUE(ans.isApprox(gt, 1e-10));
}
