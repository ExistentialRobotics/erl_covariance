#include <gtest/gtest.h>

#include "../cov_fnc.cpp"
#include "erl_common/test_helper.hpp"
#include "erl_covariance/ornstein_uhlenbeck.hpp"

constexpr double kOrnsteinUhlenbeckScale = 2.;
constexpr double kOrnsteinUhlenbeckAlpha = 1.;
using namespace erl;

struct Config {
    int d = 2;
    int n = 10;
    int m = 5;
    Eigen::MatrixXd mat_x_1;
    Eigen::MatrixXd mat_x_2;
    double sigma_output_scalar = 2.;
    Eigen::VectorXd sigma_x;
    std::shared_ptr<covariance::OrnsteinUhlenbeck<2>::Setting> ou_setting;

    Config()
        : ou_setting(std::make_shared<covariance::OrnsteinUhlenbeck<2>::Setting>()) {
        mat_x_1 = Eigen::MatrixXd::Random(d, n);
        mat_x_2 = Eigen::MatrixXd::Random(d, m);
        mat_x_1.array() *= 10;
        mat_x_2.array() *= 10;

        sigma_x = Eigen::VectorXd::Random(n);

        ou_setting->type = covariance::Covariance::Type::kOrnsteinUhlenbeck;
        ou_setting->alpha = kOrnsteinUhlenbeckAlpha;
        ou_setting->scale = kOrnsteinUhlenbeckScale;
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

// verify that my implementation is consistent with GPisMap's
TEST(OrnsteinUhlenbeck, ComputeKtrain) {
    const auto &kConfig = TestEnvironment::GetConfig();
    auto ornstein_uhlenbeck = covariance::OrnsteinUhlenbeck<2>::Create(kConfig.ou_setting);

    std::cout << "==============" << std::endl;
    std::pair<long, long> size = covariance::Covariance::GetMinimumKtrainSize(kConfig.n, 0, 2);
    Eigen::MatrixXd ans(size.first, size.second), gt;
    common::ReportTime<std::chrono::microseconds>("ans", 10, false, [&]() {
        (void) ornstein_uhlenbeck->ComputeKtrain(ans, kConfig.mat_x_1, Eigen::VectorXd::Constant(kConfig.n, kConfig.sigma_output_scalar));
    });
    common::ReportTime<std::chrono::microseconds>("gt", 10, false, [&]() {
        gt = OrnsteinUhlenbeck(kConfig.mat_x_1, kOrnsteinUhlenbeckScale, kConfig.sigma_output_scalar);
    });
    ASSERT_EIGEN_MATRIX_EQUAL("ComputeKtrain1", ans, gt);

    std::cout << "==============" << std::endl;
    common::ReportTime<std::chrono::microseconds>("ans", 10, false, [&]() { (void) ornstein_uhlenbeck->ComputeKtrain(ans, kConfig.mat_x_1, kConfig.sigma_x); });
    common::ReportTime<std::chrono::microseconds>("gt", 10, false, [&]() {
        gt = OrnsteinUhlenbeck(kConfig.mat_x_1, kOrnsteinUhlenbeckScale, kConfig.sigma_x);
    });
    ASSERT_EIGEN_MATRIX_EQUAL("ComputeKtrain2", ans, gt);
}

TEST(OrnsteinUhlenbeck, ComputeKtest) {
    const auto &kConfig = TestEnvironment::GetConfig();
    auto ornstein_uhlenbeck = covariance::OrnsteinUhlenbeck<2>::Create(kConfig.ou_setting);

    std::cout << "==============" << std::endl;
    Eigen::MatrixXd ans, gt;
    std::pair<long, long> size = covariance::Covariance::GetMinimumKtestSize(kConfig.n, 0, 0, kConfig.m);
    ans.resize(size.first, size.second);
    common::ReportTime<std::chrono::microseconds>("ans", 10, false, [&]() { (void) ornstein_uhlenbeck->ComputeKtest(ans, kConfig.mat_x_1, kConfig.mat_x_2); });
    common::ReportTime<std::chrono::microseconds>("gt", 10, false, [&]() {
        gt = OrnsteinUhlenbeck(kConfig.mat_x_1, kConfig.mat_x_2, kOrnsteinUhlenbeckScale);
    });
    ASSERT_EIGEN_MATRIX_EQUAL("ComputeKtest", ans, gt);
}
