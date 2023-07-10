#include <gtest/gtest.h>

#include "cov_fnc.h"
#include "erl_common/eigen.hpp"
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
    std::shared_ptr<covariance::OrnsteinUhlenbeck::Setting> ou_setting;

    Config()
        : ou_setting(std::make_shared<covariance::OrnsteinUhlenbeck::Setting>()) {
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
TEST(OrnsteinUhlenbeckTest, ComputeKtrain) {
    const auto &kConfig = TestEnvironment::GetConfig();
    auto ornstein_uhlenbeck = covariance::OrnsteinUhlenbeck::Create(kConfig.ou_setting);

    Eigen::MatrixXd ans, gt;
    common::ReportTime<std::chrono::milliseconds>("ans", 10, false, [&]() {
        ans = ornstein_uhlenbeck->ComputeKtrain(kConfig.mat_x_1, Eigen::VectorXd::Constant(kConfig.n, kConfig.sigma_output_scalar));
    });
    common::ReportTime<std::chrono::milliseconds>("gt", 10, false, [&]() { gt = OrnsteinUhlenbeck(kConfig.mat_x_1, kOrnsteinUhlenbeckScale, kConfig.sigma_output_scalar); });
    common::GtestAssertSequenceEqual(ans, gt);

    common::ReportTime<std::chrono::milliseconds>("ans", 10, false, [&]() { ans = ornstein_uhlenbeck->ComputeKtrain(kConfig.mat_x_1, kConfig.sigma_x); });
    common::ReportTime<std::chrono::milliseconds>("gt", 10, false, [&]() { gt = OrnsteinUhlenbeck(kConfig.mat_x_1, kOrnsteinUhlenbeckScale, kConfig.sigma_x); });
    common::GtestAssertSequenceEqual(ans, gt);

    common::ReportTime<std::chrono::milliseconds>("ans", 10, false, [&]() { ans = ornstein_uhlenbeck->ComputeKtest(kConfig.mat_x_1, kConfig.mat_x_2); });
    common::ReportTime<std::chrono::milliseconds>("gt", 10, false, [&]() { gt = OrnsteinUhlenbeck(kConfig.mat_x_1, kConfig.mat_x_2, kOrnsteinUhlenbeckScale); });
    common::GtestAssertSequenceEqual(ans, gt);
}

TEST(OrnsteinUhlenbeckTest, ComputeKtest) {
    const auto &kConfig = TestEnvironment::GetConfig();
    auto ornstein_uhlenbeck = covariance::OrnsteinUhlenbeck::Create(kConfig.ou_setting);

    Eigen::MatrixXd ans, gt;
    common::ReportTime<std::chrono::milliseconds>("ans", 10, false, [&]() { ans = ornstein_uhlenbeck->ComputeKtest(kConfig.mat_x_1, kConfig.mat_x_2); });
    common::ReportTime<std::chrono::milliseconds>("gt", 10, false, [&]() { gt = OrnsteinUhlenbeck(kConfig.mat_x_1, kConfig.mat_x_2, kOrnsteinUhlenbeckScale); });
    common::GtestAssertSequenceEqual(ans, gt);
}
