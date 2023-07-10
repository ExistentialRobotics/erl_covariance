#include "cov_fnc.h"
#include "erl_common/eigen.hpp"
#include "erl_common/test_helper.hpp"
#include "erl_covariance/matern32.hpp"

constexpr double kMatern32Scale = 2.;
constexpr double kMatern32Alpha = 1.;
constexpr double kNoiseParam = 0.01;
using namespace erl;

struct Config {
    int d = 2;
    int n = 10;
    int m = 5;
    bool parallel = false;
    Eigen::MatrixXd mat_x_1;
    Eigen::MatrixXd mat_x_2;
    Eigen::VectorXd vec_sigma_x;
    Eigen::VectorXb grad_flag_1;
    std::vector<double> grad_flag_2;
    Eigen::VectorXd sigma_grad;
    Eigen::VectorXd sigma_y;

    std::shared_ptr<covariance::Matern32::Setting> matern_32_setting;

    Config()
        : matern_32_setting(std::make_shared<covariance::Matern32::Setting>()) {

        mat_x_1 = Eigen::MatrixXd::Random(d, n);
        mat_x_2 = Eigen::MatrixXd::Random(d, m);
        mat_x_1.array() *= 10;
        mat_x_2.array() *= 10;
        vec_sigma_x = Eigen::VectorXd::Random(n);

        matern_32_setting->type = covariance::Covariance::Type::kMatern32;
        matern_32_setting->alpha = kMatern32Alpha;
        matern_32_setting->scale = kMatern32Scale;

        Eigen::VectorXd random_flag = Eigen::VectorXd::Random(n);
        for (int i = 0; i < n; ++i) {
            if (random_flag(i) < double(0.5)) {
                grad_flag_1[i] = false;
                grad_flag_2.push_back(0.);
            } else {
                grad_flag_1[i] = true;
                grad_flag_2.push_back(1.);
            }
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
    auto matern_32 = covariance::Matern32::Create(kConfig.matern_32_setting);

    Eigen::MatrixXd ans, gt;
    common::ReportTime<std::chrono::milliseconds>("ans", 10, false, [&]() {
        ans = matern_32->ComputeKtrainWithGradient(kConfig.mat_x_1, kConfig.grad_flag_1, kConfig.vec_sigma_x, kConfig.sigma_y, kConfig.sigma_grad);
    });
    common::ReportTime<std::chrono::milliseconds>("gt", 10, false, [&]() {
        gt = Matern32SparseDeriv1(kConfig.mat_x_1, kConfig.grad_flag_2, kMatern32Scale, kConfig.vec_sigma_x, kNoiseParam, kConfig.sigma_grad);
    });
    common::GtestAssertSequenceEqual(ans, gt);
    common::GtestAssertSequenceEqual(Eigen::MatrixXd(ans.llt().matrixL()), Eigen::MatrixXd(gt.llt().matrixL()));
}

TEST(Matern32Test, ComputeKtestWithGradient) {
    const auto &kConfig = TestEnvironment::GetConfig();
    auto matern_32 = covariance::Matern32::Create(kConfig.matern_32_setting);

    Eigen::MatrixXd ans, gt;
    common::ReportTime<std::chrono::milliseconds>("ans", 10, false, [&]() {
        ans = matern_32->ComputeKtestWithGradient(kConfig.mat_x_1, kConfig.grad_flag_1, kConfig.mat_x_2);
    });
    common::ReportTime<std::chrono::milliseconds>("gt", 10, false, [&]() {
        gt = Matern32SparseDeriv1(kConfig.mat_x_1, kConfig.grad_flag_2, kConfig.mat_x_2, kMatern32Scale);
    });
    common::GtestAssertSequenceEqual(ans, gt);
}
