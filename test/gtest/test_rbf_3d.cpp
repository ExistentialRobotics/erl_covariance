#include "erl_common/test_helper.hpp"
#include "erl_covariance/radial_bias_function.hpp"

TEST(Rbf, ExpBias) {
    using namespace erl::common;
    using namespace erl::covariance;

    GTEST_PREPARE_OUTPUT_DIR();

    const auto kernel_setting = std::make_shared<RadialBiasFunction3d::Setting>();
    kernel_setting->x_dim = 3;
    auto rbf = std::make_shared<RadialBiasFunction3d>(kernel_setting);

    const Eigen::Matrix3Xd mat_x =
        LoadEigenMatrixFromTextFile<double>(gtest_src_dir / "x_train.txt");
    const long n_train = mat_x.cols();
    Eigen::VectorXl vec_grad_flags = Eigen::VectorXb::Random(n_train).cast<long>();
    const long num_samples_with_gradient = vec_grad_flags.cast<long>().sum();

    constexpr double exp_bias = 1.0;
    long rows = 0;
    long cols = 0;
    std::tie(rows, cols) = rbf->GetMinimumKtrainSize(n_train, num_samples_with_gradient, 3);
    Eigen::MatrixXd k_mat1(rows, cols);
    Eigen::MatrixXd alpha;
    (void) rbf->ComputeKtrainWithGradient(mat_x, n_train, vec_grad_flags, k_mat1, alpha);
    Eigen::MatrixXd k_mat2(rows, cols);
    (void) rbf->ComputeKtrainWithGradient(mat_x, n_train, exp_bias, vec_grad_flags, k_mat2, alpha);
    const Eigen::MatrixXd diff1 = (k_mat1.array() * std::exp(exp_bias) - k_mat2.array()).abs();
    ERL_INFO("diff1 min: {}, max: {}", diff1.minCoeff(), diff1.maxCoeff());
    EXPECT_LT(diff1.maxCoeff(), 1.e-6);

    std::tie(rows, cols) = rbf->GetMinimumKtrainSize(n_train, 0, 3);
    k_mat1.resize(rows, cols);
    (void) rbf->ComputeKtrain(mat_x, n_train, k_mat1, alpha);
    k_mat2.resize(rows, cols);
    (void) rbf->ComputeKtrain(mat_x, n_train, exp_bias, k_mat2, alpha);
    const Eigen::MatrixXd diff2 = (k_mat1.array() * std::exp(exp_bias) - k_mat2.array()).abs();
    ERL_INFO("diff2 min: {}, max: {}", diff2.minCoeff(), diff2.maxCoeff());
    EXPECT_LT(diff2.maxCoeff(), 1.e-6);

    constexpr long n_test = 4;
    // const Eigen::Matrix3Xd mat_x_test = mat_x.leftCols(n_test);
    const Eigen::Matrix3Xd mat_x_test = Eigen::Matrix3Xd::Random(3, n_test);
    std::tie(rows, cols) = rbf->GetMinimumKtestSize(n_train, 0, 3, n_test, false);
    k_mat1.resize(rows, cols);
    (void) rbf->ComputeKtest(mat_x, n_train, mat_x_test, n_test, k_mat1);
    k_mat2.resize(rows, cols);
    (void) rbf->ComputeKtest(mat_x, n_train, mat_x_test, n_test, exp_bias, k_mat2);
    const Eigen::MatrixXd diff3 = (k_mat1.array() * std::exp(exp_bias) - k_mat2.array()).abs();
    ERL_INFO("diff3 min: {}, max: {}", diff3.minCoeff(), diff3.maxCoeff());
    EXPECT_LT(diff3.maxCoeff(), 1.e-6);

    std::tie(rows, cols) = rbf->GetMinimumKtestSize(n_train, 0, 3, n_test, true);
    k_mat1.resize(rows, cols);
    (void) rbf->ComputeKtestWithGradient(
        mat_x,
        n_train,
        Eigen::VectorXl::Zero(n_train),
        mat_x_test,
        n_test,
        true,
        k_mat1);
    k_mat2.resize(rows, cols);
    (void) rbf->ComputeKtestWithGradient(
        mat_x,
        n_train,
        Eigen::VectorXl::Zero(n_train),
        mat_x_test,
        n_test,
        true,
        exp_bias,
        k_mat2);
    const Eigen::MatrixXd diff4 = (k_mat1.array() * std::exp(exp_bias) - k_mat2.array()).abs();
    ERL_INFO("diff4 min: {}, max: {}", diff4.minCoeff(), diff4.maxCoeff());
    EXPECT_LT(diff4.maxCoeff(), 1.e-6);

    std::tie(rows, cols) =
        rbf->GetMinimumKtestSize(n_train, num_samples_with_gradient, 3, n_test, false);
    k_mat1.resize(rows, cols);
    (void) rbf->ComputeKtestWithGradient(
        mat_x,
        n_train,
        vec_grad_flags,
        mat_x_test,
        n_test,
        false,
        k_mat1);
    k_mat2.resize(rows, cols);
    (void) rbf->ComputeKtestWithGradient(
        mat_x,
        n_train,
        vec_grad_flags,
        mat_x_test,
        n_test,
        false,
        exp_bias,
        k_mat2);
    const Eigen::MatrixXd diff5 = (k_mat1.array() * std::exp(exp_bias) - k_mat2.array()).abs();
    ERL_INFO("diff5 min: {}, max: {}", diff5.minCoeff(), diff5.maxCoeff());
    EXPECT_LT(diff5.maxCoeff(), 1.e-6);

    std::tie(rows, cols) =
        rbf->GetMinimumKtestSize(n_train, num_samples_with_gradient, 3, n_test, true);
    k_mat1.resize(rows, cols);
    (void) rbf->ComputeKtestWithGradient(
        mat_x,
        n_train,
        vec_grad_flags,
        mat_x_test,
        n_test,
        true,
        k_mat1);
    k_mat2.resize(rows, cols);
    (void) rbf->ComputeKtestWithGradient(
        mat_x,
        n_train,
        vec_grad_flags,
        mat_x_test,
        n_test,
        true,
        exp_bias,
        k_mat2);
    const Eigen::MatrixXd diff6 = (k_mat1.array() * std::exp(exp_bias) - k_mat2.array()).abs();
    ERL_INFO("diff6 min: {}, max: {}", diff6.minCoeff(), diff6.maxCoeff());
    EXPECT_LT(diff6.maxCoeff(), 1.e-6);
}
