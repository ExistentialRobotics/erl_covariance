#include "erl_common/serialization.hpp"
#include "erl_common/test_helper.hpp"
#include "erl_covariance/reduced_rank_matern32.hpp"

TEST(ReducedRank, Serialization) {
    GTEST_PREPARE_OUTPUT_DIR();

    using namespace erl::common;
    using namespace erl::covariance;

    const auto kernel_setting = std::make_shared<ReducedRankMatern32_3d::Setting>();
    kernel_setting->x_dim = 3;
    kernel_setting->num_basis = Eigen::Vector3l{3, 3, 3};
    kernel_setting->boundaries = Eigen::Vector3d::Ones();
    auto matern32 = std::make_shared<ReducedRankMatern32_3d>(kernel_setting);

    {
        EXPECT_TRUE(
            Serialization<ReducedRankMatern32_3d>::Write("reduced_rank_matern32.bin", matern32));
        ReducedRankMatern32_3d matern32_read(std::make_shared<ReducedRankMatern32_3d::Setting>());
        EXPECT_TRUE(
            Serialization<ReducedRankMatern32_3d>::Read(
                "reduced_rank_matern32.bin",
                &matern32_read));
        EXPECT_TRUE(*matern32 == matern32_read);
    }

    matern32->BuildSpectralDensities();
    {
        EXPECT_TRUE(
            Serialization<ReducedRankMatern32_3d>::Write("reduced_rank_matern32.bin", matern32));
        ReducedRankMatern32_3d matern32_read(std::make_shared<ReducedRankMatern32_3d::Setting>());
        EXPECT_TRUE(
            Serialization<ReducedRankMatern32_3d>::Read(
                "reduced_rank_matern32.bin",
                &matern32_read));
        EXPECT_TRUE(*matern32 == matern32_read);
    }

    Eigen::MatrixXd mat_x = LoadEigenMatrixFromTextFile<double>(gtest_src_dir / "x_train.txt");
    Eigen::VectorXl vec_grad_flags = Eigen::VectorXb::Random(mat_x.cols()).cast<long>();
    const long num_samples_with_gradient = vec_grad_flags.cast<long>().sum();

    auto [rows, cols] = matern32->GetMinimumKtrainSize(mat_x.cols(), num_samples_with_gradient, 3);
    Eigen::MatrixXd k_mat1(rows, cols);
    Eigen::MatrixXd mat_alpha =
        Eigen::MatrixXd::Random(mat_x.cols() + 3 * num_samples_with_gradient, 1);
    matern32->ComputeKtrainWithGradient(mat_x, mat_x.cols(), vec_grad_flags, k_mat1, mat_alpha);

    {
        EXPECT_TRUE(
            Serialization<ReducedRankMatern32_3d>::Write("reduced_rank_matern32.bin", matern32));
        ReducedRankMatern32_3d matern32_read(std::make_shared<ReducedRankMatern32_3d::Setting>());
        EXPECT_TRUE(
            Serialization<ReducedRankMatern32_3d>::Read(
                "reduced_rank_matern32.bin",
                &matern32_read));
        EXPECT_TRUE(*matern32 == matern32_read);
    }
}
