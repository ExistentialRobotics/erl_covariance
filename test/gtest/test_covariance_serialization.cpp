#include "erl_common/serialization.hpp"
#include "erl_common/test_helper.hpp"
#include "erl_covariance/matern32.hpp"

TEST(Covariance, Serialization) {
    GTEST_PREPARE_OUTPUT_DIR();

    using namespace erl::common;
    using namespace erl::covariance;

    const auto kernel_setting = std::make_shared<Matern32_3d::Setting>();
    kernel_setting->x_dim = 3;
    auto matern32 = std::make_shared<Matern32_3d>(kernel_setting);

    {
        EXPECT_TRUE(Serialization<Matern32_3d>::Write("matern32.bin", matern32));
        Matern32_3d matern32_read(std::make_shared<Matern32_3d::Setting>());
        EXPECT_TRUE(Serialization<Matern32_3d>::Read("matern32.bin", &matern32_read));
        EXPECT_TRUE(*matern32 == matern32_read);
    }

    Eigen::MatrixXd mat_x = LoadEigenMatrixFromTextFile<double>(gtest_src_dir / "x_train.txt");
    Eigen::VectorXl vec_grad_flags = Eigen::VectorXb::Random(mat_x.cols()).cast<long>();
    const long num_samples_with_gradient = vec_grad_flags.cast<long>().sum();

    auto [rows, cols] = matern32->GetMinimumKtrainSize(mat_x.cols(), num_samples_with_gradient, 3);
    Eigen::MatrixXd k_mat1(rows, cols);
    (void) matern32->ComputeKtrainWithGradient(mat_x, mat_x.cols(), vec_grad_flags, k_mat1);

    {
        EXPECT_TRUE(Serialization<Matern32_3d>::Write("matern32.bin", matern32));
        Matern32_3d matern32_read(std::make_shared<Matern32_3d::Setting>());
        EXPECT_TRUE(Serialization<Matern32_3d>::Read("matern32.bin", &matern32_read));
        EXPECT_TRUE(*matern32 == matern32_read);
    }
}
