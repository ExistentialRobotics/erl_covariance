#include "erl_common/test_helper.hpp"
#include "erl_covariance/reduced_rank_matern32.hpp"

TEST(ReducedRankMatern32_2d, Copy) {
    GTEST_PREPARE_OUTPUT_DIR();
    using namespace erl::covariance;
    auto setting = std::make_shared<ReducedRankMatern32_2d::Setting>();
    ASSERT_TRUE(setting->FromYamlFile(gtest_src_dir / "reduced_rank_matern32.yaml"));
    std::cout << *setting << std::endl;

    std::shared_ptr<ReducedRankMatern32_2d> cov = std::make_shared<ReducedRankMatern32_2d>(setting);
    cov->GetKtrainBuffer() = Eigen::MatrixXd::Random(10, 10);
    cov->BuildSpectralDensities();
    std::shared_ptr<ReducedRankMatern32_2d::Factory::BaseType> cov_base = cov;
    std::shared_ptr<ReducedRankMatern32_2d::Factory::BaseType> cov_new =
        std::make_shared<ReducedRankMatern32_2d>(setting);
    *std::dynamic_pointer_cast<ReducedRankMatern32_2d>(cov_new) =
        *std::dynamic_pointer_cast<ReducedRankMatern32_2d>(cov_base);  // PASSED
    // *cov_new = *cov_base;  // FAILED
    std::shared_ptr<ReducedRankMatern32_2d> cov_new_derived =
        std::dynamic_pointer_cast<ReducedRankMatern32_2d>(cov_new);
    EXPECT_EQ(cov_new_derived->GetKtrainBuffer(), cov->GetKtrainBuffer());
}

TEST(ReducedRankMatern32_2d, EigenFunctions) {
    GTEST_PREPARE_OUTPUT_DIR();
    using namespace erl::common;
    using namespace erl::covariance;
    auto setting = std::make_shared<ReducedRankMatern32_2d::Setting>();
    ASSERT_TRUE(setting->FromYamlFile(gtest_src_dir / "reduced_rank_matern32.yaml"));

    Eigen::MatrixXd xy = LoadEigenMatrixFromBinaryFile(gtest_src_dir / "train_xy.dat");
    Eigen::MatrixXd phi_gt =
        LoadEigenMatrixFromBinaryFile(gtest_src_dir / "train_phi.dat");  // eigen functions
    Eigen::VectorXd spect_density_gt =
        LoadEigenMatrixFromBinaryFile(gtest_src_dir / "spect_density.dat").diagonal();

    std::shared_ptr<ReducedRankMatern32_2d> cov = std::make_shared<ReducedRankMatern32_2d>(setting);
    cov->BuildSpectralDensities();
    Eigen::VectorXd spect_density = setting->GetSpectralDensities();
    EXPECT_TRUE(spect_density.isApprox(spect_density_gt, 1e-7));

    Eigen::MatrixXd phi = cov->ComputeEigenFunctions(xy, xy.rows(), xy.cols());

    EXPECT_EQ(phi_gt.rows(), phi.rows());
    EXPECT_EQ(phi_gt.cols(), phi.cols());

    for (long i = 0; i < phi.cols(); ++i) {
        Eigen::VectorXd diff = phi.col(i) - phi_gt.col(i);
        for (long j = 0; j < diff.size(); ++j) {
            ASSERT_TRUE(std::abs(diff[j]) < 1e-10)
                << "i = " << i << ", j = " << j << ", diff = " << diff[j];
        }
    }
}

TEST(ReducedRankMatern32_2f, EigenFunctions) {
    GTEST_PREPARE_OUTPUT_DIR();
    using namespace erl::common;
    using namespace erl::covariance;
    auto setting = std::make_shared<ReducedRankMatern32_2f::Setting>();
    ASSERT_TRUE(setting->FromYamlFile(gtest_src_dir / "reduced_rank_matern32.yaml"));

    Eigen::MatrixXf xy =
        LoadEigenMatrixFromBinaryFile(gtest_src_dir / "train_xy.dat").cast<float>();
    Eigen::MatrixXf phi_gt = LoadEigenMatrixFromBinaryFile(gtest_src_dir / "train_phi.dat")
                                 .cast<float>();  // eigen functions
    Eigen::VectorXf spect_density_gt =
        LoadEigenMatrixFromBinaryFile(gtest_src_dir / "spect_density.dat").diagonal().cast<float>();

    std::shared_ptr<ReducedRankMatern32_2f> cov = std::make_shared<ReducedRankMatern32_2f>(setting);
    cov->BuildSpectralDensities();
    Eigen::VectorXf spect_density = setting->GetSpectralDensities();
    constexpr float kTolerance = 1e-6f;
    EXPECT_TRUE(spect_density.isApprox(spect_density_gt, kTolerance));

    Eigen::MatrixXf phi = cov->ComputeEigenFunctions(xy, xy.rows(), xy.cols());

    EXPECT_EQ(phi_gt.rows(), phi.rows());
    EXPECT_EQ(phi_gt.cols(), phi.cols());

    for (long i = 0; i < phi.cols(); ++i) {
        Eigen::VectorXf diff = phi.col(i) - phi_gt.col(i);
        for (long j = 0; j < diff.size(); ++j) {
            ASSERT_TRUE(std::abs(diff[j]) < kTolerance)
                << "i = " << i << ", j = " << j << ", diff = " << diff[j];
        }
    }
}
