#include "erl_common/block_timer.hpp"
#include "erl_common/eigen.hpp"
#include "erl_common/test_helper.hpp"

#include <Eigen/Geometry>

TEST(LLT, Adj) {
    GTEST_PREPARE_OUTPUT_DIR();

    Eigen::MatrixXd rand_mat = Eigen::MatrixXd::Random(10, 10);
    std::cout << rand_mat << std::endl << std::endl;
    rand_mat.triangularView<Eigen::Upper>() = rand_mat.transpose().triangularView<Eigen::Upper>();
    std::cout << rand_mat << std::endl;

    Eigen::MatrixXd mat =
        erl::common::LoadEigenMatrixFromTextFile<double>(gtest_src_dir / "matern32_ktrain.txt");
    Eigen::VectorXd alpha = Eigen::VectorXd::Ones(mat.cols());

    constexpr long n = 100;

    std::vector<Eigen::MatrixXd> mats_llt(n);
    std::vector<Eigen::VectorXd> vecs_llt(n);
    {
        const erl::common::BlockTimer<std::chrono::milliseconds> timer("LLT");
        (void) timer;
        for (long i = 0; i < n; ++i) {
            Eigen::MatrixXd& mat_l = mats_llt[i];
            mat_l = mat.llt().matrixL();
            // vecs_llt[i] = llt.solve(alpha);
            // vecs_llt[i] = mat.lu().solve(alpha);
            Eigen::VectorXd vec = alpha;
            mat_l.triangularView<Eigen::Lower>().solveInPlace(vec);
            mat_l.transpose().triangularView<Eigen::Upper>().solveInPlace(vec);
            vecs_llt[i] = std::move(vec);
        }
    }

    std::vector<Eigen::MatrixXd> mats_adj(n);
    std::vector<Eigen::VectorXd> vecs_adj(n);
    {
        const erl::common::BlockTimer<std::chrono::milliseconds> timer("Adj");
        (void) timer;
        for (long i = 0; i < n; ++i) {
            Eigen::MatrixXd& mat_l = mats_llt[i];
            mats_adj[i] = mat.selfadjointView<Eigen::Lower>().llt().matrixL();
            // vecs_adj[i] = llt.solve(alpha);
            Eigen::VectorXd vec = alpha;
            mat_l.triangularView<Eigen::Lower>().solveInPlace(vec);
            mat_l.transpose().triangularView<Eigen::Upper>().solveInPlace(vec);
            vecs_llt[i] = std::move(vec);
        }
    }

    // Check
    for (long i = 0; i < n; ++i) {
        EXPECT_TRUE(mats_llt[i].isApprox(mats_adj[i], 1e-10));
        EXPECT_TRUE(vecs_llt[i].isApprox(vecs_adj[i], 1e-10));
    }
}
