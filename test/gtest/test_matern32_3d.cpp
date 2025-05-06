#include "erl_common/block_timer.hpp"
#include "erl_common/test_helper.hpp"
#include "erl_covariance/matern32.hpp"

static double
InlineMatern32(const double a, const double r, const double exp_term) {
    // return (alpha + a1 * r) * std::exp(-a2 * r);
    return (1.0 + a * r) * exp_term;
}

// cov(f1, df2/dx2) = d k(x1, x2) / dx2
// note: dx = x1 - x2
static double
InlineMatern32X1BetweenGradx2(const double dx, const double b_exp_term) {
    // return b * dx * std::exp(-a * r);
    // exp_term = std::exp(-a * r);
    return dx * b_exp_term;
}

// d^2 k(x1, x2) / dx_1 dx_2
static double
InlineMatern32Gradx1BetweenGradx2(
    const double a,
    const double b,
    const double delta,
    const double dx_1,
    const double dx_2,
    const double r,
    const double b_exp_term) {
    if (std::abs(dx_1 * dx_2) < 1.e-6 && std::abs(r) < 1.e-6) { return b; }
    // return b * (delta - a * dx_1 * dx_2 / r) * std::exp(-a * r);
    // b_exp_term = b * std::exp(-a2 * r);
    return (delta - a * dx_1 * dx_2 / r) * b_exp_term;
}

Eigen::MatrixXd
matern32_new(
    const std::shared_ptr<erl::covariance::Matern32_3d::Setting> &setting,
    const Eigen::MatrixXd &mat_x,
    const Eigen::VectorXl &vec_grad_flags) {

    constexpr long dim = 3;
    const long n = mat_x.cols();
    std::vector<long> grad_indices;
    std::vector<long> has_grad_indices;
    grad_indices.reserve(vec_grad_flags.size());
    has_grad_indices.reserve(vec_grad_flags.size());
    long n_grad = 0;
    for (long i = 0; i < n; ++i) {
        if (vec_grad_flags[i]) {
            grad_indices.push_back(n + n_grad++);
            has_grad_indices.push_back(i);
        } else {
            grad_indices.push_back(-1);
        }
    }

    const long n_rows = n + n_grad * dim;
    const long n_cols = n_rows;

    const double a1 = std::sqrt(3.) / setting->scale;
    const double b = a1 * a1;

#if EIGEN_VERSION_AT_LEAST(3, 4, 0)
    Eigen::MatrixXd mat_x_with_grad = mat_x(Eigen::all, has_grad_indices).transpose();
#else
    Eigen::MatrixXd mat_x_with_grad(has_grad_indices.size(), dim);
    for (long i = 0; i < static_cast<long>(has_grad_indices.size()); ++i) {
        mat_x_with_grad.row(i) = mat_x.col(has_grad_indices[i]).transpose();
    }
#endif

    Eigen::MatrixXd k_mat(n_rows, n_cols);
    auto k_mat_diag = k_mat.diagonal();
    k_mat_diag.head(n).array() = 1.0;
    k_mat_diag.tail(n_grad * dim).array() = b;

    for (long j = 0; j < n; ++j) {
        auto k_mat_col_j = k_mat.col(j);
        k_mat_col_j[j] = 1.0;

        Eigen::Matrix3Xd diffs(dim, n);
        Eigen::VectorXd diff_norms(n);
        Eigen::VectorXd exp_terms(n);

        if (const long m = n - j - 1; m > 0) {
            auto diffs_right = diffs.rightCols(m);
            auto mat_x_right_ptr = mat_x.rightCols(m).data();
            auto diffs_right_ptr = diffs_right.data();
            auto x_col_j = mat_x.col(j).data();
            auto diff_norms_right = diff_norms.tail(m).data();
            auto exp_terms_right = exp_terms.tail(m).data();
            auto k_mat_col_j_right = k_mat_col_j.segment(j + 1, m).data();

#pragma omp simd
            for (long i = 0; i < m; ++i) {
                double &x = diffs_right_ptr[0];
                double &y = diffs_right_ptr[1];
                double &z = diffs_right_ptr[2];
                x = mat_x_right_ptr[0] - x_col_j[0];
                y = mat_x_right_ptr[1] - x_col_j[1];
                z = mat_x_right_ptr[2] - x_col_j[2];
                diff_norms_right[i] = std::sqrt(x * x + y * y + z * z);
                exp_terms_right[i] = std::exp(-a1 * diff_norms_right[i]);
                k_mat_col_j_right[i] = exp_terms_right[i] * (1.0 + a1 * diff_norms_right[i]);
                mat_x_right_ptr += dim;
                diffs_right_ptr += dim;
            }
        }

        if (n_grad == 0) { continue; }

        Eigen::MatrixX3d diffs_grad(n_grad, dim);
        Eigen::VectorXd diff_norms_grad(n_grad);
        Eigen::VectorXd b_exp_terms(n_grad);

        // lower triangular part of cov(df_i/dx_k, f_j)
        for (long i = 0; i <= j; ++i) {
            if (!vec_grad_flags[i]) { continue; }
            auto diff_ij = diffs.col(i);
            diff_ij << mat_x.col(i) - mat_x.col(j);
            double &norm = diff_norms[i];
            norm = diff_ij.norm();
            double &exp_term = exp_terms[i];
            exp_term = std::exp(-a1 * norm);
        }
#if EIGEN_VERSION_AT_LEAST(3, 4, 0)
        diff_norms_grad = diff_norms(has_grad_indices);
        b_exp_terms = b * exp_terms(has_grad_indices);
#else
        for (long i = 0; i < static_cast<long>(has_grad_indices.size()); ++i) {
            diff_norms_grad[i] = diff_norms[has_grad_indices[i]];
            b_exp_terms[i] = b * exp_terms[has_grad_indices[i]];
        }
#endif
        for (long k = 0, ki = n; k < dim; ++k, ki += n_grad) {
            double *k_mat_col_j_ptr = k_mat_col_j.segment(ki, n_grad).data();
            const double xj_k = mat_x(k, j);
            double *mat_x_with_grad_k_ptr = mat_x_with_grad.col(k).data();
            double *diffs_k_ptr = diffs_grad.col(k).data();
            double *b_exp_terms_ptr = b_exp_terms.data();
#pragma omp simd
            for (long i = 0; i < n_grad; ++i) {
                diffs_k_ptr[i] = mat_x_with_grad_k_ptr[i] - xj_k;
                k_mat_col_j_ptr[i] = -diffs_k_ptr[i] * b_exp_terms_ptr[i];
            }
        }

        // lower triangular part of cov(df_i/dx_k, df_j/dx_l)
        if (!vec_grad_flags[j]) { continue; }
        for (long l = 0, lj = grad_indices[j], rows_grad = n + n_grad; l < dim;
             ++l, lj += n_grad, rows_grad += n_grad) {
            auto k_mat_col_lj = k_mat.col(lj);

            if (const long m_grad = rows_grad - lj - 1; m_grad > 0) {
                auto dx = diffs_grad.col(l).tail(m_grad);
                auto r = diff_norms_grad.tail(m_grad);
                auto b_exp_term = b_exp_terms.tail(m_grad);
                k_mat_col_lj.segment(lj + 1, m_grad) =
                    (1.0 - a1 * dx.array().square() / r.array()) * b_exp_term.array();
            }

            for (long k = l + 1, start = rows_grad; k < dim; ++k, start += n_grad) {
                double *cov_ki_lj_ptr = k_mat_col_lj.segment(start, n_grad).data();
                double *diffs_grad_k_ptr = diffs_grad.col(k).data();
                double *diffs_grad_l_ptr = diffs_grad.col(l).data();
                double *diff_norms_grad_ptr = diff_norms_grad.data();
                double *b_exp_terms_ptr = b_exp_terms.data();
                for (long i = 0; i < n_grad; ++i) {
                    if (has_grad_indices[i] == j) {
                        cov_ki_lj_ptr[i] = 0;
                        continue;
                    }
                    const double dx_kl = diffs_grad_k_ptr[i] * diffs_grad_l_ptr[i];
                    if (std::abs(dx_kl) < 1.e-6 && std::abs(diff_norms_grad_ptr[i]) < 1.e-6) {
                        cov_ki_lj_ptr[i] = b;
                        continue;
                    }
                    cov_ki_lj_ptr[i] = -a1 * dx_kl / diff_norms_grad_ptr[i] * b_exp_terms_ptr[i];
                }
            }
        }
    }

    k_mat.triangularView<Eigen::Upper>() = k_mat.transpose().triangularView<Eigen::Upper>();
    return k_mat;
}

Eigen::MatrixXd
matern32_new2(
    const std::shared_ptr<erl::covariance::Matern32_3d::Setting> &setting,
    const Eigen::MatrixXd &mat_x,
    const Eigen::VectorXl &vec_grad_flags) {
    const long dim = mat_x.rows();

    const long n = mat_x.cols();
    std::vector<long> grad_indices;
    grad_indices.reserve(vec_grad_flags.size());
    long n_grad = 0;
    const long *grad_flags_ptr = vec_grad_flags.data();
    for (long i = 0; i < vec_grad_flags.size(); ++i) {
        if (grad_flags_ptr[i]) {
            grad_indices.push_back(n + n_grad++);
        } else {
            grad_indices.push_back(-1);
        }
    }
    const long n_rows = n + n_grad * dim;
    const long n_cols = n_rows;

    const double a1 = std::sqrt(3.) / setting->scale;
    const double b = a1 * a1;
    Eigen::MatrixXd k_mat(n_rows, n_cols);
    for (long j = 0; j < n; ++j) {
        std::vector<double> diff_ij(dim);
        k_mat(j, j) = 1.0;
        if (vec_grad_flags[j]) {
            for (long k = 0, kj = grad_indices[j]; k < dim; ++k, kj += n_grad) {
                k_mat(kj, kj) = b;  // cov(df_j/dx_k, df_j/dx_k)
                k_mat(j, kj) = 0.;  // cov(f_j, df_j/dx_k)
                k_mat(kj, j) = 0.;  // cov(df_j/dx_k, f_j)
                for (long l = k + 1, lj = kj + n_grad; l < dim; ++l, lj += n_grad) {
                    k_mat(kj, lj) = 0.;  // cov(df_j/dx_k, df_j/dx_l)
                    k_mat(lj, kj) = 0.;  // cov(df_j/dx_l, df_j/dx_k)
                }
            }
        }

        const double *xj_ptr = mat_x.col(j).data();
        for (long i = j + 1; i < n; ++i) {
            double r = 0;
            const double *xi_ptr = mat_x.col(i).data();
            for (long k = 0; k < dim; ++k) {
                double &dx = diff_ij[k];
                dx = xi_ptr[k] - xj_ptr[k];
                r += dx * dx;
            }
            r = std::sqrt(r);  // norm(xi - xj)
            const double exp_term = std::exp(-a1 * r);

            // cov(f_i, f_j) = cov(f_j, f_i)
            double &k_ij = k_mat(i, j);
            k_ij = InlineMatern32(a1, r, exp_term);
            k_mat(j, i) = k_ij;

            const double b_exp_term = b * exp_term;
            if (vec_grad_flags[j]) {
                // cov(df_j, f_i) = cov(f_i, df_j)
                for (long k = 0, kj = grad_indices[j]; k < dim; ++k, kj += n_grad) {
                    double &k_i_kj = k_mat(i, kj);
                    // cov(f_i, df_j/dx_k)
                    k_i_kj = InlineMatern32X1BetweenGradx2(diff_ij[k], b_exp_term);
                    k_mat(kj, i) = k_i_kj;  // cov(df_j/dx_k, f_i)
                }

                if (vec_grad_flags[i]) {
                    for (long k = 0, ki = grad_indices[i], kj = grad_indices[j]; k < dim;
                         ++k, ki += n_grad, kj += n_grad) {
                        double &k_ki_j = k_mat(ki, j);
                        // cov(df_i, f_j) = -cov(f_i, df_j) = cov(f_j, df_i)
                        k_ki_j = -k_mat(i, kj);
                        // cov(f_j, df_i) = -cov(df_j, f_i) = -cov(f_i, df_j)
                        k_mat(j, ki) = k_ki_j;
                    }

                    // cov(df_j, df_i) = cov(df_i, df_j)
                    for (long k = 0, ki = grad_indices[i], kj = grad_indices[j]; k < dim;
                         ++k, ki += n_grad, kj += n_grad) {
                        // between Dim-k and Dim-k
                        const double &dxk = diff_ij[k];
                        double &k_kj_ki = k_mat(kj, ki);
                        k_kj_ki = InlineMatern32Gradx1BetweenGradx2(  // cov(df_j/dx_k, df_i/dx_k)
                            a1,
                            b,
                            1.,
                            dxk,
                            dxk,
                            r,
                            b_exp_term);
                        k_mat(ki, kj) = k_kj_ki;  // cov(df_i/dx_k, df_j/dx_k)
                        for (long l = k + 1, li = ki + n_grad, lj = kj + n_grad; l < dim;
                             ++l, li += n_grad, lj += n_grad) {
                            // between Dim-k and Dim-l
                            const double &dxl = diff_ij[l];
                            double &k_kj_li = k_mat(kj, li);
                            // cov(df_j/dx_k, df_i/dx_l)
                            k_kj_li = InlineMatern32Gradx1BetweenGradx2(
                                a1,
                                b,
                                0.,
                                dxk,
                                dxl,
                                r,
                                b_exp_term);
                            k_mat(lj, ki) = k_kj_li;  // cov(df_j/dx_l, df_i/dx_k)
                            k_mat(li, kj) = k_kj_li;  // cov(df_i/dx_l, df_j/dx_k)
                            k_mat(ki, lj) = k_kj_li;  // cov(df_i/dx_k, df_j/dx_l)
                        }
                    }
                }
            } else if (vec_grad_flags[i]) {
                // cov(f_j, df_i) = cov(df_i, f_j)
                for (long k = 0, ki = grad_indices[i]; k < dim; ++k, ki += n_grad) {
                    double &k_ki_j = k_mat(ki, j);
                    // cov(f_j, df_i)
                    k_ki_j = InlineMatern32X1BetweenGradx2(-diff_ij[k], b_exp_term);
                    k_mat(j, ki) = k_ki_j;
                }
            }
        }  // for (long i = j + 1; i < n; ++i)
    }  // for (long j = 0; j < n; ++j)
    return k_mat;
}

TEST(Matern32, 3D) {
    GTEST_PREPARE_OUTPUT_DIR();

    const auto kernel_setting = std::make_shared<erl::covariance::Matern32_3d::Setting>();
    kernel_setting->x_dim = 3;
    auto matern32_old = std::make_shared<erl::covariance::Matern32_3d>(kernel_setting);

    Eigen::MatrixXd mat_x =
        erl::common::LoadEigenMatrixFromTextFile<double>(gtest_src_dir / "x_train.txt");
    Eigen::VectorXl vec_grad_flags = Eigen::VectorXb::Random(mat_x.cols()).cast<long>();
    const long num_samples_with_gradient = vec_grad_flags.cast<long>().sum();

    auto [rows, cols] =
        matern32_old->GetMinimumKtrainSize(mat_x.cols(), num_samples_with_gradient, 3);
    Eigen::MatrixXd k_mat1(rows, cols);

    constexpr long n_tests = 1000;
    {
        erl::common::BlockTimer<std::chrono::milliseconds> timer("Matern32Old");
        (void) timer;
        for (long i = 0; i < n_tests; ++i) {
            Eigen::MatrixXd alpha;
            (void) matern32_old
                ->ComputeKtrainWithGradient(mat_x, mat_x.cols(), vec_grad_flags, k_mat1, alpha);
        }
    }

    Eigen::MatrixXd k_mat2;
    {
        erl::common::BlockTimer<std::chrono::milliseconds> timer("Matern32New");
        (void) timer;
        for (long i = 0; i < n_tests; ++i) {
            k_mat2 = matern32_new(kernel_setting, mat_x, vec_grad_flags);
        }
    }

    Eigen::MatrixXd k_mat3;
    {
        erl::common::BlockTimer<std::chrono::milliseconds> timer("Matern32New2");
        (void) timer;
        for (long i = 0; i < n_tests; ++i) {
            k_mat3 = matern32_new2(kernel_setting, mat_x, vec_grad_flags);
        }
    }

    // const long n = mat_x.cols();
    // EXPECT_TRUE(k_mat1.topLeftCorner(n, n).isApprox(k_mat2.topLeftCorner(n, n)));

    std::cout << "vec_grad_flags = " << vec_grad_flags.transpose()
              << ", n_grad = " << vec_grad_flags.cast<int>().sum() << std::endl;
    for (long j = 0; j < cols; ++j) {
        for (long i = 0; i < rows; ++i) {
            ASSERT_NEAR(k_mat1(i, j), k_mat2(i, j), 1.e-6)
                << "i = " << i << ", j = " << j << ", mat_x.cols() = " << mat_x.cols() << std::endl;
        }
    }
    // EXPECT_TRUE(k_mat1.isApprox(k_mat3));
    for (long j = 0; j < cols; ++j) {
        for (long i = 0; i < rows; ++i) {
            ASSERT_NEAR(k_mat1(i, j), k_mat3(i, j), 1.e-6)
                << "i = " << i << ", j = " << j << ", mat_x.cols() = " << mat_x.cols() << std::endl;
        }
    }
}
