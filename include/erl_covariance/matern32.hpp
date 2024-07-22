#pragma once

#include "covariance.hpp"

namespace erl::covariance {

    static double
    InlineMatern32(const double alpha, const double a, const double r, const double exp_term) {
        // return (alpha + a1 * r) * std::exp(-a2 * r);
        return (alpha + a * r) * exp_term;
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

    template<long Dim>
    class Matern32 : public Covariance {

    public:
        [[nodiscard]] std::shared_ptr<Covariance>
        Create(std::shared_ptr<Setting> setting) const override {
            if (setting == nullptr) { setting = std::make_shared<Setting>(); }
            return std::make_shared<Matern32>(std::move(setting));
        }

        explicit Matern32(std::shared_ptr<Setting> setting)
            : Covariance(std::move(setting)) {
            if (Dim != Eigen::Dynamic) { m_setting_->x_dim = Dim; }  // set x_dim
        }

        [[nodiscard]] std::string
        GetCovarianceType() const override {
            if (Dim == Eigen::Dynamic) { return "Matern32_Xd"; }
            return "Matern32_" + std::to_string(Dim) + "D";
        }

        [[nodiscard]] std::pair<long, long>
        ComputeKtrain(const Eigen::Ref<const Eigen::MatrixXd> &mat_x, const long num_samples, Eigen::MatrixXd &k_mat) const final {
            ERL_DEBUG_ASSERT(k_mat.rows() >= num_samples, "k_mat.rows() = {}, it should be >= {}.", k_mat.rows(), num_samples);
            ERL_DEBUG_ASSERT(k_mat.cols() >= num_samples, "k_mat.cols() = {}, it should be >= {}.", k_mat.cols(), num_samples);
            long dim;
            if constexpr (Dim == Eigen::Dynamic) {
                dim = mat_x.rows();
            } else {
                dim = Dim;
            }
            const double alpha = m_setting_->alpha;
            const double a2 = std::sqrt(3.) / m_setting_->scale;
            const double a1 = a2 * alpha;
            const long stride = k_mat.outerStride();
            for (long j = 0; j < num_samples; ++j) {
                double *k_mat_j_ptr = k_mat.col(j).data();   // use raw pointer to improve performance
                const double *xj_ptr = mat_x.col(j).data();  // use raw pointer to improve performance
                k_mat_j_ptr[j] = alpha;                      // k_mat(j, j)
                if (j + 1 >= num_samples) { continue; }
                double *k_ji_ptr = &k_mat(j, j + 1);  // k_mat(j, i)
                for (long i = j + 1; i < num_samples; ++i, k_ji_ptr += stride) {
                    const double *xi_ptr = mat_x.col(i).data();
                    double r = 0.0;
                    for (long k = 0; k < dim; ++k) {
                        const double dx = xi_ptr[k] - xj_ptr[k];
                        r += dx * dx;
                    }
                    r = std::sqrt(r);               // (mat_x.col(i) - mat_x.col(j)).norm();
                    double &k_ij = k_mat_j_ptr[i];  // k_mat(i, j)
                    k_ij = InlineMatern32(alpha, a1, r, std::exp(-a2 * r));
                    *k_ji_ptr = k_ij;  // k_mat(j, i) = k_ij;
                }
            }
            return {num_samples, num_samples};
        }

        [[nodiscard]] std::pair<long, long>
        ComputeKtrain(
            const Eigen::Ref<const Eigen::MatrixXd> &mat_x,
            const Eigen::Ref<const Eigen::VectorXd> &vec_var_y,
            const long num_samples,
            Eigen::MatrixXd &k_mat) const final {
            ERL_DEBUG_ASSERT(k_mat.rows() >= num_samples, "k_mat.rows() = {}, it should be >= {}.", k_mat.rows(), num_samples);
            ERL_DEBUG_ASSERT(k_mat.cols() >= num_samples, "k_mat.cols() = {}, it should be >= {}.", k_mat.cols(), num_samples);
            long dim;
            if constexpr (Dim == Eigen::Dynamic) {
                dim = mat_x.rows();
            } else {
                dim = Dim;
            }
            const double alpha = m_setting_->alpha;
            const double a2 = std::sqrt(3.) / m_setting_->scale;
            const double a1 = a2 * alpha;
            const long stride = k_mat.outerStride();
            for (long j = 0; j < num_samples; ++j) {
                double *k_mat_j_ptr = k_mat.col(j).data();   // use raw pointer to improve performance
                const double *xj_ptr = mat_x.col(j).data();  // use raw pointer to improve performance
                k_mat_j_ptr[j] = alpha + vec_var_y[j];       // k_mat(j, j)
                if (j + 1 >= num_samples) { continue; }
                double *k_ji_ptr = &k_mat(j, j + 1);  // k_mat(j, i)
                for (long i = j + 1; i < num_samples; ++i, k_ji_ptr += stride) {
                    const double *xi_ptr = mat_x.col(i).data();
                    double r = 0.0;
                    for (long k = 0; k < dim; ++k) {
                        const double dx = xi_ptr[k] - xj_ptr[k];
                        r += dx * dx;
                    }
                    r = std::sqrt(r);               // (mat_x.col(i) - mat_x.col(j)).norm();
                    double &k_ij = k_mat_j_ptr[i];  // k_mat(i, j)
                    k_ij = InlineMatern32(alpha, a1, r, std::exp(-a2 * r));
                    *k_ji_ptr = k_ij;  // k_mat(j, i) = k_ij;
                }
            }
            return {num_samples, num_samples};
        }

        [[nodiscard]] std::pair<long, long>
        ComputeKtest(
            const Eigen::Ref<const Eigen::MatrixXd> &mat_x1,
            const long num_samples1,
            const Eigen::Ref<const Eigen::MatrixXd> &mat_x2,
            const long num_samples2,
            Eigen::MatrixXd &k_mat) const final {
            ERL_DEBUG_ASSERT(mat_x1.rows() == mat_x2.rows(), "Sample vectors stored in x1 and x_2 should have the same dimension.");
            ERL_DEBUG_ASSERT(k_mat.rows() >= num_samples1, "k_mat.rows() = {}, it should be >= {}.", k_mat.rows(), num_samples1);
            ERL_DEBUG_ASSERT(k_mat.cols() >= num_samples2, "k_mat.cols() = {}, it should be >= {}.", k_mat.cols(), num_samples2);
            long dim;
            if constexpr (Dim == Eigen::Dynamic) {
                dim = mat_x1.rows();
            } else {
                dim = Dim;
            }
            const double a2 = std::sqrt(3.) / m_setting_->scale;
            const double a1 = a2 * m_setting_->alpha;
            for (long j = 0; j < num_samples2; ++j) {
                const double *x2_ptr = mat_x2.col(j).data();
                double *col_j_ptr = k_mat.col(j).data();
                for (long i = 0; i < num_samples1; ++i) {
                    const double *x1_ptr = mat_x1.col(i).data();
                    double r = 0;
                    for (long k = 0; k < dim; ++k) {
                        const double dx = x1_ptr[k] - x2_ptr[k];
                        r += dx * dx;
                    }
                    r = std::sqrt(r);  // (mat_x1.col(i) - mat_x2.col(j)).norm();
                    col_j_ptr[i] = InlineMatern32(m_setting_->alpha, a1, r, std::exp(-a2 * r));
                }
            }
            return {num_samples1, num_samples2};
        }

        [[nodiscard]] std::pair<long, long>
        ComputeKtrainWithGradient(
            const Eigen::Ref<const Eigen::MatrixXd> &mat_x,
            const long num_samples,
            Eigen::VectorXl &vec_grad_flags,
            Eigen::MatrixXd &k_mat) const final {

            long dim;
            if constexpr (Dim == Eigen::Dynamic) {
                dim = mat_x.rows();
            } else {
                dim = Dim;
            }

            ERL_DEBUG_ASSERT(mat_x.rows() == dim, "Each column of mat_x should be {}-D vector.", dim);
            long n_grad = 0;
            for (long i = 0; i < num_samples; ++i) {
                if (long &flag = vec_grad_flags[i]; flag > 0) { flag = num_samples + n_grad++; }
            }
            long n_rows = num_samples + n_grad * dim;
            long n_cols = n_rows;
            ERL_DEBUG_ASSERT(k_mat.rows() >= n_rows, "k_mat.rows() = {}, it should be >= {}.", k_mat.rows(), n_rows);
            ERL_DEBUG_ASSERT(k_mat.cols() >= n_cols, "k_mat.cols() = {}, it should be >= {}.", k_mat.cols(), n_cols);

            const double alpha = m_setting_->alpha;
            const double a2 = std::sqrt(3.) / m_setting_->scale;
            const double a1 = a2 * alpha;
            const double b = a2 * a2 * alpha;
            // buffer to store the difference between x1_i and x2_j
            Eigen::Vector<double, Dim> diff_ij;          // avoid memory allocation on the heap
            Eigen::Vector<double *, Dim> k_mat_kj_ptrs;  // avoid memory allocation on the heap
            Eigen::Vector<double *, Dim> k_mat_ki_ptrs;  // avoid memory allocation on the heap
            if constexpr (Dim == Eigen::Dynamic) {
                diff_ij.resize(dim);
                k_mat_kj_ptrs.resize(dim);
                k_mat_ki_ptrs.resize(dim);
            }
            for (long j = 0; j < num_samples; ++j) {
                double *k_mat_j_ptr = k_mat.col(j).data();
                k_mat_j_ptr[j] = alpha;  // k_mat(j, j)
                if (vec_grad_flags[j]) {
                    for (long k = 0, kj = vec_grad_flags[j]; k < dim; ++k, kj += n_grad) { k_mat_kj_ptrs[k] = k_mat.col(kj).data(); }

                    for (long k = 0, kj = vec_grad_flags[j]; k < dim; ++k, kj += n_grad) {
                        k_mat_kj_ptrs[k][kj] = b;  // k_mat(kj, kj) = cov(df_j/dx_k, f_j)
                        k_mat_kj_ptrs[k][j] = 0.;  // k_mat(j, kj) = cov(df_j/dx_k, f_j)
                        k_mat_j_ptr[kj] = 0.;      // k_mat(kj, j) = cov(df_j/dx_k, f_j)
                        for (long l = k + 1, lj = kj + n_grad; l < dim; ++l, lj += n_grad) {
                            k_mat_kj_ptrs[l][kj] = 0.;  // k_mat(kj, lj) = cov(df_j/dx_k, df_j/dx_l)
                            k_mat_kj_ptrs[k][lj] = 0.;  // k_mat(lj, kj) = cov(df_j/dx_l, df_j/dx_k)
                        }
                    }
                }

                const double *xj_ptr = mat_x.col(j).data();
                for (long i = j + 1; i < num_samples; ++i) {
                    const double *xi_ptr = mat_x.col(i).data();
                    double *k_mat_i_ptr = k_mat.col(i).data();
                    double r = 0;
                    for (long k = 0; k < dim; ++k) {
                        double &dx = diff_ij[k];
                        dx = xi_ptr[k] - xj_ptr[k];
                        r += dx * dx;
                    }
                    r = std::sqrt(r);  // norm(xi - xj)
                    const double exp_term = std::exp(-a2 * r);

                    // cov(f_i, f_j) = cov(f_j, f_i)
                    double &k_ij = k_mat_j_ptr[i];  // k_mat(i, j)
                    k_ij = InlineMatern32(alpha, a1, r, exp_term);
                    k_mat_i_ptr[j] = k_ij;  // k_mat(j, i)

                    const double b_exp_term = b * exp_term;
                    if (vec_grad_flags[j]) {
                        // cov(df_j, f_i) = cov(f_i, df_j)
                        for (long k = 0, kj = vec_grad_flags[j]; k < dim; ++k, kj += n_grad) {
                            double &k_i_kj = k_mat_kj_ptrs[k][i];                            // k_mat(i, kj)
                            k_i_kj = InlineMatern32X1BetweenGradx2(diff_ij[k], b_exp_term);  // cov(f_i, df_j/dx_k)
                            k_mat_i_ptr[kj] = k_i_kj;                                        // k_mat(kj, i) = cov(df_j/dx_k, f_i)
                        }

                        if (vec_grad_flags[i]) {
                            for (long k = 0, ki = vec_grad_flags[i]; k < dim; ++k, ki += n_grad) { k_mat_ki_ptrs[k] = k_mat.col(ki).data(); }

                            for (long k = 0, ki = vec_grad_flags[i], kj = vec_grad_flags[j]; k < dim; ++k, ki += n_grad, kj += n_grad) {
                                double &k_ki_j = k_mat_j_ptr[ki];  // k_mat(ki, j), use reference to improve performance
                                k_ki_j = -k_mat_kj_ptrs[k][i];     // cov(df_i, f_j) = -cov(f_i, df_j) = cov(f_j, df_i)
                                k_mat_ki_ptrs[k][j] = k_ki_j;      // k_mat(j, ki) = cov(f_j, df_i) = -cov(df_j, f_i) = -cov(f_i, df_j)

                                // cov(df_j, df_i) = cov(df_i, df_j)
                                // between Dim-k and Dim-k
                                const double &dxk = diff_ij[k];                                                   // use reference to improve performance
                                double &k_kj_ki = k_mat_ki_ptrs[k][kj];                                           // k_mat(kj, ki)
                                k_kj_ki = InlineMatern32Gradx1BetweenGradx2(a2, b, 1., dxk, dxk, r, b_exp_term);  // cov(df_j/dx_k, df_i/dx_k)
                                k_mat_kj_ptrs[k][ki] = k_kj_ki;                                                   // cov(df_i/dx_k, df_j/dx_k)
                                for (long l = k + 1, li = ki + n_grad, lj = kj + n_grad; l < dim; ++l, li += n_grad, lj += n_grad) {
                                    // between Dim-k and Dim-l
                                    const double &dxl = diff_ij[l];
                                    double &k_kj_li = k_mat_ki_ptrs[l][kj];                                           // k_mat(kj, li)
                                    k_kj_li = InlineMatern32Gradx1BetweenGradx2(a2, b, 0., dxk, dxl, r, b_exp_term);  // cov(df_j/dx_k, df_i/dx_l)
                                    k_mat_ki_ptrs[k][lj] = k_kj_li;  // k_mat(lj, ki) = cov(df_j/dx_l, df_i/dx_k)
                                    k_mat_kj_ptrs[k][li] = k_kj_li;  // k_mat(li, kj) = cov(df_i/dx_l, df_j/dx_k)
                                    k_mat_kj_ptrs[l][ki] = k_kj_li;  // k_mat(ki, lj) = cov(df_i/dx_k, df_j/dx_l)
                                }
                            }
                        }
                    } else if (vec_grad_flags[i]) {
                        // cov(f_j, df_i) = cov(df_i, f_j)
                        for (long k = 0, ki = vec_grad_flags[i]; k < dim; ++k, ki += n_grad) {
                            double &k_ki_j = k_mat_j_ptr[ki];                                 // k_mat(ki, j), use reference to improve performance
                            k_ki_j = InlineMatern32X1BetweenGradx2(-diff_ij[k], b_exp_term);  // cov(f_j, df_i)
                            k_mat(j, ki) = k_ki_j;
                        }
                    }
                }  // for (long i = j + 1; i < n; ++i)
            }  // for (long j = 0; j < n; ++j)
            return {n_rows, n_cols};
        }

        [[nodiscard]] std::pair<long, long>
        ComputeKtrainWithGradient(
            const Eigen::Ref<const Eigen::MatrixXd> &mat_x,
            const long num_samples,
            Eigen::VectorXl &vec_grad_flags,
            const Eigen::Ref<const Eigen::VectorXd> &vec_var_x,
            const Eigen::Ref<const Eigen::VectorXd> &vec_var_y,
            const Eigen::Ref<const Eigen::VectorXd> &vec_var_grad,
            Eigen::MatrixXd &k_mat) const final {

            long dim;
            if constexpr (Dim == Eigen::Dynamic) {
                dim = mat_x.rows();
            } else {
                dim = Dim;
            }

            ERL_DEBUG_ASSERT(mat_x.rows() == dim, "Each column of mat_x should be {}-D vector.", dim);
            long n_grad = 0;
            for (long i = 0; i < num_samples; ++i) {
                if (long &flag = vec_grad_flags[i]; flag > 0) { flag = num_samples + n_grad++; }
            }
            long n_rows = num_samples + n_grad * dim;
            long n_cols = n_rows;
            ERL_DEBUG_ASSERT(k_mat.rows() >= n_rows, "k_mat.rows() = {}, it should be >= {}.", k_mat.rows(), n_rows);
            ERL_DEBUG_ASSERT(k_mat.cols() >= n_cols, "k_mat.cols() = {}, it should be >= {}.", k_mat.cols(), n_cols);

            const double alpha = m_setting_->alpha;
            const double a2 = std::sqrt(3.) / m_setting_->scale;
            const double a1 = a2 * alpha;
            const double b = a2 * a2 * alpha;
            // buffer to store the difference between x1_i and x2_j
            Eigen::Vector<double, Dim> diff_ij;          // avoid memory allocation on the heap
            Eigen::Vector<double *, Dim> k_mat_kj_ptrs;  // avoid memory allocation on the heap
            Eigen::Vector<double *, Dim> k_mat_ki_ptrs;  // avoid memory allocation on the heap
            if constexpr (Dim == Eigen::Dynamic) {
                diff_ij.resize(dim);
                k_mat_kj_ptrs.resize(dim);
                k_mat_ki_ptrs.resize(dim);
            }
            for (long j = 0; j < num_samples; ++j) {
                double *k_mat_j_ptr = k_mat.col(j).data();
                k_mat_j_ptr[j] = alpha + vec_var_x[j] + vec_var_y[j];  // k_mat(j, j)
                if (vec_grad_flags[j]) {
                    for (long k = 0, kj = vec_grad_flags[j]; k < dim; ++k, kj += n_grad) { k_mat_kj_ptrs[k] = k_mat.col(kj).data(); }

                    for (long k = 0, kj = vec_grad_flags[j]; k < dim; ++k, kj += n_grad) {
                        k_mat_kj_ptrs[k][kj] = b + vec_var_grad[j];  // k_mat(kj, kj) = cov(df_j/dx_k, f_j)
                        k_mat_kj_ptrs[k][j] = 0.;                    // k_mat(j, kj) = cov(df_j/dx_k, f_j)
                        k_mat_j_ptr[kj] = 0.;                        // k_mat(kj, j) = cov(df_j/dx_k, f_j)
                        for (long l = k + 1, lj = kj + n_grad; l < dim; ++l, lj += n_grad) {
                            k_mat_kj_ptrs[l][kj] = 0.;  // k_mat(kj, lj) = cov(df_j/dx_k, df_j/dx_l)
                            k_mat_kj_ptrs[k][lj] = 0.;  // k_mat(lj, kj) = cov(df_j/dx_l, df_j/dx_k)
                        }
                    }
                }

                const double *xj_ptr = mat_x.col(j).data();
                for (long i = j + 1; i < num_samples; ++i) {
                    const double *xi_ptr = mat_x.col(i).data();
                    double *k_mat_i_ptr = k_mat.col(i).data();
                    double r = 0;
                    for (long k = 0; k < dim; ++k) {
                        double &dx = diff_ij[k];
                        dx = xi_ptr[k] - xj_ptr[k];
                        r += dx * dx;
                    }
                    r = std::sqrt(r);  // norm(xi - xj)
                    const double exp_term = std::exp(-a2 * r);

                    // cov(f_i, f_j) = cov(f_j, f_i)
                    double &k_ij = k_mat_j_ptr[i];  // k_mat(i, j)
                    k_ij = InlineMatern32(alpha, a1, r, exp_term);
                    k_mat_i_ptr[j] = k_ij;  // k_mat(j, i)

                    const double b_exp_term = b * exp_term;
                    if (vec_grad_flags[j]) {
                        // cov(df_j, f_i) = cov(f_i, df_j)
                        for (long k = 0, kj = vec_grad_flags[j]; k < dim; ++k, kj += n_grad) {
                            double &k_i_kj = k_mat_kj_ptrs[k][i];                            // k_mat(i, kj)
                            k_i_kj = InlineMatern32X1BetweenGradx2(diff_ij[k], b_exp_term);  // cov(f_i, df_j/dx_k)
                            k_mat_i_ptr[kj] = k_i_kj;                                        // k_mat(kj, i) = cov(df_j/dx_k, f_i)
                        }

                        if (vec_grad_flags[i]) {
                            for (long k = 0, ki = vec_grad_flags[i]; k < dim; ++k, ki += n_grad) { k_mat_ki_ptrs[k] = k_mat.col(ki).data(); }

                            for (long k = 0, ki = vec_grad_flags[i], kj = vec_grad_flags[j]; k < dim; ++k, ki += n_grad, kj += n_grad) {
                                double &k_ki_j = k_mat_j_ptr[ki];  // k_mat(ki, j), use reference to improve performance
                                k_ki_j = -k_mat_kj_ptrs[k][i];     // cov(df_i, f_j) = -cov(f_i, df_j) = cov(f_j, df_i)
                                k_mat_ki_ptrs[k][j] = k_ki_j;      // k_mat(j, ki) = cov(f_j, df_i) = -cov(df_j, f_i) = -cov(f_i, df_j)

                                // cov(df_j, df_i) = cov(df_i, df_j)
                                // between Dim-k and Dim-k
                                const double &dxk = diff_ij[k];                                                   // use reference to improve performance
                                double &k_kj_ki = k_mat_ki_ptrs[k][kj];                                           // k_mat(kj, ki)
                                k_kj_ki = InlineMatern32Gradx1BetweenGradx2(a2, b, 1., dxk, dxk, r, b_exp_term);  // cov(df_j/dx_k, df_i/dx_k)
                                k_mat_kj_ptrs[k][ki] = k_kj_ki;                                                   // cov(df_i/dx_k, df_j/dx_k)
                                for (long l = k + 1, li = ki + n_grad, lj = kj + n_grad; l < dim; ++l, li += n_grad, lj += n_grad) {
                                    // between Dim-k and Dim-l
                                    const double &dxl = diff_ij[l];
                                    double &k_kj_li = k_mat_ki_ptrs[l][kj];                                           // k_mat(kj, li)
                                    k_kj_li = InlineMatern32Gradx1BetweenGradx2(a2, b, 0., dxk, dxl, r, b_exp_term);  // cov(df_j/dx_k, df_i/dx_l)
                                    k_mat_ki_ptrs[k][lj] = k_kj_li;  // k_mat(lj, ki) = cov(df_j/dx_l, df_i/dx_k)
                                    k_mat_kj_ptrs[k][li] = k_kj_li;  // k_mat(li, kj) = cov(df_i/dx_l, df_j/dx_k)
                                    k_mat_kj_ptrs[l][ki] = k_kj_li;  // k_mat(ki, lj) = cov(df_i/dx_k, df_j/dx_l)
                                }
                            }
                        }
                    } else if (vec_grad_flags[i]) {
                        // cov(f_j, df_i) = cov(df_i, f_j)
                        for (long k = 0, ki = vec_grad_flags[i]; k < dim; ++k, ki += n_grad) {
                            double &k_ki_j = k_mat_j_ptr[ki];                                 // k_mat(ki, j), use reference to improve performance
                            k_ki_j = InlineMatern32X1BetweenGradx2(-diff_ij[k], b_exp_term);  // cov(f_j, df_i)
                            k_mat(j, ki) = k_ki_j;
                        }
                    }
                }  // for (long i = j + 1; i < n; ++i)
            }  // for (long j = 0; j < n; ++j)
            return {n_rows, n_cols};
        }

        [[nodiscard]] std::pair<long, long>
        ComputeKtestWithGradient(
            const Eigen::Ref<const Eigen::MatrixXd> &mat_x1,
            const long num_samples1,
            const Eigen::Ref<const Eigen::VectorXl> &vec_grad1_flags,
            const Eigen::Ref<const Eigen::MatrixXd> &mat_x2,
            const long num_samples2,
            Eigen::MatrixXd &k_mat) const final {

            long dim;
            if constexpr (Dim == Eigen::Dynamic) {
                dim = mat_x1.rows();
            } else {
                dim = Dim;
            }

            ERL_DEBUG_ASSERT(mat_x1.rows() == dim, "Each column of mat_x1 should be {}-D vector.", dim);
            ERL_DEBUG_ASSERT(mat_x2.rows() == dim, "Each column of mat_x2 should be {}-D vector.", dim);
            const long n_grad = vec_grad1_flags.head(num_samples1).count();
            const long n_rows = num_samples1 + n_grad * dim;
            const long n_cols = num_samples2 * (dim + 1);
            ERL_DEBUG_ASSERT(k_mat.rows() >= n_rows, "k_mat.rows() = {}, it should be >= {}.", k_mat.rows(), n_rows);
            ERL_DEBUG_ASSERT(k_mat.cols() >= n_cols, "k_mat.cols() = {}, it should be >= {}.", k_mat.cols(), n_cols);

            const double a2 = std::sqrt(3.) / m_setting_->scale;
            const double a1 = a2 * m_setting_->alpha;
            const double b = a2 * a2 * m_setting_->alpha;
            // buffer to store the difference between x1_i and x2_j
            Eigen::Vector<double, Dim> diff_ij;          // avoid memory allocation on the heap
            Eigen::Vector<double *, Dim> k_mat_kj_ptrs;  // avoid memory allocation on the heap
            if constexpr (Dim == Eigen::Dynamic) {
                diff_ij.resize(dim);
                k_mat_kj_ptrs.resize(dim);
            }
            for (long j = 0; j < num_samples2; ++j) {
                const double *x2_j_ptr = mat_x2.col(j).data();
                double *k_mat_j_ptr = k_mat.col(j).data();
                for (long k = 0, kj = j + num_samples2; k < dim; ++k, kj += num_samples2) { k_mat_kj_ptrs[k] = k_mat.col(kj).data(); }

                for (long i = 0, ki_init = num_samples1; i < num_samples1; ++i) {
                    const double *x1_i_ptr = mat_x1.col(i).data();
                    double r = 0;
                    for (long k = 0; k < dim; ++k) {
                        double &dx = diff_ij[k];
                        dx = x1_i_ptr[k] - x2_j_ptr[k];
                        r += dx * dx;
                    }
                    r = std::sqrt(r);                                                                 // (mat_x1.col(i) - mat_x2.col(j)).norm();
                    const double exp_term = std::exp(-a2 * r);                                        // exp(-a2 * r), which is frequently used
                    const double b_exp_term = b * exp_term;                                           // b * exp(-a2 * r)
                    k_mat_j_ptr[i] = InlineMatern32(m_setting_->alpha, a1, r, exp_term);              // cov(f1_i, f2_j)
                    for (long k = 0, kj = j + num_samples2; k < dim; ++k, kj += num_samples2) {       // cov(f1_i, df2_j/dx_k)
                        k_mat_kj_ptrs[k][i] = InlineMatern32X1BetweenGradx2(diff_ij[k], b_exp_term);  // k_mat(i, kj)
                    }

                    if (!vec_grad1_flags[i]) { continue; }
                    for (long k = 0, ki = ki_init, kj = j + num_samples2; k < dim; ++k, ki += n_grad, kj += num_samples2) {
                        k_mat_j_ptr[ki] = -k_mat_kj_ptrs[k][i];  // k_mat(ki, j) = -k_mat(i, kj), i.e. cov(df1_i/dx_k, f2_j) = -cov(f1_i, df2_j/dx_k)

                        // between Dim-k and Dim-k
                        const double &dxk = diff_ij[k];
                        // k_mat(ki, kj) = cov(df1_i/dx_k, df2_j/dx_k)
                        k_mat_kj_ptrs[k][ki] = InlineMatern32Gradx1BetweenGradx2(a2, b, 1., dxk, dxk, r, b_exp_term);

                        for (long l = k + 1, li = ki + n_grad, lj = kj + num_samples2; l < dim; ++l, li += n_grad, lj += num_samples2) {
                            // between Dim-k and Dim-l
                            const double &dxl = diff_ij[l];
                            double &k_ki_lj = k_mat_kj_ptrs[l][li];                                           // k_mat(li, lj)
                            k_ki_lj = InlineMatern32Gradx1BetweenGradx2(a2, b, 0., dxk, dxl, r, b_exp_term);  // cov(df1_i/dx_k, df2_j/dx_l)
                            k_mat_kj_ptrs[k][li] = k_ki_lj;                                                   // k_mat(li, kj) = cov(df1_i/dx_l, df2_j/dx_k)
                        }
                    }
                    ++ki_init;
                }
            }
            return {n_rows, n_cols};
        }
    };

    using Matern32_1D = Matern32<1>;
    using Matern32_2D = Matern32<2>;
    using Matern32_3D = Matern32<3>;
    using Matern32_Xd = Matern32<Eigen::Dynamic>;

    ERL_REGISTER_COVARIANCE(Matern32_1D);
    ERL_REGISTER_COVARIANCE(Matern32_2D);
    ERL_REGISTER_COVARIANCE(Matern32_3D);
    ERL_REGISTER_COVARIANCE(Matern32_Xd);

}  // namespace erl::covariance
