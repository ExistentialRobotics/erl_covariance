#pragma once

#include "covariance.hpp"

namespace erl::covariance {

    template<long Dim>
    class RadialBiasFunction : public Covariance {
        // ref: https://en.wikipedia.org/wiki/Radial_basis_function_kernel
    public:
        explicit RadialBiasFunction(std::shared_ptr<Setting> setting)
            : Covariance(std::move(setting)) {
            if (Dim != Eigen::Dynamic) { m_setting_->x_dim = Dim; }  // set x_dim
        }

        [[nodiscard]] std::string
        GetCovarianceType() const override {
            if (Dim == Eigen::Dynamic) { return "RadialBiasFunction_Xd"; }
            return "RadialBiasFunction_" + std::to_string(Dim) + "D";
        }

        [[nodiscard]] std::pair<long, long>
        ComputeKtrain(const Eigen::Ref<const Eigen::MatrixXd> &mat_x, const long num_samples, Eigen::MatrixXd &mat_k, Eigen::VectorXd & /*vec_alpha*/)
            const override {
            ERL_DEBUG_ASSERT(mat_k.rows() >= num_samples, "mat_k.rows() = {}, it should be >= {}.", mat_k.rows(), num_samples);
            ERL_DEBUG_ASSERT(mat_k.cols() >= num_samples, "mat_k.cols() = {}, it should be >= {}.", mat_k.cols(), num_samples);
            long dim;
            if constexpr (Dim == Eigen::Dynamic) {
                dim = mat_x.rows();
            } else {
                dim = Dim;
            }
            const double alpha = m_setting_->alpha;
            const double a = 0.5 / (m_setting_->scale * m_setting_->scale);
            const long stride = mat_k.outerStride();
            for (long j = 0; j < num_samples; ++j) {
                double *mat_k_j_ptr = mat_k.col(j).data();   // use raw pointer to improve performance
                const double *xj_ptr = mat_x.col(j).data();  // use raw pointer to improve performance
                mat_k_j_ptr[j] = alpha;                      // mat_k(j, j)
                if (j + 1 >= num_samples) { continue; }
                double *k_ji_ptr = &mat_k(j, j + 1);  // mat_k(j, i)
                for (long i = j + 1; i < num_samples; ++i, k_ji_ptr += stride) {
                    const double *xi_ptr = mat_x.col(i).data();
                    double r = 0.0;
                    for (long k = 0; k < dim; ++k) {
                        const double dx = xi_ptr[k] - xj_ptr[k];
                        r += dx * dx;
                    }
                    double &k_ij = mat_k_j_ptr[i];
                    k_ij = alpha * std::exp(-a * r);  // mat_k(i, j)
                    *k_ji_ptr = k_ij;                 // mat_k(j, i) = k_ij;
                }
            }
            return {num_samples, num_samples};
        }

        [[nodiscard]] std::pair<long, long>
        ComputeKtrain(
            const Eigen::Ref<const Eigen::MatrixXd> &mat_x,
            const Eigen::Ref<const Eigen::VectorXd> &vec_var_y,
            const long num_samples,
            Eigen::MatrixXd &mat_k,
            Eigen::VectorXd & /*vec_alpha*/) const override {
            ERL_DEBUG_ASSERT(mat_k.rows() >= num_samples, "mat_k.rows() = {}, it should be >= {}.", mat_k.rows(), num_samples);
            ERL_DEBUG_ASSERT(mat_k.cols() >= num_samples, "mat_k.cols() = {}, it should be >= {}.", mat_k.cols(), num_samples);
            long dim;
            if constexpr (Dim == Eigen::Dynamic) {
                dim = mat_x.rows();
            } else {
                dim = Dim;
            }
            const double alpha = m_setting_->alpha;
            const double a = 0.5 / (m_setting_->scale * m_setting_->scale);
            const long stride = mat_k.outerStride();
            for (long j = 0; j < num_samples; ++j) {
                double *mat_k_j_ptr = mat_k.col(j).data();   // use raw pointer to improve performance
                const double *xj_ptr = mat_x.col(j).data();  // use raw pointer to improve performance
                mat_k_j_ptr[j] = alpha + vec_var_y[j];       // mat_k(j, j)
                if (j + 1 >= num_samples) { continue; }
                double *k_ji_ptr = &mat_k(j, j + 1);  // mat_k(j, i)
                for (long i = j + 1; i < num_samples; ++i, k_ji_ptr += stride) {
                    const double *xi_ptr = mat_x.col(i).data();
                    double r = 0.0;
                    for (long k = 0; k < dim; ++k) {
                        const double dx = xi_ptr[k] - xj_ptr[k];
                        r += dx * dx;
                    }
                    double &k_ij = mat_k_j_ptr[i];  // mat_k(i, j)
                    k_ij = alpha * std::exp(-a * r);
                    *k_ji_ptr = k_ij;  // mat_k(j, i) = k_ij;
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
            Eigen::MatrixXd &mat_k) const override {
            ERL_DEBUG_ASSERT(mat_x1.rows() == mat_x2.rows(), "Sample vectors stored in x1 and x2 should have the same dimension.");
            ERL_DEBUG_ASSERT(mat_k.rows() >= num_samples1, "mat_k.rows() = {}, it should be >= {}.", mat_k.rows(), num_samples1);
            ERL_DEBUG_ASSERT(mat_k.cols() >= num_samples2, "mat_k.cols() = {}, it should be >= {}.", mat_k.cols(), num_samples2);
            long dim;
            if constexpr (Dim == Eigen::Dynamic) {
                dim = mat_x1.rows();
            } else {
                dim = Dim;
            }
            const double a = 0.5 / (m_setting_->scale * m_setting_->scale);
            const double alpha = m_setting_->alpha;
            for (long j = 0; j < num_samples2; ++j) {
                const double *x2_ptr = mat_x2.col(j).data();
                double *col_j_ptr = mat_k.col(j).data();
                for (long i = 0; i < num_samples1; ++i) {
                    const double *x1_ptr = mat_x1.col(i).data();
                    double r = 0.0;
                    for (long k = 0; k < dim; ++k) {
                        const double dx = x1_ptr[k] - x2_ptr[k];
                        r += dx * dx;
                    }
                    col_j_ptr[i] = alpha * std::exp(-a * r);
                }
            }
            return {num_samples1, num_samples2};
        }

        [[nodiscard]] std::pair<long, long>
        ComputeKtrainWithGradient(
            const Eigen::Ref<const Eigen::MatrixXd> &mat_x,
            const long num_samples,
            Eigen::VectorXl &vec_grad_flags,
            Eigen::MatrixXd &mat_k,
            Eigen::VectorXd & /*vec_alpha*/) const override {

            long dim;
            if constexpr (Dim == Eigen::Dynamic) {
                dim = mat_x.rows();
            } else {
                dim = Dim;
            }

            ERL_DEBUG_ASSERT(mat_x.rows() == dim, "Each column of mat_x should be {}-D vector.", dim);
            long n_grad = 0;
            long *grad_flags = vec_grad_flags.data();
            for (long i = 0; i < num_samples; ++i) {
                if (long &flag = grad_flags[i]; flag > 0) { flag = num_samples + n_grad++; }
            }
            long n_rows = num_samples + n_grad * dim;
            long n_cols = n_rows;
            ERL_DEBUG_ASSERT(mat_k.rows() >= n_rows, "mat_k.rows() = {}, it should be >= {}.", mat_k.rows(), n_rows);
            ERL_DEBUG_ASSERT(mat_k.cols() >= n_cols, "mat_k.cols() = {}, it should be >= {}.", mat_k.cols(), n_cols);

            const double alpha = m_setting_->alpha;
            const double l2_inv = 1.0 / (m_setting_->scale * m_setting_->scale);
            const double a = 0.5 * l2_inv;
            // buffer to store the difference between x1_i and x2_j
            Eigen::Vector<double, Dim> diff_ij;          // avoid memory allocation on the heap
            Eigen::Vector<double *, Dim> mat_k_kj_ptrs;  // avoid memory allocation on the heap
            Eigen::Vector<double *, Dim> mat_k_ki_ptrs;  // avoid memory allocation on the heap
            if constexpr (Dim == Eigen::Dynamic) {
                diff_ij.resize(dim);
                mat_k_kj_ptrs.resize(dim);
                mat_k_ki_ptrs.resize(dim);
            }
            for (long j = 0; j < num_samples; ++j) {
                double *mat_k_j_ptr = mat_k.col(j).data();
                mat_k_j_ptr[j] = alpha;  // mat_k(j, j)
                if (grad_flags[j]) {
                    for (long k = 0, kj = grad_flags[j]; k < dim; ++k, kj += n_grad) { mat_k_kj_ptrs[k] = mat_k.col(kj).data(); }

                    for (long k = 0, kj = grad_flags[j]; k < dim; ++k, kj += n_grad) {
                        mat_k_kj_ptrs[k][kj] = l2_inv;  // mat_k(kj, kj) = cov(df_j/dx_k, f_j)
                        mat_k_kj_ptrs[k][j] = 0.0;      // mat_k(j, kj) = cov(df_j/dx_k, f_j)
                        mat_k_j_ptr[kj] = 0.0;          // mat_k(kj, j) = cov(df_j/dx_k, f_j)
                        for (long l = k + 1, lj = kj + n_grad; l < dim; ++l, lj += n_grad) {
                            mat_k_kj_ptrs[l][kj] = 0.0;  // mat_k(kj, lj) = cov(df_j/dx_k, df_j/dx_l)
                            mat_k_kj_ptrs[k][lj] = 0.0;  // mat_k(lj, kj) = cov(df_j/dx_l, df_j/dx_k)
                        }
                    }
                }

                const double *xj_ptr = mat_x.col(j).data();
                for (long i = j + 1; i < num_samples; ++i) {
                    const double *xi_ptr = mat_x.col(i).data();
                    double *mat_k_i_ptr = mat_k.col(i).data();
                    double r = 0;
                    for (long k = 0; k < dim; ++k) {
                        double &dx = diff_ij[k];
                        dx = xi_ptr[k] - xj_ptr[k];
                        r += dx * dx;
                    }
                    double &k_ij = mat_k_j_ptr[i];    // mat_k(i, j)
                    k_ij = alpha * std::exp(-a * r);  // cov(f_i, f_j)
                    mat_k_i_ptr[j] = k_ij;

                    if (grad_flags[j]) {
                        // cov(df_j, f_i) = cov(f_i, df_j)
                        for (long k = 0, kj = grad_flags[j]; k < dim; ++k, kj += n_grad) {
                            double &k_i_kj = mat_k_kj_ptrs[k][i];  // mat_k(i, kj)
                            k_i_kj = l2_inv * diff_ij[k] * k_ij;   // cov(f_i, df_j/dx_k)
                            mat_k_i_ptr[kj] = k_i_kj;              // mat_k(kj, i) = cov(df_j/dx_k, f_i)
                        }

                        if (grad_flags[i]) {
                            for (long k = 0, ki = grad_flags[i]; k < dim; ++k, ki += n_grad) { mat_k_ki_ptrs[k] = mat_k.col(ki).data(); }

                            for (long k = 0, ki = grad_flags[i], kj = grad_flags[j]; k < dim; ++k, ki += n_grad, kj += n_grad) {
                                double &k_ki_j = mat_k_j_ptr[ki];  // mat_k(ki, j), use reference to improve performance
                                k_ki_j = -mat_k_kj_ptrs[k][i];     // cov(df_i, f_j) = -cov(f_i, df_j) = cov(f_j, df_i)
                                mat_k_ki_ptrs[k][j] = k_ki_j;      // mat_k(j, ki) = cov(f_j, df_i) = -cov(df_j, f_i) = -cov(f_i, df_j)

                                // cov(df_j, df_i) = cov(df_i, df_j)
                                // between Dim-k and Dim-k
                                const double &dxk = diff_ij[k];                        // use reference to improve performance
                                double &k_kj_ki = mat_k_ki_ptrs[k][kj];                // mat_k(kj, ki)
                                k_kj_ki = l2_inv * k_ij * (1.0 - l2_inv * dxk * dxk);  // cov(df_j/dx_k, df_i/dx_k)
                                mat_k_kj_ptrs[k][ki] = k_kj_ki;                        // cov(df_i/dx_k, df_j/dx_k)
                                for (long l = k + 1, li = ki + n_grad, lj = kj + n_grad; l < dim; ++l, li += n_grad, lj += n_grad) {
                                    // between Dim-k and Dim-l
                                    const double &dxl = diff_ij[l];
                                    double &k_kj_li = mat_k_ki_ptrs[l][kj];           // mat_k(kj, li)
                                    k_kj_li = l2_inv * k_ij * (-l2_inv * dxk * dxl);  // cov(df_j/dx_k, df_i/dx_l)
                                    mat_k_ki_ptrs[k][lj] = k_kj_li;                   // mat_k(lj, ki) = cov(df_j/dx_l, df_i/dx_k)
                                    mat_k_kj_ptrs[k][li] = k_kj_li;                   // mat_k(li, kj) = cov(df_i/dx_l, df_j/dx_k)
                                    mat_k_kj_ptrs[l][ki] = k_kj_li;                   // mat_k(ki, lj) = cov(df_i/dx_k, df_j/dx_l)
                                }
                            }
                        }
                    } else if (grad_flags[i]) {
                        // cov(f_j, df_i) = cov(df_i, f_j)
                        for (long k = 0, ki = grad_flags[i]; k < dim; ++k, ki += n_grad) {
                            double &k_ki_j = mat_k_j_ptr[ki];      // mat_k(ki, j), use reference to improve performance
                            k_ki_j = -l2_inv * diff_ij[k] * k_ij;  // cov(f_j, df_i)
                            mat_k(j, ki) = k_ki_j;
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
            Eigen::MatrixXd &mat_k,
            Eigen::VectorXd & /*vec_alpha*/) const override {

            long dim;
            if constexpr (Dim == Eigen::Dynamic) {
                dim = mat_x.rows();
            } else {
                dim = Dim;
            }

            ERL_DEBUG_ASSERT(mat_x.rows() == dim, "Each column of mat_x should be {}-D vector.", dim);
            long n_grad = 0;
            long *grad_flags = vec_grad_flags.data();
            for (long i = 0; i < num_samples; ++i) {
                if (long &flag = grad_flags[i]; flag > 0) { flag = num_samples + n_grad++; }
            }
            long n_rows = num_samples + n_grad * dim;
            long n_cols = n_rows;
            ERL_DEBUG_ASSERT(mat_k.rows() >= n_rows, "mat_k.rows() = {}, it should be >= {}.", mat_k.rows(), n_rows);
            ERL_DEBUG_ASSERT(mat_k.cols() >= n_cols, "mat_k.cols() = {}, it should be >= {}.", mat_k.cols(), n_cols);

            const double alpha = m_setting_->alpha;
            const double l2_inv = 1.0 / (m_setting_->scale * m_setting_->scale);
            const double a = 0.5 * l2_inv;
            // buffer to store the difference between x1_i and x2_j
            Eigen::Vector<double, Dim> diff_ij;          // avoid memory allocation on the heap
            Eigen::Vector<double *, Dim> mat_k_kj_ptrs;  // avoid memory allocation on the heap
            Eigen::Vector<double *, Dim> mat_k_ki_ptrs;  // avoid memory allocation on the heap
            if constexpr (Dim == Eigen::Dynamic) {
                diff_ij.resize(dim);
                mat_k_kj_ptrs.resize(dim);
                mat_k_ki_ptrs.resize(dim);
            }
            for (long j = 0; j < num_samples; ++j) {
                double *mat_k_j_ptr = mat_k.col(j).data();
                mat_k_j_ptr[j] = alpha + vec_var_x[j] + vec_var_y[j];  // mat_k(j, j)
                if (grad_flags[j]) {
                    for (long k = 0, kj = grad_flags[j]; k < dim; ++k, kj += n_grad) { mat_k_kj_ptrs[k] = mat_k.col(kj).data(); }

                    for (long k = 0, kj = grad_flags[j]; k < dim; ++k, kj += n_grad) {
                        mat_k_kj_ptrs[k][kj] = l2_inv + vec_var_grad[j];  // mat_k(kj, kj) = cov(df_j/dx_k, f_j)
                        mat_k_kj_ptrs[k][j] = 0.;                         // mat_k(j, kj) = cov(df_j/dx_k, f_j)
                        mat_k_j_ptr[kj] = 0.;                             // mat_k(kj, j) = cov(df_j/dx_k, f_j)
                        for (long l = k + 1, lj = kj + n_grad; l < dim; ++l, lj += n_grad) {
                            mat_k_kj_ptrs[l][kj] = 0.;  // mat_k(kj, lj) = cov(df_j/dx_k, df_j/dx_l)
                            mat_k_kj_ptrs[k][lj] = 0.;  // mat_k(lj, kj) = cov(df_j/dx_l, df_j/dx_k)
                        }
                    }
                }

                const double *xj_ptr = mat_x.col(j).data();
                for (long i = j + 1; i < num_samples; ++i) {
                    const double *xi_ptr = mat_x.col(i).data();
                    double *mat_k_i_ptr = mat_k.col(i).data();
                    double r2 = 0;
                    for (long k = 0; k < dim; ++k) {
                        double &dx = diff_ij[k];
                        dx = xi_ptr[k] - xj_ptr[k];
                        r2 += dx * dx;
                    }
                    double &k_ij = mat_k_j_ptr[i];     // mat_k(i, j)
                    k_ij = alpha * std::exp(-a * r2);  // cov(f_i, f_j)
                    mat_k_i_ptr[j] = k_ij;             // mat_k(j, i)

                    if (grad_flags[j]) {
                        // cov(df_j, f_i) = cov(f_i, df_j)
                        for (long k = 0, kj = grad_flags[j]; k < dim; ++k, kj += n_grad) {
                            double &k_i_kj = mat_k_kj_ptrs[k][i];  // mat_k(i, kj)
                            k_i_kj = l2_inv * diff_ij[k] * k_ij;   // cov(f_i, df_j/dx_k)
                            mat_k_i_ptr[kj] = k_i_kj;              // mat_k(kj, i) = cov(df_j/dx_k, f_i)
                        }

                        if (grad_flags[i]) {
                            for (long k = 0, ki = grad_flags[i]; k < dim; ++k, ki += n_grad) { mat_k_ki_ptrs[k] = mat_k.col(ki).data(); }

                            for (long k = 0, ki = grad_flags[i], kj = grad_flags[j]; k < dim; ++k, ki += n_grad, kj += n_grad) {
                                double &k_ki_j = mat_k_j_ptr[ki];  // mat_k(ki, j), use reference to improve performance
                                k_ki_j = -mat_k_kj_ptrs[k][i];     // cov(df_i, f_j) = -cov(f_i, df_j) = cov(f_j, df_i)
                                mat_k_ki_ptrs[k][j] = k_ki_j;      // mat_k(j, ki) = cov(f_j, df_i) = -cov(df_j, f_i) = -cov(f_i, df_j)

                                // cov(df_j, df_i) = cov(df_i, df_j)
                                // between Dim-k and Dim-k
                                const double &dxk = diff_ij[k];                        // use reference to improve performance
                                double &k_kj_ki = mat_k_ki_ptrs[k][kj];                // mat_k(kj, ki)
                                k_kj_ki = l2_inv * k_ij * (1.0 - l2_inv * dxk * dxk);  // cov(df_j/dx_k, df_i/dx_k)
                                mat_k_kj_ptrs[k][ki] = k_kj_ki;                        // cov(df_i/dx_k, df_j/dx_k)
                                for (long l = k + 1, li = ki + n_grad, lj = kj + n_grad; l < dim; ++l, li += n_grad, lj += n_grad) {
                                    // between Dim-k and Dim-l
                                    const double &dxl = diff_ij[l];
                                    double &k_kj_li = mat_k_ki_ptrs[l][kj];           // mat_k(kj, li)
                                    k_kj_li = l2_inv * k_ij * (-l2_inv * dxk * dxl);  // cov(df_j/dx_k, df_i/dx_l)
                                    mat_k_ki_ptrs[k][lj] = k_kj_li;                   // mat_k(lj, ki) = cov(df_j/dx_l, df_i/dx_k)
                                    mat_k_kj_ptrs[k][li] = k_kj_li;                   // mat_k(li, kj) = cov(df_i/dx_l, df_j/dx_k)
                                    mat_k_kj_ptrs[l][ki] = k_kj_li;                   // mat_k(ki, lj) = cov(df_i/dx_k, df_j/dx_l)
                                }
                            }
                        }
                    } else if (grad_flags[i]) {
                        // cov(f_j, df_i) = cov(df_i, f_j)
                        for (long k = 0, ki = grad_flags[i]; k < dim; ++k, ki += n_grad) {
                            double &k_ki_j = mat_k_j_ptr[ki];      // mat_k(ki, j), use reference to improve performance
                            k_ki_j = -l2_inv * diff_ij[k] * k_ij;  // cov(f_j, df_i)
                            mat_k(j, ki) = k_ki_j;
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
            Eigen::MatrixXd &mat_k) const override {

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
            ERL_DEBUG_ASSERT(mat_k.rows() >= n_rows, "mat_k.rows() = {}, it should be >= {}.", mat_k.rows(), n_rows);
            ERL_DEBUG_ASSERT(mat_k.cols() >= n_cols, "mat_k.cols() = {}, it should be >= {}.", mat_k.cols(), n_cols);

            const double alpha = m_setting_->alpha;
            const double l2_inv = 1.0 / (m_setting_->scale * m_setting_->scale);
            const double a = 0.5 * l2_inv;
            // buffer to store the difference between x1_i and x2_j
            Eigen::Vector<double, Dim> diff_ij;          // avoid memory allocation on the heap
            Eigen::Vector<double *, Dim> mat_k_kj_ptrs;  // avoid memory allocation on the heap
            if constexpr (Dim == Eigen::Dynamic) {
                diff_ij.resize(dim);
                mat_k_kj_ptrs.resize(dim);
            }
            for (long j = 0; j < num_samples2; ++j) {
                const double *x2_j_ptr = mat_x2.col(j).data();
                double *mat_k_j_ptr = mat_k.col(j).data();
                for (long k = 0, kj = j + num_samples2; k < dim; ++k, kj += num_samples2) { mat_k_kj_ptrs[k] = mat_k.col(kj).data(); }

                for (long i = 0, ki_init = num_samples1; i < num_samples1; ++i) {
                    const double *x1_i_ptr = mat_x1.col(i).data();
                    double r2 = 0;
                    for (long k = 0; k < dim; ++k) {
                        double &dx = diff_ij[k];
                        dx = x1_i_ptr[k] - x2_j_ptr[k];
                        r2 += dx * dx;
                    }
                    double &k_ij = mat_k_j_ptr[i];                                               // mat_k(i, j)
                    k_ij = alpha * std::exp(-a * r2);                                            // cov(f1_i, f2_j)
                    for (long k = 0, kj = j + num_samples2; k < dim; ++k, kj += num_samples2) {  // cov(f1_i, df2_j)
                        mat_k_kj_ptrs[k][i] = l2_inv * k_ij * diff_ij[k];
                    }

                    if (!vec_grad1_flags[i]) { continue; }
                    for (long k = 0, ki = ki_init, kj = j + num_samples2; k < dim; ++k, ki += n_grad, kj += num_samples2) {
                        mat_k_j_ptr[ki] = -mat_k_kj_ptrs[k][i];

                        // between Dim-k and Dim-k
                        const double &dxk = diff_ij[k];
                        // mat_k(ki, kj) = cov(df1_i/dx_k, df2_j/dx_k)
                        mat_k_kj_ptrs[k][ki] = l2_inv * k_ij * (1.0 - l2_inv * dxk * dxk);

                        for (long l = k + 1, li = ki + n_grad, lj = kj + num_samples2; l < dim; ++l, li += n_grad, lj += num_samples2) {
                            // between Dim-k and Dim-l
                            const double &dxl = diff_ij[l];
                            double &k_ki_lj = mat_k_kj_ptrs[l][li];
                            k_ki_lj = l2_inv * k_ij * (-l2_inv * dxk * dxl);  // cov(df1_i, df2_j)
                            mat_k_kj_ptrs[k][li] = k_ki_lj;
                        }
                    }
                    ++ki_init;
                }
            }
            return {n_rows, n_cols};
        }
    };

    using RadialBiasFunction1D = RadialBiasFunction<1>;
    using RadialBiasFunction2D = RadialBiasFunction<2>;
    using RadialBiasFunction3D = RadialBiasFunction<3>;
    using RadialBiasFunctionXd = RadialBiasFunction<Eigen::Dynamic>;

    ERL_REGISTER_COVARIANCE(RadialBiasFunction1D);
    ERL_REGISTER_COVARIANCE(RadialBiasFunction2D);
    ERL_REGISTER_COVARIANCE(RadialBiasFunction3D);
    ERL_REGISTER_COVARIANCE(RadialBiasFunctionXd);
}  // namespace erl::covariance
