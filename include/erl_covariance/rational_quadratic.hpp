#pragma once

#include "covariance.hpp"

namespace erl::covariance {
    static inline double
    InlineRq(double a, double b, double squared_norm) {
        return std::pow(1 + squared_norm * a, -b);
    }

    template<long Dim>
    class RationalQuadratic : public Covariance {
        // ref: https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.RationalQuadratic.html
    public:
        std::shared_ptr<Covariance>
        Create() const override {
            return std::make_shared<RationalQuadratic>(std::make_shared<Setting>());
        }

        explicit RationalQuadratic(std::shared_ptr<Setting> setting)
            : Covariance(std::move(setting)) {
            ERL_DEBUG_ASSERT(Dim == Eigen::Dynamic || m_setting_->x_dim == Dim, "setting->x_dim should be {}.", Dim);
            ERL_WARN_ONCE_COND(Dim == Eigen::Dynamic, "Dim is Eigen::Dynamic, it may cause performance issue.");
        }

        [[nodiscard]] std::pair<long, long>
        ComputeKtrain(Eigen::Ref<Eigen::MatrixXd> k_mat, const Eigen::Ref<const Eigen::MatrixXd> &mat_x) const final {
            long n = mat_x.cols();
            ERL_DEBUG_ASSERT(k_mat.rows() >= n, "k_mat.rows() = {}, it should be >= {}.", k_mat.rows(), n);
            ERL_DEBUG_ASSERT(k_mat.cols() >= n, "k_mat.cols() = {}, it should be >= {}.", k_mat.cols(), n);
            long dim;
            if constexpr (Dim == Eigen::Dynamic) {
                dim = mat_x.rows();
            } else {
                dim = Dim;
            }
            const double a = 0.5 / (m_setting_->scale * m_setting_->scale * m_setting_->scale_mix);
            for (long i = 0; i < n; ++i) {
                for (long j = i; j < n; ++j) {
                    if (i == j) {
                        k_mat(i, i) = m_setting_->alpha;
                    } else {
                        double r = 0;  // (mat_x.col(i) - mat_x.col(j)).squaredNorm();
                        for (long k = 0; k < dim; ++k) {
                            const double dx = mat_x(k, i) - mat_x(k, j);
                            r += dx * dx;
                        }
                        k_mat(i, j) = m_setting_->alpha * InlineRq(a, m_setting_->scale_mix, r);
                        k_mat(j, i) = k_mat(i, j);
                    }
                }
            }
            return {n, n};
        }

        [[nodiscard]] std::pair<long, long>
        ComputeKtrain(Eigen::Ref<Eigen::MatrixXd> k_mat, const Eigen::Ref<const Eigen::MatrixXd> &mat_x, const Eigen::Ref<const Eigen::VectorXd> &vec_var_y)
            const final {
            long n = mat_x.cols();
            ERL_DEBUG_ASSERT(k_mat.rows() >= n, "k_mat.rows() = {}, it should be >= {}.", k_mat.rows(), n);
            ERL_DEBUG_ASSERT(k_mat.cols() >= n, "k_mat.cols() = {}, it should be >= {}.", k_mat.cols(), n);
            ERL_DEBUG_ASSERT(n == vec_var_y.size(), "#elements of vec_sigma_y does not equal to #columns of m_x_.");
            long dim;
            if constexpr (Dim == Eigen::Dynamic) {
                dim = mat_x.rows();
            } else {
                dim = Dim;
            }
            const double a = 0.5 / (m_setting_->scale * m_setting_->scale * m_setting_->scale_mix);
            for (long i = 0; i < n; ++i) {
                for (long j = i; j < n; ++j) {
                    if (i == j) {
                        k_mat(i, i) = m_setting_->alpha + vec_var_y[i];
                    } else {
                        double r = 0;  // (mat_x.col(i) - mat_x.col(j)).squaredNorm();
                        for (long k = 0; k < dim; ++k) {
                            const double dx = mat_x(k, i) - mat_x(k, j);
                            r += dx * dx;
                        }
                        k_mat(i, j) = m_setting_->alpha * InlineRq(a, m_setting_->scale_mix, r);
                        k_mat(j, i) = k_mat(i, j);
                    }
                }
            }
            return {n, n};
        }

        [[nodiscard]] std::pair<long, long>
        ComputeKtest(Eigen::Ref<Eigen::MatrixXd> k_mat, const Eigen::Ref<const Eigen::MatrixXd> &mat_x1, const Eigen::Ref<const Eigen::MatrixXd> &mat_x2)
            const final {
            ERL_DEBUG_ASSERT(mat_x1.rows() == mat_x2.rows(), "Sample vectors stored in x_1 and x_2 should have the same dimension.");
            long n = mat_x1.cols();
            long m = mat_x2.cols();
            ERL_DEBUG_ASSERT(k_mat.rows() >= n, "k_mat.rows() = {}, it should be >= {}.", k_mat.rows(), n);
            ERL_DEBUG_ASSERT(k_mat.cols() >= m, "k_mat.cols() = {}, it should be >= {}.", k_mat.cols(), m);

            const double a = 0.5 / (m_setting_->scale * m_setting_->scale * m_setting_->scale_mix);
            for (long i = 0; i < n; ++i) {
                for (long j = 0; j < m; ++j) {
                    double r = 0;  // (mat_x1.col(i) - mat_x2.col(j)).squaredNorm();
                    for (long k = 0; k < mat_x1.rows(); ++k) {
                        const double dx = mat_x1(k, i) - mat_x2(k, j);
                        r += dx * dx;
                    }
                    k_mat(i, j) = m_setting_->alpha * InlineRq(a, m_setting_->scale_mix, r);
                }
            }
            return {n, m};
        }

        [[nodiscard]] std::pair<long, long>
        ComputeKtrainWithGradient(
            Eigen::Ref<Eigen::MatrixXd> k_mat,
            const Eigen::Ref<const Eigen::MatrixXd> &mat_x,
            const Eigen::Ref<const Eigen::VectorXb> &vec_grad_flags) const final {

            long dim;
            if constexpr (Dim == Eigen::Dynamic) {
                dim = mat_x.rows();
            } else {
                dim = Dim;
            }

            ERL_DEBUG_ASSERT(mat_x.rows() == dim, "Each column of mat_x should be {}-D vector.", dim);
            const long n = mat_x.cols();
            std::vector<long> grad_indices;
            grad_indices.reserve(vec_grad_flags.size());
            long n_grad = 0;
            for (const bool &flag: vec_grad_flags) {
                if (flag) {
                    grad_indices.push_back(n + n_grad++);
                } else {
                    grad_indices.push_back(-1);
                }
            }
            long n_rows = n + n_grad * dim;
            long n_cols = n_rows;
            ERL_DEBUG_ASSERT(k_mat.rows() >= n_rows, "k_mat.rows() = {}, it should be >= {}.", k_mat.rows(), n_rows);
            ERL_DEBUG_ASSERT(k_mat.cols() >= n_cols, "k_mat.cols() = {}, it should be >= {}.", k_mat.cols(), n_cols);

            const double l2_inv = 1. / (m_setting_->scale * m_setting_->scale);
            const double a = 0.5 * l2_inv / m_setting_->scale_mix;
            for (long i = 0; i < n; ++i) {
                k_mat(i, i) = m_setting_->alpha;  // cov(f_i, f_i)
                if (vec_grad_flags[i]) {
                    for (long k = 0, ki = grad_indices[i]; k < dim; ++k, ki += n_grad) {
                        // cov(df_i, df_i)
                        k_mat(ki, ki) = l2_inv;
                        // cov(f_i, df_i) and cov(df_i, f_i) are zeros
                        k_mat(i, ki) = 0.;
                        k_mat(ki, i) = 0.;
                        for (long l = k + 1, li = ki + n_grad; l < dim; ++l, li += n_grad) {
                            k_mat(ki, li) = 0.;  // cov(df_i/dx_k, df_i/dx_l)
                            k_mat(li, ki) = 0.;  // cov(df_i/dx_l, df_i/dx_k)
                        }
                    }
                }

                for (long j = i + 1; j < n; ++j) {
                    double r2 = 0;  // (mat_x.col(i) - mat_x.col(j)).squaredNorm()
                    for (long k = 0; k < dim; ++k) {
                        const double dx = mat_x(k, i) - mat_x(k, j);
                        r2 += dx * dx;
                    }

                    k_mat(i, j) = m_setting_->alpha * InlineRq(a, m_setting_->scale_mix, r2);
                    k_mat(j, i) = k_mat(i, j);

                    if (const double beta = 1. / (1. + a * r2), gamma = beta * beta * l2_inv * (1. + m_setting_->scale_mix) / m_setting_->scale_mix;
                        vec_grad_flags[i]) {

                        // cov(f_j, df_i) = cov(df_i, f_j)
                        for (long k = 0, ki = grad_indices[i]; k < dim; ++k, ki += n_grad) {
                            k_mat(j, ki) = beta * l2_inv * k_mat(j, i) * (mat_x(k, j) - mat_x(k, i));  // cov(f_j, df_i/dx_k)
                            k_mat(ki, j) = k_mat(j, ki);                                               // cov(df_i/dx_k, f_j)
                        }

                        if (vec_grad_flags[j]) {
                            for (long k = 0, ki = grad_indices[i], kj = grad_indices[j]; k < dim; ++k, ki += n_grad, kj += n_grad) {
                                k_mat(i, kj) = -k_mat(j, ki);  // cov(f_i, df_j/dx_k) = -cov(df_i/dx_k, f_j)
                                k_mat(kj, i) = k_mat(i, kj);   // cov(df_j/dx_k, f_i) = -cov(f_j, df_i/dx_k) = cov(f_i, df_j/dx_k)
                            }

                            // cov(df_i, df_j) = cov(df_j, df_i)
                            for (long k = 0, ki = grad_indices[i], kj = grad_indices[j]; k < dim; ++k, ki += n_grad, kj += n_grad) {
                                // between Dim-k and Dim-k
                                const double dxk = mat_x(k, i) - mat_x(k, j);
                                k_mat(ki, kj) = l2_inv * k_mat(i, j) * (beta - gamma * dxk * dxk);  // cov(df_i/dx_k, df_j/dx_k)
                                k_mat(kj, ki) = k_mat(ki, kj);                                      // cov(df_j/dx_k, df_i/dx_k)
                                for (long l = k + 1, li = ki + n_grad, lj = kj + n_grad; l < dim; ++l, li += n_grad, lj += n_grad) {
                                    // between Dim-k and Dim-l
                                    const double dxl = mat_x(l, i) - mat_x(l, j);
                                    k_mat(ki, lj) = l2_inv * k_mat(i, j) * (-gamma * dxk * dxl);  // cov(df_i/dx_k, df_j/dx_l)
                                    k_mat(li, kj) = k_mat(ki, lj);                                // cov(df_i/dx_l, df_j/dx_k)
                                    k_mat(lj, ki) = k_mat(ki, lj);                                // cov(df_j/dx_l, df_i/dx_k)
                                    k_mat(kj, li) = k_mat(lj, ki);                                // cov(df_j/dx_k, df_i/dx_l)
                                }
                            }
                        }
                    } else if (vec_grad_flags[j]) {
                        // cov(f_i, df_j) = cov(df_j, f_i)
                        for (long k = 0, kj = grad_indices[j]; k < dim; ++k, kj += n_grad) {
                            k_mat(i, kj) = beta * l2_inv * k_mat(i, j) * (mat_x(k, i) - mat_x(k, j));  // cov(f_i, df_j/dx_k)
                            k_mat(kj, i) = k_mat(i, kj);
                        }
                    }
                }  // for (long j = i + 1; j < n; ++j)
            }      // for (long i = 0; i < n; ++i)
            return {n_rows, n_cols};
        }

        [[nodiscard]] std::pair<long, long>
        ComputeKtrainWithGradient(
            Eigen::Ref<Eigen::MatrixXd> k_mat,
            const Eigen::Ref<const Eigen::MatrixXd> &mat_x,
            const Eigen::Ref<const Eigen::VectorXb> &vec_grad_flags,
            const Eigen::Ref<const Eigen::VectorXd> &vec_var_x,
            const Eigen::Ref<const Eigen::VectorXd> &vec_var_y,
            const Eigen::Ref<const Eigen::VectorXd> &vec_var_grad) const final {

            long dim;
            if constexpr (Dim == Eigen::Dynamic) {
                dim = mat_x.rows();
            } else {
                dim = Dim;
            }

            ERL_DEBUG_ASSERT(mat_x.rows() == dim, "Each column of mat_x should be {}-D vector.", dim);
            const long n = mat_x.cols();
            std::vector<long> grad_indices;
            grad_indices.reserve(vec_grad_flags.size());
            long n_grad = 0;
            for (const bool &flag: vec_grad_flags) {
                if (flag) {
                    grad_indices.push_back(n + n_grad++);
                } else {
                    grad_indices.push_back(-1);
                }
            }
            long n_rows = n + n_grad * dim;
            long n_cols = n_rows;
            ERL_DEBUG_ASSERT(k_mat.rows() >= n_rows, "k_mat.rows() = {}, it should be >= {}.", k_mat.rows(), n_rows);
            ERL_DEBUG_ASSERT(k_mat.cols() >= n_cols, "k_mat.cols() = {}, it should be >= {}.", k_mat.cols(), n_cols);

            const double l2_inv = 1. / (m_setting_->scale * m_setting_->scale);
            const double a = 0.5 * l2_inv / m_setting_->scale_mix;
            for (long i = 0; i < n; ++i) {
                k_mat(i, i) = m_setting_->alpha + vec_var_x(i) + vec_var_y(i);  // cov(f_i, f_i)
                if (vec_grad_flags[i]) {
                    for (long k = 0, ki = grad_indices[i]; k < dim; ++k, ki += n_grad) {
                        // cov(df_i, df_i)
                        k_mat(ki, ki) = l2_inv + vec_var_grad(i);
                        // cov(f_i, df_i) and cov(df_i, f_i) are zeros
                        k_mat(i, ki) = 0.;
                        k_mat(ki, i) = 0.;
                        for (long l = k + 1, li = ki + n_grad; l < dim; ++l, li += n_grad) {
                            k_mat(ki, li) = 0.;  // cov(df_i/dx_k, df_i/dx_l)
                            k_mat(li, ki) = 0.;  // cov(df_i/dx_l, df_i/dx_k)
                        }
                    }
                }

                for (long j = i + 1; j < n; ++j) {
                    double r2 = 0;  // (mat_x.col(i) - mat_x.col(j)).squaredNorm()
                    for (long k = 0; k < dim; ++k) {
                        const double dx = mat_x(k, i) - mat_x(k, j);
                        r2 += dx * dx;
                    }

                    k_mat(i, j) = m_setting_->alpha * InlineRq(a, m_setting_->scale_mix, r2);
                    k_mat(j, i) = k_mat(i, j);

                    if (const double beta = 1. / (1. + a * r2), gamma = beta * beta * l2_inv * (1. + m_setting_->scale_mix) / m_setting_->scale_mix;
                        vec_grad_flags[i]) {
                        // cov(f_j, df_i) = cov(df_i, f_j)
                        for (long k = 0, ki = grad_indices[i]; k < dim; ++k, ki += n_grad) {
                            k_mat(j, ki) = beta * l2_inv * k_mat(j, i) * (mat_x(k, j) - mat_x(k, i));  // cov(f_j, df_i/dx_k)
                            k_mat(ki, j) = k_mat(j, ki);                                               // cov(df_i/dx_k, f_j)
                        }

                        if (vec_grad_flags[j]) {
                            for (long k = 0, ki = grad_indices[i], kj = grad_indices[j]; k < dim; ++k, ki += n_grad, kj += n_grad) {
                                k_mat(i, kj) = -k_mat(j, ki);  // cov(f_i, df_j/dx_k) = -cov(df_i/dx_k, f_j)
                                k_mat(kj, i) = k_mat(i, kj);   // cov(df_j/dx_k, f_i) = -cov(f_j, df_i/dx_k) = cov(f_i, df_j/dx_k)
                            }

                            // cov(df_i, df_j) = cov(df_j, df_i)
                            for (long k = 0, ki = grad_indices[i], kj = grad_indices[j]; k < dim; ++k, ki += n_grad, kj += n_grad) {
                                // between Dim-k and Dim-k
                                const double dxk = mat_x(k, i) - mat_x(k, j);
                                k_mat(ki, kj) = l2_inv * k_mat(i, j) * (beta - gamma * dxk * dxk);  // cov(df_i/dx_k, df_j/dx_k)
                                k_mat(kj, ki) = k_mat(ki, kj);                                      // cov(df_j/dx_k, df_i/dx_k)
                                for (long l = k + 1, li = ki + n_grad, lj = kj + n_grad; l < dim; ++l, li += n_grad, lj += n_grad) {
                                    // between Dim-k and Dim-l
                                    const double dxl = mat_x(l, i) - mat_x(l, j);
                                    k_mat(ki, lj) = l2_inv * k_mat(i, j) * (-gamma * dxk * dxl);  // cov(df_i/dx_k, df_j/dx_l)
                                    k_mat(li, kj) = k_mat(ki, lj);                                // cov(df_i/dx_l, df_j/dx_k)
                                    k_mat(lj, ki) = k_mat(ki, lj);                                // cov(df_j/dx_l, df_i/dx_k)
                                    k_mat(kj, li) = k_mat(lj, ki);                                // cov(df_j/dx_k, df_i/dx_l)
                                }
                            }
                        }
                    } else if (vec_grad_flags[j]) {
                        // cov(f_i, df_j) = cov(df_j, f_i)
                        for (long k = 0, kj = grad_indices[j]; k < dim; ++k, kj += n_grad) {
                            k_mat(i, kj) = beta * l2_inv * k_mat(i, j) * (mat_x(k, i) - mat_x(k, j));  // cov(f_i, df_j/dx_k)
                            k_mat(kj, i) = k_mat(i, kj);
                        }
                    }
                }  // for (long j = i + 1; j < n; ++j)
            }      // for (long i = 0; i < n; ++i)
            return {n_rows, n_cols};
        }

        [[nodiscard]] std::pair<long, long>
        ComputeKtestWithGradient(
            Eigen::Ref<Eigen::MatrixXd> k_mat,
            const Eigen::Ref<const Eigen::MatrixXd> &mat_x1,
            const Eigen::Ref<const Eigen::VectorXb> &vec_grad1_flags,
            const Eigen::Ref<const Eigen::MatrixXd> &mat_x2) const final {

            long dim;
            if constexpr (Dim == Eigen::Dynamic) {
                dim = mat_x1.rows();
            } else {
                dim = Dim;
            }

            ERL_DEBUG_ASSERT(dim == mat_x2.rows(), "Sample vectors stored in x_1 and x_2 should have the same dimension.");
            const long n = mat_x1.cols();
            const long m = mat_x2.cols();
            std::vector<long> grad_indices;
            grad_indices.reserve(vec_grad1_flags.size());
            long n_grad = 0;
            for (const bool &flag: vec_grad1_flags) {
                if (flag) {
                    grad_indices.push_back(n + n_grad++);
                } else {
                    grad_indices.push_back(-1);
                }
            }
            long n_rows = n + n_grad * dim;
            long n_cols = m * (dim + 1);
            ERL_DEBUG_ASSERT(k_mat.rows() >= n_rows, "k_mat.rows() = {}, it should be >= {}.", k_mat.rows(), n_rows);
            ERL_DEBUG_ASSERT(k_mat.cols() >= n_cols, "k_mat.cols() = {}, it should be >= {}.", k_mat.cols(), n_cols);

            const double l2_inv = 1. / (m_setting_->scale * m_setting_->scale);
            const double a = 0.5 * l2_inv / m_setting_->scale_mix;
            for (long i = 0; i < n; ++i) {
                for (long j = 0; j < m; ++j) {
                    const double r2 = (mat_x1.col(i) - mat_x2.col(j)).norm();
                    const double beta = 1. / (1. + a * r2);

                    k_mat(i, j) = m_setting_->alpha * InlineRq(a, m_setting_->scale_mix, r2);  // cov(f1_i, f2_j)
                    for (long k = 0, kj = j + m; k < dim; ++k, kj += m) {                      // cov(f1_i, df2_j/dx_k)
                        k_mat(i, kj) = beta * l2_inv * k_mat(i, j) * (mat_x1(k, i) - mat_x2(k, j));
                    }

                    if (vec_grad1_flags[i]) {
                        const double gamma = beta * beta * l2_inv * (1. + m_setting_->scale_mix) / m_setting_->scale_mix;
                        for (long k = 0, ki = grad_indices[i], kj = j + m; k < dim; ++k, ki += n_grad, kj += m) {
                            k_mat(ki, j) = -k_mat(i, kj);  // cov(df1_i/dx_k, f2_j) = -cov(f1_i, df2_j/dx_k)

                            // cov(df1_i, df2_j)
                            // between Dim-k and Dim-k
                            const double dxk = mat_x1(k, i) - mat_x2(k, j);
                            k_mat(ki, kj) = l2_inv * k_mat(i, j) * (beta - gamma * dxk * dxk);

                            for (long l = k + 1, li = ki + n_grad, lj = kj + m; l < dim; ++l, li += n_grad, lj += m) {
                                // between Dim-k and Dim-l
                                const double dxl = mat_x1(l, i) - mat_x2(l, j);
                                k_mat(ki, lj) = l2_inv * k_mat(i, j) * (-gamma * dxk * dxl);
                                k_mat(li, kj) = k_mat(ki, lj);
                            }
                        }
                    }
                }
            }
            return {n_rows, n_cols};
        }
    };

    using RationalQuadratic1D = RationalQuadratic<1>;
    using RationalQuadratic2D = RationalQuadratic<2>;
    using RationalQuadratic3D = RationalQuadratic<3>;
    using RationalQuadraticXd = RationalQuadratic<Eigen::Dynamic>;

    ERL_REGISTER_COVARIANCE(RationalQuadratic1D);
    ERL_REGISTER_COVARIANCE(RationalQuadratic2D);
    ERL_REGISTER_COVARIANCE(RationalQuadratic3D);
    ERL_REGISTER_COVARIANCE(RationalQuadraticXd);
}  // namespace erl::covariance
