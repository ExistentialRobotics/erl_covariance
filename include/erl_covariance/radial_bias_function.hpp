#pragma once

#include "covariance.hpp"

namespace erl::covariance {

    template<long Dim>
    class RadialBiasFunction : public Covariance {
        // ref: https://en.wikipedia.org/wiki/Radial_basis_function_kernel
    public:
        static std::shared_ptr<RadialBiasFunction>
        Create(std::shared_ptr<Setting> setting = nullptr) {
            if (setting == nullptr) {
                setting = std::make_shared<RadialBiasFunction::Setting>();
                setting->type = Type::kRadialBiasFunction;
            }
            return std::shared_ptr<RadialBiasFunction>(new RadialBiasFunction(std::move(setting)));
        }

        [[nodiscard]] std::pair<long, long>
        ComputeKtrain(Eigen::Ref<Eigen::MatrixXd> k_mat, const Eigen::Ref<const Eigen::MatrixXd> &mat_x) const final {
            long n = mat_x.cols();
            ERL_ASSERTM(k_mat.rows() >= n, "k_mat.rows() = %ld, it should be >= %ld.", k_mat.rows(), n);
            ERL_ASSERTM(k_mat.cols() >= n, "k_mat.cols() = %ld, it should be >= %ld.", k_mat.cols(), n);
            long dim;
            if constexpr (Dim == Eigen::Dynamic) {
                dim = mat_x.rows();
            } else {
                dim = Dim;
            }
            double a = 0.5 / (m_setting_->scale * m_setting_->scale);
            for (long i = 0; i < n; ++i) {
                for (long j = i; j < n; ++j) {
                    if (i == j) {
                        k_mat(i, i) = m_setting_->alpha;
                    } else {
                        double r = 0;  // (mat_x.col(i) - mat_x.col(j)).squaredNorm()
                        for (long k = 0; k < dim; ++k) {
                            double dx = mat_x(k, i) - mat_x(k, j);
                            r += dx * dx;
                        }
                        k_mat(i, j) = m_setting_->alpha * std::exp(-a * r);
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
            ERL_ASSERTM(n == vec_var_y.size(), "#elements of vec_sigma_y does not equal to #columns of m_x_.");
            ERL_ASSERTM(k_mat.rows() >= n, "k_mat.rows() = %ld, it should be >= %ld.", k_mat.rows(), n);
            ERL_ASSERTM(k_mat.cols() >= n, "k_mat.cols() = %ld, it should be >= %ld.", k_mat.cols(), n);
            long dim;
            if constexpr (Dim == Eigen::Dynamic) {
                dim = mat_x.rows();
            } else {
                dim = Dim;
            }
            double a = 0.5 / (m_setting_->scale * m_setting_->scale);
            for (long i = 0; i < n; ++i) {
                for (long j = i; j < n; ++j) {
                    if (i == j) {
                        k_mat(i, i) = m_setting_->alpha + vec_var_y[i];
                    } else {
                        double r = 0;  // (mat_x.col(i) - mat_x.col(j)).squaredNorm()
                        for (long k = 0; k < dim; ++k) {
                            double dx = mat_x(k, i) - mat_x(k, j);
                            r += dx * dx;
                        }
                        k_mat(i, j) = m_setting_->alpha * std::exp(-a * r);
                        k_mat(j, i) = k_mat(i, j);
                    }
                }
            }
            return {n, n};
        }

        [[nodiscard]] std::pair<long, long>
        ComputeKtest(Eigen::Ref<Eigen::MatrixXd> k_mat, const Eigen::Ref<const Eigen::MatrixXd> &mat_x1, const Eigen::Ref<const Eigen::MatrixXd> &mat_x2)
            const final {
            ERL_ASSERTM(mat_x1.rows() == mat_x2.rows(), "Sample vectors stored in x_1 and x_2 should have the same dimension.");
            long n = mat_x1.cols();
            long m = mat_x2.cols();
            ERL_ASSERTM(k_mat.rows() >= n, "k_mat.rows() = %ld, it should be >= %ld.", k_mat.rows(), n);
            ERL_ASSERTM(k_mat.cols() >= m, "k_mat.cols() = %ld, it should be >= %ld.", k_mat.cols(), m);
            long dim;
            if constexpr (Dim == Eigen::Dynamic) {
                dim = mat_x1.rows();
            } else {
                dim = Dim;
            }
            double a = 0.5 / (m_setting_->scale * m_setting_->scale);
            for (long i = 0; i < n; ++i) {
                for (long j = 0; j < m; ++j) {
                    double r = 0;  // (mat_x1.col(i) - mat_x2.col(j)).squaredNorm()
                    for (long k = 0; k < dim; ++k) {
                        double dx = mat_x1(k, i) - mat_x2(k, j);
                        r += dx * dx;
                    }
                    k_mat(i, j) = m_setting_->alpha * std::exp(-a * r);
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

            ERL_ASSERTM(mat_x.rows() == dim, "Each column of mat_x should be %ld-D vector.", dim);
            long n = mat_x.cols();
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
            ERL_ASSERTM(k_mat.rows() >= n_rows, "k_mat.rows() = %ld, it should be >= %ld.", k_mat.rows(), n_rows);
            ERL_ASSERTM(k_mat.cols() >= n_cols, "k_mat.cols() = %ld, it should be >= %ld.", k_mat.cols(), n_cols);

            double l2_inv = 1.0 / (m_setting_->scale * m_setting_->scale);
            double a = 0.5 * l2_inv;
            for (long i = 0; i < n; ++i) {
                k_mat(i, i) = m_setting_->alpha;  // cov(f_i, f_i)
                // k_mat(i, i) = m_setting_->alpha + vec_sigma_x(i) + vec_sigma_y(i);  // cov(f_i, f_i)
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
                        double dx = mat_x(k, i) - mat_x(k, j);
                        r2 += dx * dx;
                    }
                    k_mat(i, j) = m_setting_->alpha * std::exp(-a * r2);  // cov(f_i, f_j)
                    k_mat(j, i) = k_mat(i, j);

                    if (vec_grad_flags[i]) {
                        // cov(f_j, df_i) = cov(df_i, f_j)
                        for (long k = 0, ki = grad_indices[i]; k < dim; ++k, ki += n_grad) {
                            k_mat(j, ki) = l2_inv * (mat_x(k, j) - mat_x(k, i)) * k_mat(i, j);  // cov(f_j, df_i)
                            k_mat(ki, j) = k_mat(j, ki);                                        // cov(df_i, f_j)
                        }

                        if (vec_grad_flags[j]) {
                            for (long k = 0, ki = grad_indices[i], kj = grad_indices[j]; k < dim; ++k, ki += n_grad, kj += n_grad) {
                                k_mat(i, kj) = -k_mat(j, ki);  // cov(f_i, df_j) = -cov(df_i, f_j)
                                k_mat(kj, i) = k_mat(i, kj);   // cov(df_j, f_i) = -cov(f_j, df_i) = cov(f_i, df_j)
                            }

                            // cov(df_i, df_j) = cov(df_j, df_i)
                            for (long k = 0, ki = grad_indices[i], kj = grad_indices[j]; k < dim; ++k, ki += n_grad, kj += n_grad) {
                                // between Dim-k and Dim-k
                                double dxk = mat_x(k, i) - mat_x(k, j);
                                k_mat(ki, kj) = l2_inv * k_mat(i, j) * (1. - l2_inv * dxk * dxk);  // cov(df_i, df_j)
                                k_mat(kj, ki) = k_mat(ki, kj);                                     // cov(df_j, df_i)
                                for (long l = k + 1, li = ki + n_grad, lj = kj + n_grad; l < dim; ++l, li += n_grad, lj += n_grad) {
                                    // between Dim-k and Dim-l
                                    double dxl = mat_x(l, i) - mat_x(l, j);
                                    k_mat(ki, lj) = l2_inv * k_mat(i, j) * (-l2_inv * dxk * dxl);  // cov(df_i, df_j)
                                    k_mat(li, kj) = k_mat(ki, lj);
                                    k_mat(lj, ki) = k_mat(ki, lj);  // cov(df_j, df_i)
                                    k_mat(kj, li) = k_mat(lj, ki);
                                }
                            }
                        }
                    } else if (vec_grad_flags[j]) {
                        // cov(f_i, df_j) = cov(df_j, f_i)
                        for (long k = 0, kj = grad_indices[j]; k < dim; ++k, kj += n_grad) {
                            k_mat(i, kj) = l2_inv * (mat_x(k, i) - mat_x(k, j)) * k_mat(i, j);  // cov(f_i, df_j
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

            ERL_ASSERTM(mat_x.rows() == dim, "Each column of mat_x should be %ld-D vector.", dim);
            long n = mat_x.cols();
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
            ERL_ASSERTM(k_mat.rows() >= n_rows, "k_mat.rows() = %ld, it should be >= %ld.", k_mat.rows(), n_rows);
            ERL_ASSERTM(k_mat.cols() >= n_cols, "k_mat.cols() = %ld, it should be >= %ld.", k_mat.cols(), n_cols);

            double l2_inv = 1.0 / (m_setting_->scale * m_setting_->scale);
            double a = 0.5 * l2_inv;
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
                        double dx = mat_x(k, i) - mat_x(k, j);
                        r2 += dx * dx;
                    }
                    k_mat(i, j) = m_setting_->alpha * std::exp(-a * r2);  // cov(f_i, f_j)
                    k_mat(j, i) = k_mat(i, j);

                    if (vec_grad_flags[i]) {
                        // cov(f_j, df_i) = cov(df_i, f_j)
                        for (long k = 0, ki = grad_indices[i]; k < dim; ++k, ki += n_grad) {
                            k_mat(j, ki) = l2_inv * (mat_x(k, j) - mat_x(k, i)) * k_mat(i, j);  // cov(f_j, df_i)
                            k_mat(ki, j) = k_mat(j, ki);                                        // cov(df_i, f_j)
                        }

                        if (vec_grad_flags[j]) {
                            for (long k = 0, ki = grad_indices[i], kj = grad_indices[j]; k < dim; ++k, ki += n_grad, kj += n_grad) {
                                k_mat(i, kj) = -k_mat(j, ki);  // cov(f_i, df_j) = -cov(df_i, f_j)
                                k_mat(kj, i) = k_mat(i, kj);   // cov(df_j, f_i) = -cov(f_j, df_i) = cov(f_i, df_j)
                            }

                            // cov(df_i, df_j) = cov(df_j, df_i)
                            for (long k = 0, ki = grad_indices[i], kj = grad_indices[j]; k < dim; ++k, ki += n_grad, kj += n_grad) {
                                // between Dim-k and Dim-k
                                double dxk = mat_x(k, i) - mat_x(k, j);
                                k_mat(ki, kj) = l2_inv * k_mat(i, j) * (1. - l2_inv * dxk * dxk);  // cov(df_i, df_j)
                                k_mat(kj, ki) = k_mat(ki, kj);                                     // cov(df_j, df_i)
                                for (long l = k + 1, li = ki + n_grad, lj = kj + n_grad; l < dim; ++l, li += n_grad, lj += n_grad) {
                                    // between Dim-k and Dim-l
                                    double dxl = mat_x(l, i) - mat_x(l, j);
                                    k_mat(ki, lj) = l2_inv * k_mat(i, j) * (-l2_inv * dxk * dxl);  // cov(df_i, df_j)
                                    k_mat(li, kj) = k_mat(ki, lj);
                                    k_mat(lj, ki) = k_mat(ki, lj);  // cov(df_j, df_i)
                                    k_mat(kj, li) = k_mat(lj, ki);
                                }
                            }
                        }
                    } else if (vec_grad_flags[j]) {
                        // cov(f_i, df_j) = cov(df_j, f_i)
                        for (long k = 0, kj = grad_indices[j]; k < dim; ++k, kj += n_grad) {
                            k_mat(i, kj) = l2_inv * (mat_x(k, i) - mat_x(k, j)) * k_mat(i, j);  // cov(f_i, df_j)
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

            ERL_ASSERTM(mat_x1.rows() == dim, "Each column of mat_x1 should be %ld-D vector.", dim);
            ERL_ASSERTM(mat_x2.rows() == dim, "Each column of mat_x2 should be %ld-D vector.", dim);
            long n = mat_x1.cols();
            long m = mat_x2.cols();
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
            ERL_ASSERTM(k_mat.rows() >= n_rows, "k_mat.rows() = %ld, it should be >= %ld.", k_mat.rows(), n_rows);
            ERL_ASSERTM(k_mat.cols() >= n_cols, "k_mat.cols() = %ld, it should be >= %ld.", k_mat.cols(), n_cols);

            double l2_inv = 1.0 / (m_setting_->scale * m_setting_->scale);
            double a = 0.5 * l2_inv;
            for (long i = 0; i < n; ++i) {
                for (long j = 0; j < m; ++j) {
                    double r2 = 0;  // (mat_x1.col(i) - mat_x2.col(j)).squaredNorm()
                    for (long k = 0; k < dim; ++k) {
                        double dx = mat_x1(k, i) - mat_x2(k, j);
                        r2 += dx * dx;
                    }

                    k_mat(i, j) = m_setting_->alpha * std::exp(-a * r2);   // cov(f1_i, f2_j)
                    for (long k = 0, kj = j + m; k < dim; ++k, kj += m) {  // cov(f1_i, df2_j)
                        k_mat(i, kj) = l2_inv * k_mat(i, j) * (mat_x1(k, i) - mat_x2(k, j));
                    }

                    if (vec_grad1_flags[i]) {
                        for (long k = 0, ki = grad_indices[i], kj = j + m; k < dim; ++k, ki += n_grad, kj += m) {
                            k_mat(ki, j) = -k_mat(i, kj);  // cov(df1_i, f2_j) = -cov(f1_i, df2_j)

                            // cov(df1_i, df2_j)
                            // between Dim-k and Dim-k
                            double dxk = mat_x1(k, i) - mat_x2(k, j);
                            k_mat(ki, kj) = l2_inv * k_mat(i, j) * (1. - l2_inv * dxk * dxk);  // cov(df1_i, df2_j)

                            for (long l = k + 1, li = ki + n_grad, lj = kj + m; l < dim; ++l, li += n_grad, lj += m) {
                                // between Dim-k and Dim-l
                                double dxl = mat_x1(l, i) - mat_x2(l, j);
                                k_mat(ki, lj) = l2_inv * k_mat(i, j) * (-l2_inv * dxk * dxl);  // cov(df1_i, df2_j)
                                k_mat(li, kj) = k_mat(ki, lj);
                            }
                        }
                    }
                }
            }
            return {n_rows, n_cols};
        }

    private:
        explicit RadialBiasFunction(std::shared_ptr<Setting> setting)
            : Covariance(std::move(setting)) {
            ERL_ASSERTM(m_setting_->type == Type::kRadialBiasFunction, "setting->type should be kRadialBiasFunction.");
            ERL_ASSERTM(Dim == Eigen::Dynamic || m_setting_->x_dim == Dim, "setting->x_dim should be %ld.", Dim);
            ERL_WARN_ONCE_COND(Dim == Eigen::Dynamic, "Dim is Eigen::Dynamic, it may cause performance issue.");
        }
    };
}  // namespace erl::covariance
