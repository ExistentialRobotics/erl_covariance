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
        static std::shared_ptr<RationalQuadratic>
        Create(std::shared_ptr<Setting> setting = nullptr) {
            if (setting == nullptr) {
                setting = std::make_shared<RationalQuadratic::Setting>();
                setting->type = Type::kRationalQuadratic;
            }
            return std::shared_ptr<RationalQuadratic>(new RationalQuadratic(std::move(setting)));
        }

        [[nodiscard]] std::pair<long, long>
        ComputeKtrain(Eigen::Ref<Eigen::MatrixXd> k_mat, const Eigen::Ref<const Eigen::MatrixXd> &mat_x) const final {
            long n = mat_x.cols();
            ERL_ASSERTM(k_mat.rows() >= n, "k_mat.rows() = %ld, it should be >= %ld.", k_mat.rows(), n);
            ERL_ASSERTM(k_mat.cols() >= n, "k_mat.cols() = %ld, it should be >= %ld.", k_mat.cols(), n);

            double a = double(0.5) / (m_setting_->scale * m_setting_->scale * m_setting_->scale_mix);
            for (long i = 0; i < n; ++i) {
                for (long j = i; j < n; ++j) {
                    if (i == j) {
                        k_mat(i, i) = m_setting_->alpha;
                    } else {
                        k_mat(i, j) = m_setting_->alpha * InlineRq(a, m_setting_->scale_mix, (mat_x.col(i) - mat_x.col(j)).squaredNorm());
                        k_mat(j, i) = k_mat(i, j);
                    }
                }
            }
            return {n, n};
        }

        [[nodiscard]] std::pair<long, long>
        ComputeKtrain(Eigen::Ref<Eigen::MatrixXd> k_mat, const Eigen::Ref<const Eigen::MatrixXd> &mat_x, const Eigen::Ref<const Eigen::VectorXd> &vec_sigma_y)
            const final {
            long n = mat_x.cols();
            ERL_ASSERTM(k_mat.rows() >= n, "k_mat.rows() = %ld, it should be >= %ld.", k_mat.rows(), n);
            ERL_ASSERTM(k_mat.cols() >= n, "k_mat.cols() = %ld, it should be >= %ld.", k_mat.cols(), n);
            ERL_ASSERTM(n == vec_sigma_y.size(), "#elements of vec_sigma_y does not equal to #columns of m_x_.");

            double a = double(0.5) / (m_setting_->scale * m_setting_->scale * m_setting_->scale_mix);
            for (long i = 0; i < n; ++i) {
                for (long j = i; j < n; ++j) {
                    if (i == j) {
                        k_mat(i, i) = m_setting_->alpha + vec_sigma_y[i];
                    } else {
                        k_mat(i, j) = m_setting_->alpha * InlineRq(a, m_setting_->scale_mix, (mat_x.col(i) - mat_x.col(j)).squaredNorm());
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

            double a = double(0.5) / (m_setting_->scale * m_setting_->scale * m_setting_->scale_mix);
            for (long i = 0; i < n; ++i) {
                for (long j = 0; j < m; ++j) {
                    k_mat(i, j) = m_setting_->alpha * InlineRq(a, m_setting_->scale_mix, (mat_x1.col(i) - mat_x2.col(j)).squaredNorm());
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

            double l2_inv = 1. / (m_setting_->scale * m_setting_->scale);
            double a = 0.5 * l2_inv / m_setting_->scale_mix;
            for (long i = 0; i < n; ++i) {
                k_mat(i, i) = m_setting_->alpha;  // cov(f_i, f_i)
                if (vec_grad_flags[i]) {
                    for (long k = 0, ki = grad_indices[i]; k < dim; ++k, ki += n_grad) {
                        // cov(df_i, df_i)
                        k_mat(ki, ki) = l2_inv;
                        // cov(f_i, df_i) and cov(df_i, f_i) are zeros
                        k_mat(i, ki) = 0.;
                        k_mat(ki, i) = 0.;
                    }
                }

                for (long j = i + 1; j < n; ++j) {
                    double r2 = (mat_x.col(i) - mat_x.col(j)).squaredNorm();
                    double beta = 1. / (1. + a * r2);
                    double gamma = beta * beta * l2_inv * (1. + m_setting_->scale_mix) / m_setting_->scale_mix;

                    k_mat(i, j) = m_setting_->alpha * InlineRq(a, m_setting_->scale_mix, r2);
                    k_mat(j, i) = k_mat(i, j);

                    if (vec_grad_flags[i]) {
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
                                double dxk = mat_x(k, i) - mat_x(k, j);
                                k_mat(ki, kj) = l2_inv * k_mat(i, j) * (beta - gamma * dxk * dxk);  // cov(df_i/dx_k, df_j/dx_k)
                                k_mat(kj, ki) = k_mat(ki, kj);                                      // cov(df_j/dx_k, df_i/dx_k)
                                for (long l = k + 1, li = ki + n_grad, lj = kj + n_grad; l < dim; ++l, li += n_grad, lj += n_grad) {
                                    // between Dim-k and Dim-l
                                    double dxl = mat_x(l, i) - mat_x(l, j);
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
            const Eigen::Ref<const Eigen::VectorXd> &vec_sigma_x,
            const Eigen::Ref<const Eigen::VectorXd> &vec_sigma_y,
            const Eigen::Ref<const Eigen::VectorXd> &vec_sigma_grad) const final {

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

            double l2_inv = 1. / (m_setting_->scale * m_setting_->scale);
            double a = 0.5 * l2_inv / m_setting_->scale_mix;
            for (long i = 0; i < n; ++i) {
                k_mat(i, i) = m_setting_->alpha + vec_sigma_x(i) + vec_sigma_y(i);  // cov(f_i, f_i)
                if (vec_grad_flags[i]) {
                    for (long k = 0, ki = grad_indices[i]; k < dim; ++k, ki += n_grad) {
                        // cov(df_i, df_i)
                        k_mat(ki, ki) = l2_inv + vec_sigma_grad(i);
                        // cov(f_i, df_i) and cov(df_i, f_i) are zeros
                        k_mat(i, ki) = 0.;
                        k_mat(ki, i) = 0.;
                    }
                }

                for (long j = i + 1; j < n; ++j) {
                    double r2 = (mat_x.col(i) - mat_x.col(j)).squaredNorm();
                    double beta = 1. / (1. + a * r2);
                    double gamma = beta * beta * l2_inv * (1. + m_setting_->scale_mix) / m_setting_->scale_mix;

                    k_mat(i, j) = m_setting_->alpha * InlineRq(a, m_setting_->scale_mix, r2);
                    k_mat(j, i) = k_mat(i, j);

                    if (vec_grad_flags[i]) {
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
                                double dxk = mat_x(k, i) - mat_x(k, j);
                                k_mat(ki, kj) = l2_inv * k_mat(i, j) * (beta - gamma * dxk * dxk);  // cov(df_i/dx_k, df_j/dx_k)
                                k_mat(kj, ki) = k_mat(ki, kj);                                      // cov(df_j/dx_k, df_i/dx_k)
                                for (long l = k + 1, li = ki + n_grad, lj = kj + n_grad; l < dim; ++l, li += n_grad, lj += n_grad) {
                                    // between Dim-k and Dim-l
                                    double dxl = mat_x(l, i) - mat_x(l, j);
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

            ERL_ASSERTM(dim == mat_x2.rows(), "Sample vectors stored in x_1 and x_2 should have the same dimension.");
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

            double l2_inv = 1. / (m_setting_->scale * m_setting_->scale);
            double a = 0.5 * l2_inv / m_setting_->scale_mix;
            for (long i = 0; i < n; ++i) {
                for (long j = 0; j < m; ++j) {
                    double r2 = (mat_x1.col(i) - mat_x2.col(j)).norm();
                    double beta = 1. / (1. + a * r2);
                    double gamma = beta * beta * l2_inv * (1. + m_setting_->scale_mix) / m_setting_->scale_mix;

                    k_mat(i, j) = m_setting_->alpha * InlineRq(a, m_setting_->scale_mix, r2);  // cov(f1_i, f2_j)
                    for (long k = 0, kj = j + m; k < dim; ++k, kj += m) {                      // cov(f1_i, df2_j/dx_k)
                        k_mat(i, kj) = beta * l2_inv * k_mat(i, j) * (mat_x1(k, i) - mat_x2(k, j));
                    }

                    if (vec_grad1_flags[i]) {
                        for (long k = 0, ki = grad_indices[i], kj = j + m; k < dim; ++k, ki += n_grad, kj += m) {
                            k_mat(ki, j) = -k_mat(i, kj);  // cov(df1_i/dx_k, f2_j) = -cov(f1_i, df2_j/dx_k)

                            // cov(df1_i, df2_j)
                            // between Dim-k and Dim-k
                            double dxk = mat_x1(k, i) - mat_x2(k, j);
                            k_mat(ki, kj) = l2_inv * k_mat(i, j) * (beta - gamma * dxk * dxk);

                            for (long l = k + 1, li = ki + n_grad, lj = kj + m; l < dim; ++l, li += n_grad, lj += m) {
                                // between Dim-k and Dim-l
                                double dxl = mat_x1(l, i) - mat_x2(l, j);
                                k_mat(ki, lj) = l2_inv * k_mat(i, j) * (-gamma * dxk * dxl);
                                k_mat(li, kj) = k_mat(ki, lj);
                            }
                        }
                    }
                }
            }
            return {n_rows, n_cols};
        }

    private:
        explicit RationalQuadratic(std::shared_ptr<Setting> setting)
            : Covariance(std::move(setting)) {
            ERL_ASSERTM(m_setting_->type == Type::kRationalQuadratic, "setting->type should be kRationalQuadratic.");
            ERL_ASSERTM(Dim == Eigen::Dynamic || m_setting_->x_dim == Dim, "setting->x_dim should be %ld.", Dim);
            ERL_WARN_ONCE_COND(Dim == Eigen::Dynamic, "Dim is Eigen::Dynamic, it may cause performance issue.");
        }
    };
}  // namespace erl::covariance
