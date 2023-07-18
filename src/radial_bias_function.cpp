#include "erl_covariance/radial_bias_function.hpp"

namespace erl::covariance {
    std::shared_ptr<RadialBiasFunction>
    RadialBiasFunction::Create() {
        return std::shared_ptr<RadialBiasFunction>(new RadialBiasFunction(std::make_shared<Setting>(Type::kRadialBiasFunction)));
    }

    std::shared_ptr<RadialBiasFunction>
    RadialBiasFunction::Create(std::shared_ptr<Setting> setting) {
        return std::shared_ptr<RadialBiasFunction>(new RadialBiasFunction(std::move(setting)));
    }

//    static inline double
//    InlineRbf(double a, double squared_norm) {
//        return std::exp(-a * squared_norm);
//    }

    Eigen::MatrixXd
    RadialBiasFunction::ComputeKtrain(const Eigen::Ref<const Eigen::MatrixXd> &mat_x) const {
        auto n = mat_x.cols();
        auto a = double(0.5) / (m_setting_->scale * m_setting_->scale);
        Eigen::MatrixXd k_mat(n, n);

        for (long i = 0; i < n; ++i) {
            for (long j = i; j < n; ++j) {
                if (i == j) {
                    k_mat(i, i) = m_setting_->alpha;
                } else {
                    // k_mat(i, j) = m_setting_->alpha * InlineRbf(a, (mat_x.col(i) - mat_x.col(j)).squaredNorm());
                    k_mat(i, j) = m_setting_->alpha * std::exp(-a * (mat_x.col(i) - mat_x.col(j)).squaredNorm());
                    k_mat(j, i) = k_mat(i, j);
                }
            }
        }

        return k_mat;
    }

    Eigen::MatrixXd
    RadialBiasFunction::ComputeKtrain(const Eigen::Ref<const Eigen::MatrixXd> &mat_x, const Eigen::Ref<const Eigen::VectorXd> &vec_sigma_y) const {
        auto n = mat_x.cols();
        ERL_DEBUG_ASSERT(n == vec_sigma_y.size(), "#elements of vec_sigma_y does not equal to #columns of m_x_.");
        auto a = double(0.5) / (m_setting_->scale * m_setting_->scale);
        Eigen::MatrixXd k_mat(n, n);

        for (long i = 0; i < n; ++i) {
            for (long j = i; j < n; ++j) {
                if (i == j) {
                    k_mat(i, i) = m_setting_->alpha + vec_sigma_y[i];
                } else {
                    // k_mat(i, j) = m_setting_->alpha * InlineRbf(a, (mat_x.col(i) - mat_x.col(j)).squaredNorm());
                    k_mat(i, j) = m_setting_->alpha * std::exp(-a * (mat_x.col(i) - mat_x.col(j)).squaredNorm());
                    k_mat(j, i) = k_mat(i, j);
                }
            }
        }

        return k_mat;
    }

    Eigen::MatrixXd
    RadialBiasFunction::ComputeKtest(const Eigen::Ref<const Eigen::MatrixXd> &mat_x1, const Eigen::Ref<const Eigen::MatrixXd> &mat_x2) const {
        ERL_DEBUG_ASSERT(mat_x1.rows() == mat_x2.rows(), "Sample vectors stored in x_1 and x_2 should have the same dimension.");

        auto n = mat_x1.cols();
        auto m = mat_x2.cols();

        auto a = double(0.5) / (m_setting_->scale * m_setting_->scale);
        Eigen::MatrixXd k_mat(n, m);

        for (long i = 0; i < n; ++i) {
            for (long j = 0; j < m; ++j) { k_mat(i, j) = m_setting_->alpha * std::exp(-a * (mat_x1.col(i) - mat_x2.col(j)).squaredNorm()); }
        }

        return k_mat;
    }

    // cov(f1, df2/dx2) = d k(x1, x2) / dx2
    // note: dx = x1 - x2
//    static inline double
//    InlineRbfX1BetweenGradx2(double a, double dx, double squared_norm) {
//        return 2 * a * dx * std::exp(-a * squared_norm);
//    }
//
//    // d^2 k(x1, x2) / dx_1 dx_2
//    static inline double
//    InlineRbfGradx1BetweenGradx2(double a, double delta, double dx_1, double dx_2, double squared_norm) {
//        double double_a = 2 * a;
//        double k = std::exp(-a * squared_norm);
//        return -double_a * k * (double_a * dx_1 * dx_2 - delta);
//    }

    Eigen::MatrixXd
    RadialBiasFunction::ComputeKtrainWithGradient(const Eigen::Ref<const Eigen::MatrixXd> &mat_x, const Eigen::Ref<const Eigen::VectorXb> &vec_grad_flags)
        const {

        long dim = mat_x.rows();
        long n = mat_x.cols();
        double l2_inv = 1.0 / (m_setting_->scale * m_setting_->scale);
        double a = 0.5 * l2_inv;

        std::vector<decltype(n)> grad_indices;
        grad_indices.reserve(vec_grad_flags.size());
        decltype(n) n_grad = 0;
        for (bool flag: vec_grad_flags) {
            if (flag) {
                grad_indices.push_back(n + n_grad++);
            } else {
                grad_indices.push_back(-1);
            }
        }

        Eigen::MatrixXd k_mat = Eigen::MatrixXd::Zero(n + n_grad * dim, n + n_grad * dim);

        for (long i = 0; i < n; ++i) {
            k_mat(i, i) = m_setting_->alpha;  // cov(f_i, f_i)
            // k_mat(i, i) = m_setting_->alpha + vec_sigma_x(i) + vec_sigma_y(i);  // cov(f_i, f_i)
            if (vec_grad_flags[i]) {
                for (decltype(n) k = 0, ki = grad_indices[i]; k < dim; ++k, ki += n_grad) {
                    // cov(df_i, df_i)
                    k_mat(ki, ki) = m_setting_->alpha * l2_inv;
                    // cov(f_i, df_i) and cov(df_i, f_i) are zeros
                }
            }

            for (long j = i + 1; j < n; ++j) {
                auto r2 = (mat_x.col(i) - mat_x.col(j)).squaredNorm();
                k_mat(i, j) = m_setting_->alpha * std::exp(-a * r2);  // cov(f_i, f_j)
                k_mat(j, i) = k_mat(i, j);

                if (vec_grad_flags[i]) {
                    // cov(f_j, df_i) = cov(df_i, f_j)
                    for (decltype(n) k = 0, ki = grad_indices[i]; k < dim; ++k, ki += n_grad) {
                        k_mat(j, ki) = l2_inv * (mat_x(k, j) - mat_x(k, i)) * k_mat(i, j);  // cov(f_j, df_i)
                        // k_mat(j, ki) = m_setting_->alpha * InlineRbfX1BetweenGradx2(a, mat_x(k, j) - mat_x(k, i), r2);  // cov(f_j, df_i)
                        k_mat(ki, j) = k_mat(j, ki);                                                                    // cov(df_i, f_j)
                    }

                    if (vec_grad_flags[j]) {
                        for (decltype(n) k = 0, ki = grad_indices[i], kj = grad_indices[j]; k < dim; ++k, ki += n_grad, kj += n_grad) {
                            k_mat(i, kj) = -k_mat(j, ki);  // cov(f_i, df_j) = -cov(df_i, f_j)
                            k_mat(kj, i) = k_mat(i, kj);   // cov(df_j, f_i) = -cov(f_j, df_i) = cov(f_i, df_j)
                        }

                        // cov(df_i, df_j) = cov(df_j, df_i)
                        for (decltype(n) k = 0, ki = grad_indices[i], kj = grad_indices[j]; k < dim; ++k, ki += n_grad, kj += n_grad) {
                            // between Dim-k and Dim-k
                            auto dxk = mat_x(k, i) - mat_x(k, j);
                            k_mat(ki, kj) = l2_inv * k_mat(i, j) * (1. - l2_inv * dxk * dxk);  // cov(df_i, df_j)
                            // k_mat(ki, kj) = m_setting_->alpha * InlineRbfGradx1BetweenGradx2(a, 1., dxk, dxk, r2);  // cov(df_i, df_j)
                            k_mat(kj, ki) = k_mat(ki, kj);                                                          // cov(df_j, df_i)
                            for (decltype(n) l = k + 1, li = ki + n_grad, lj = kj + n_grad; l < dim; ++l, li += n_grad, lj += n_grad) {
                                // between Dim-k and Dim-l
                                auto dxl = mat_x(l, i) - mat_x(l, j);
                                k_mat(ki, lj) = l2_inv * k_mat(i, j) * (-l2_inv * dxk * dxl);  // cov(df_i, df_j)
                                // k_mat(ki, lj) = m_setting_->alpha * InlineRbfGradx1BetweenGradx2(a, 0., dxk, dxl, r2);
                                k_mat(li, kj) = k_mat(ki, lj);
                                k_mat(lj, ki) = k_mat(ki, lj);  // cov(df_j, df_i)
                                k_mat(kj, li) = k_mat(lj, ki);
                            }
                        }
                    }
                } else if (vec_grad_flags[j]) {
                    // cov(f_i, df_j) = cov(df_j, f_i)
                    for (decltype(n) k = 0, kj = grad_indices[j]; k < dim; ++k, kj += n_grad) {
                        k_mat(i, kj) = l2_inv * (mat_x(k, i) - mat_x(k, j)) * k_mat(i, j);  // cov(f_i, df_j
                        // k_mat(i, kj) = m_setting_->alpha * InlineRbfX1BetweenGradx2(a, mat_x(k, i) - mat_x(k, j), r2);  // cov(f_i, df_j)
                        k_mat(kj, i) = k_mat(i, kj);
                    }
                }
            }  // for (long j = i + 1; j < n; ++j)
        }      // for (long i = 0; i < n; ++i)

        return k_mat;
    }

    Eigen::MatrixXd
    RadialBiasFunction::ComputeKtrainWithGradient(
        const Eigen::Ref<const Eigen::MatrixXd> &mat_x,
        const Eigen::Ref<const Eigen::VectorXb> &vec_grad_flags,
        const Eigen::Ref<const Eigen::VectorXd> &vec_sigma_x,
        const Eigen::Ref<const Eigen::VectorXd> &vec_sigma_y,
        const Eigen::Ref<const Eigen::VectorXd> &vec_sigma_grad) const {

        long dim = mat_x.rows();
        long n = mat_x.cols();
        double l2_inv = 1.0 / (m_setting_->scale * m_setting_->scale);
        double a = 0.5 * l2_inv;

        std::vector<decltype(n)> grad_indices;
        grad_indices.reserve(vec_grad_flags.size());
        decltype(n) n_grad = 0;
        for (bool flag: vec_grad_flags) {
            if (flag) {
                grad_indices.push_back(n + n_grad++);
            } else {
                grad_indices.push_back(-1);
            }
        }

        Eigen::MatrixXd k_mat = Eigen::MatrixXd::Zero(n + n_grad * dim, n + n_grad * dim);

        for (long i = 0; i < n; ++i) {
            k_mat(i, i) = m_setting_->alpha + vec_sigma_x(i) + vec_sigma_y(i);  // cov(f_i, f_i)
            if (vec_grad_flags[i]) {
                for (decltype(n) k = 0, ki = grad_indices[i]; k < dim; ++k, ki += n_grad) {
                    // cov(df_i, df_i)
                    k_mat(ki, ki) = m_setting_->alpha * l2_inv + vec_sigma_grad(i);
                    // cov(f_i, df_i) and cov(df_i, f_i) are zeros
                }
            }

            for (long j = i + 1; j < n; ++j) {
                auto r2 = (mat_x.col(i) - mat_x.col(j)).squaredNorm();
                k_mat(i, j) = m_setting_->alpha * std::exp(-a * r2);  // cov(f_i, f_j)
                k_mat(j, i) = k_mat(i, j);

                if (vec_grad_flags[i]) {
                    // cov(f_j, df_i) = cov(df_i, f_j)
                    for (decltype(n) k = 0, ki = grad_indices[i]; k < dim; ++k, ki += n_grad) {
                        k_mat(j, ki) = l2_inv * (mat_x(k, j) - mat_x(k, i)) * k_mat(i, j);  // cov(f_j, df_i)
                        // k_mat(j, ki) = m_setting_->alpha * InlineRbfX1BetweenGradx2(a, mat_x(k, j) - mat_x(k, i), r2);  // cov(f_j, df_i)
                        k_mat(ki, j) = k_mat(j, ki);                                                                    // cov(df_i, f_j)
                    }

                    if (vec_grad_flags[j]) {
                        for (decltype(n) k = 0, ki = grad_indices[i], kj = grad_indices[j]; k < dim; ++k, ki += n_grad, kj += n_grad) {
                            k_mat(i, kj) = -k_mat(j, ki);  // cov(f_i, df_j) = -cov(df_i, f_j)
                            k_mat(kj, i) = k_mat(i, kj);   // cov(df_j, f_i) = -cov(f_j, df_i) = cov(f_i, df_j)
                        }

                        // cov(df_i, df_j) = cov(df_j, df_i)
                        for (decltype(n) k = 0, ki = grad_indices[i], kj = grad_indices[j]; k < dim; ++k, ki += n_grad, kj += n_grad) {
                            // between Dim-k and Dim-k
                            auto dxk = mat_x(k, i) - mat_x(k, j);
                            k_mat(ki, kj) = l2_inv * k_mat(i, j) * (1. - l2_inv * dxk * dxk);  // cov(df_i, df_j)
                            // k_mat(ki, kj) = m_setting_->alpha * InlineRbfGradx1BetweenGradx2(a, 1., dxk, dxk, r2);  // cov(df_i, df_j)
                            k_mat(kj, ki) = k_mat(ki, kj);                                                          // cov(df_j, df_i)
                            for (decltype(n) l = k + 1, li = ki + n_grad, lj = kj + n_grad; l < dim; ++l, li += n_grad, lj += n_grad) {
                                // between Dim-k and Dim-l
                                auto dxl = mat_x(l, i) - mat_x(l, j);
                                k_mat(ki, lj) = l2_inv * k_mat(i, j) * (-l2_inv * dxk * dxl);  // cov(df_i, df_j)
                                // k_mat(ki, lj) = m_setting_->alpha * InlineRbfGradx1BetweenGradx2(a, 0., dxk, dxl, r2);
                                k_mat(li, kj) = k_mat(ki, lj);
                                k_mat(lj, ki) = k_mat(ki, lj);  // cov(df_j, df_i)
                                k_mat(kj, li) = k_mat(lj, ki);
                            }
                        }
                    }
                } else if (vec_grad_flags[j]) {
                    // cov(f_i, df_j) = cov(df_j, f_i)
                    for (decltype(n) k = 0, kj = grad_indices[j]; k < dim; ++k, kj += n_grad) {
                        k_mat(i, kj) = l2_inv * (mat_x(k, i) - mat_x(k, j)) * k_mat(i, j);  // cov(f_i, df_j
                        // k_mat(i, kj) = m_setting_->alpha * InlineRbfX1BetweenGradx2(a, mat_x(k, i) - mat_x(k, j), r2);  // cov(f_i, df_j)
                        k_mat(kj, i) = k_mat(i, kj);
                    }
                }
            }  // for (long j = i + 1; j < n; ++j)
        }      // for (long i = 0; i < n; ++i)

        return k_mat;
    }

    Eigen::MatrixXd
    RadialBiasFunction::ComputeKtestWithGradient(
        const Eigen::Ref<const Eigen::MatrixXd> &mat_x1,
        const Eigen::Ref<const Eigen::VectorXb> &vec_grad1_flags,
        const Eigen::Ref<const Eigen::MatrixXd> &mat_x2) const {

        auto dim = mat_x1.rows();
        ERL_DEBUG_ASSERT(dim == mat_x2.rows(), "Sample vectors stored in x_1 and x_2 should have the same dimension.");

        auto n = mat_x1.cols();
        auto m = mat_x2.cols();
        double l2_inv = 1.0 / (m_setting_->scale * m_setting_->scale);
        double a = 0.5 * l2_inv;

        std::vector<long> grad_indices;
        grad_indices.reserve(vec_grad1_flags.size());
        long n_grad = 0;
        for (bool flag: vec_grad1_flags) {
            if (flag) {
                grad_indices.push_back(n + n_grad++);
            } else {
                grad_indices.push_back(-1);
            }
        }

        Eigen::MatrixXd k_mat = Eigen::MatrixXd::Zero(n + n_grad * dim, m * (dim + 1));

        for (long i = 0; i < n; ++i) {
            for (long j = 0; j < m; ++j) {
                auto r2 = (mat_x1.col(i) - mat_x2.col(j)).squaredNorm();
                // cov(f1_i, f2_j)
                k_mat(i, j) = m_setting_->alpha * std::exp(-a * r2);
                // cov(f1_i, df2_j)
                for (long k = 0, kj = j + m; k < dim; ++k, kj += m) {
                    k_mat(i, kj) = l2_inv * k_mat(i, j) * (mat_x1(k, i) - mat_x2(k, j));  // cov(f1_i, df2_j)
//                    k_mat(i, kj) = m_setting_->alpha * InlineRbfX1BetweenGradx2(a, mat_x1(k, i) - mat_x2(k, j), r2);
                }

                if (vec_grad1_flags[i]) {
                    for (long k = 0, ki = grad_indices[i], kj = j + m; k < dim; ++k, ki += n_grad, kj += m) {
                        // cov(df1_i, f2_j) = -cov(f1_i, df2_j)
                        k_mat(ki, j) = -k_mat(i, kj);

                        // cov(df1_i, df2_j)
                        // between Dim-k and Dim-k
                        auto dxk = mat_x1(k, i) - mat_x2(k, j);
                        k_mat(ki, kj) = l2_inv * k_mat(i, j) * (1. - l2_inv * dxk * dxk);  // cov(df1_i, df2_j)
//                        k_mat(ki, kj) = m_setting_->alpha * InlineRbfGradx1BetweenGradx2(a, 1., dxk, dxk, r2);

                        for (long l = k + 1, li = ki + n_grad, lj = kj + m; l < dim; ++l, li += n_grad, lj += m) {
                            // between Dim-k and Dim-l
                            auto dxl = mat_x1(l, i) - mat_x2(l, j);
                            k_mat(ki, lj) = l2_inv * k_mat(i, j) * (-l2_inv * dxk * dxl);  // cov(df1_i, df2_j)
//                            k_mat(ki, lj) = m_setting_->alpha * InlineRbfGradx1BetweenGradx2(a, 0., dxk, dxl, r2);
                            k_mat(li, kj) = k_mat(ki, lj);
                        }
                    }
                }
            }
        }

        return k_mat;
    }

    RadialBiasFunction::RadialBiasFunction(std::shared_ptr<Setting> setting)
        : Covariance(std::move(setting)) {
        ERL_ASSERTM(m_setting_->type == Type::kRadialBiasFunction, "setting->type should be RADIAL_BIAS_FUNCTION.");
    }
}  // namespace erl::covariance
