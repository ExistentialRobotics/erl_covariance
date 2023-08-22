#pragma once

#include "covariance.hpp"

namespace erl::covariance {

    template<long Dim>
    class OrnsteinUhlenbeck : public Covariance {
        // ref1: https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
        // ref2: https://www.cs.cmu.edu/~epxing/Class/10708-15/notes/10708_scribe_lecture21.pdf

    public:
        static std::shared_ptr<OrnsteinUhlenbeck>
        Create(std::shared_ptr<Setting> setting = nullptr) {
            if (setting == nullptr) {
                setting = std::make_shared<OrnsteinUhlenbeck::Setting>();
                setting->type = Type::kOrnsteinUhlenbeck;
            }
            return std::shared_ptr<OrnsteinUhlenbeck>(new OrnsteinUhlenbeck(std::move(setting)));
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
            double a = -1. / m_setting_->scale;
            for (long i = 0; i < n; ++i) {
                for (long j = i; j < n; ++j) {
                    if (i == j) {
                        k_mat(i, i) = m_setting_->alpha;
                    } else {
                        double r = 0;
                        for (long k = 0; k < dim; ++k) {
                            double dx = mat_x(k, i) - mat_x(k, j);
                            r += dx * dx;
                        }
                        r = std::sqrt(r);  // (mat_x.col(i) - mat_x.col(j)).norm();
                        k_mat(i, j) = m_setting_->alpha * std::exp(a * r);
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
            ERL_ASSERTM(k_mat.rows() >= n, "k_mat.rows() = %ld, it should be >= %ld.", k_mat.rows(), n);
            ERL_ASSERTM(k_mat.cols() >= n, "k_mat.cols() = %ld, it should be >= %ld.", k_mat.cols(), n);
            ERL_ASSERTM(n == vec_var_y.size(), "#elements of vec_sigma_y does not equal to #columns of mat_x.");
            long dim;
            if constexpr (Dim == Eigen::Dynamic) {
                dim = mat_x.rows();
            } else {
                dim = Dim;
            }
            double a = -1. / m_setting_->scale;
            for (long i = 0; i < n; ++i) {
                for (long j = i; j < n; ++j) {
                    if (i == j) {
                        k_mat(i, i) = m_setting_->alpha + vec_var_y[i];
                    } else {
                        double r = 0;
                        for (long k = 0; k < dim; ++k) {
                            double dx = mat_x(k, i) - mat_x(k, j);
                            r += dx * dx;
                        }
                        r = std::sqrt(r);  // (mat_x.col(i) - mat_x.col(j)).norm();
                        k_mat(i, j) = m_setting_->alpha * std::exp(a * r);
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
            double a = -1. / m_setting_->scale;
            for (long i = 0; i < n; ++i) {
                for (long j = 0; j < m; ++j) {
                    double r = 0;
                    for (long k = 0; k < dim; ++k) {
                        double dx = mat_x1(k, i) - mat_x2(k, j);
                        r += dx * dx;
                    }
                    r = std::sqrt(r);  // (mat_x1.col(i) - mat_x2.col(j)).norm();
                    k_mat(i, j) = m_setting_->alpha * std::exp(a * r);
                }
            }
            return {n, m};
        }

        [[nodiscard]] std::pair<long, long>
        ComputeKtrainWithGradient(
            Eigen::Ref<Eigen::MatrixXd>,                // k_mat
            const Eigen::Ref<const Eigen::MatrixXd> &,  // mat_x
            const Eigen::Ref<const Eigen::VectorXb> &   // vec_grad_flags
        ) const final {
            throw NotImplemented(__PRETTY_FUNCTION__);
        }

        [[nodiscard]] std::pair<long, long>
        ComputeKtrainWithGradient(
            Eigen::Ref<Eigen::MatrixXd>,                // k_mat
            const Eigen::Ref<const Eigen::MatrixXd> &,  // mat_x
            const Eigen::Ref<const Eigen::VectorXb> &,  // vec_grad_flags
            const Eigen::Ref<const Eigen::VectorXd> &,  // vec_var_x
            const Eigen::Ref<const Eigen::VectorXd> &,  // vec_var_y
            const Eigen::Ref<const Eigen::VectorXd> &   // vec_var_grad
        ) const final {
            throw NotImplemented(__PRETTY_FUNCTION__);
        }

        [[nodiscard]] std::pair<long, long>
        ComputeKtestWithGradient(
            Eigen::Ref<Eigen::MatrixXd>,                // k_mat
            const Eigen::Ref<const Eigen::MatrixXd> &,  // mat_x1
            const Eigen::Ref<const Eigen::VectorXb> &,  // vec_grad1_flags
            const Eigen::Ref<const Eigen::MatrixXd> &   // mat_x2
        ) const final {
            throw NotImplemented(__PRETTY_FUNCTION__);
        }

    private:
        explicit OrnsteinUhlenbeck(std::shared_ptr<Setting> setting)
            : Covariance(std::move(setting)) {
            ERL_ASSERTM(m_setting_->type == Type::kOrnsteinUhlenbeck, "setting->type should be kOrnsteinUhlenbeck.");
            ERL_ASSERTM(Dim == Eigen::Dynamic || m_setting_->x_dim == Dim, "setting->x_dim should be %ld.", Dim);
            ERL_WARN_ONCE_COND(Dim == Eigen::Dynamic, "Dim is Eigen::Dynamic, it may cause performance issue.");
        }
    };
}  // namespace erl::covariance
