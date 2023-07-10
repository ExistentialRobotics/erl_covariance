#include "erl_covariance/ornstein_uhlenbeck.hpp"

namespace erl::covariance {
    std::shared_ptr<OrnsteinUhlenbeck>
    OrnsteinUhlenbeck::Create() {
        return std::shared_ptr<OrnsteinUhlenbeck>(new OrnsteinUhlenbeck(std::make_shared<Setting>(Type::kOrnsteinUhlenbeck)));
    }

    std::shared_ptr<OrnsteinUhlenbeck>
    OrnsteinUhlenbeck::Create(std::shared_ptr<Setting> setting) {
        return std::shared_ptr<OrnsteinUhlenbeck>(new OrnsteinUhlenbeck(std::move(setting)));
    }

    static inline double
    InlineOu(const double &a, const double &r) {
        return std::exp(-a * r);
    }

    Eigen::MatrixXd
    OrnsteinUhlenbeck::ComputeKtrain(const Eigen::Ref<const Eigen::MatrixXd> &mat_x) const {
        auto n = mat_x.cols();
        auto a = 1. / m_setting_->scale;
        Eigen::MatrixXd k_mat(n, n);  // allocation the matrix without initialization

        for (long i = 0; i < n; ++i) {
            for (long j = i; j < n; ++j) {
                if (i == j) {
                    k_mat(i, i) = m_setting_->alpha;
                } else {
                    k_mat(i, j) = m_setting_->alpha * InlineOu(a, (mat_x.col(i) - mat_x.col(j)).norm());
                    k_mat(j, i) = k_mat(i, j);
                }
            }
        }

        return k_mat;
    }

    Eigen::MatrixXd
    OrnsteinUhlenbeck::ComputeKtrain(const Eigen::Ref<const Eigen::MatrixXd> &mat_x, const Eigen::Ref<const Eigen::VectorXd> &vec_sigma_y) const {
        auto n = mat_x.cols();
        ERL_DEBUG_ASSERT(n == vec_sigma_y.size(), "#elements of vec_sigma_y does not equal to #columns of mat_x.");
        auto a = 1. / m_setting_->scale;
        Eigen::MatrixXd k_mat(n, n);  // allocation the matrix without initialization

        for (long i = 0; i < n; ++i) {
            for (long j = i; j < n; ++j) {
                if (i == j) {
                    k_mat(i, i) = m_setting_->alpha + vec_sigma_y[i];
                } else {
                    k_mat(i, j) = m_setting_->alpha * InlineOu(a, (mat_x.col(i) - mat_x.col(j)).norm());
                    k_mat(j, i) = k_mat(i, j);
                }
            }
        }

        return k_mat;
    }

    Eigen::MatrixXd
    OrnsteinUhlenbeck::ComputeKtest(const Eigen::Ref<const Eigen::MatrixXd> &mat_x1, const Eigen::Ref<const Eigen::MatrixXd> &mat_x2) const {
        ERL_DEBUG_ASSERT(mat_x1.rows() == mat_x2.rows(), "Sample vectors stored in x_1 and x_2 should have the same dimension.");

        auto n = mat_x1.cols();
        auto m = mat_x2.cols();

        auto a = 1. / m_setting_->scale;
        Eigen::MatrixXd k_mat(n, m);

        for (long i = 0; i < n; ++i) {
            for (long j = 0; j < m; ++j) { k_mat(i, j) = m_setting_->alpha * InlineOu(a, (mat_x1.col(i) - mat_x2.col(j)).norm()); }
        }

        return k_mat;
    }

    Eigen::MatrixXd
    OrnsteinUhlenbeck::ComputeKtrainWithGradient(const Eigen::Ref<const Eigen::MatrixXd> &, const Eigen::Ref<const Eigen::VectorXb> &) const {
        throw NotImplemented(__PRETTY_FUNCTION__);
    }

    Eigen::MatrixXd
    OrnsteinUhlenbeck::ComputeKtrainWithGradient(
        const Eigen::Ref<const Eigen::MatrixXd> &,
        const Eigen::Ref<const Eigen::VectorXb> &,
        const Eigen::Ref<const Eigen::VectorXd> &,
        const Eigen::Ref<const Eigen::VectorXd> &,
        const Eigen::Ref<const Eigen::VectorXd> &) const {
        throw NotImplemented(__PRETTY_FUNCTION__);
    }

    Eigen::MatrixXd
    OrnsteinUhlenbeck::ComputeKtestWithGradient(
        const Eigen::Ref<const Eigen::MatrixXd> &,
        const Eigen::Ref<const Eigen::VectorXb> &,
        const Eigen::Ref<const Eigen::MatrixXd> &) const {
        throw NotImplemented(__PRETTY_FUNCTION__);
    }

    OrnsteinUhlenbeck::OrnsteinUhlenbeck(std::shared_ptr<Setting> setting)
        : Covariance(std::move(setting)) {
        ERL_ASSERTM(m_setting_->type == Type::kOrnsteinUhlenbeck, "setting->type should be ORNSTEIN_UHLENBECK.");
    }

}  // namespace erl::covariance
