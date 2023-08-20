#include "erl_covariance/ornstein_uhlenbeck.hpp"

namespace erl::covariance {
    std::shared_ptr<OrnsteinUhlenbeck>
    OrnsteinUhlenbeck::Create(std::shared_ptr<Setting> setting) {
        if (setting == nullptr) {
            setting = std::make_shared<OrnsteinUhlenbeck::Setting>();
            setting->type = Type::kOrnsteinUhlenbeck;
        }
        return std::shared_ptr<OrnsteinUhlenbeck>(new OrnsteinUhlenbeck(std::move(setting)));
    }

    static inline double
    InlineOu(const double &a, const double &r) {
        return std::exp(-a * r);
    }

    std::pair<long, long>
    OrnsteinUhlenbeck::ComputeKtrain(Eigen::Ref<Eigen::MatrixXd> k_mat, const Eigen::Ref<const Eigen::MatrixXd> &mat_x) const {
        long n = mat_x.cols();
        ERL_ASSERTM(k_mat.rows() >= n, "k_mat.rows() = %ld, it should be >= %ld.", k_mat.rows(), n);
        ERL_ASSERTM(k_mat.cols() >= n, "k_mat.cols() = %ld, it should be >= %ld.", k_mat.cols(), n);

        double a = 1. / m_setting_->scale;
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
        return {n, n};
    }

    std::pair<long, long>
    OrnsteinUhlenbeck::ComputeKtrain(
        Eigen::Ref<Eigen::MatrixXd> k_mat,
        const Eigen::Ref<const Eigen::MatrixXd> &mat_x,
        const Eigen::Ref<const Eigen::VectorXd> &vec_sigma_y) const {
        long n = mat_x.cols();
        ERL_ASSERTM(k_mat.rows() >= n, "k_mat.rows() = %ld, it should be >= %ld.", k_mat.rows(), n);
        ERL_ASSERTM(k_mat.cols() >= n, "k_mat.cols() = %ld, it should be >= %ld.", k_mat.cols(), n);
        ERL_ASSERTM(n == vec_sigma_y.size(), "#elements of vec_sigma_y does not equal to #columns of mat_x.");

        double a = 1. / m_setting_->scale;
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
        return {n, n};
    }

    std::pair<long, long>
    OrnsteinUhlenbeck::ComputeKtest(
        Eigen::Ref<Eigen::MatrixXd> k_mat,
        const Eigen::Ref<const Eigen::MatrixXd> &mat_x1,
        const Eigen::Ref<const Eigen::MatrixXd> &mat_x2) const {
        ERL_ASSERTM(mat_x1.rows() == mat_x2.rows(), "Sample vectors stored in x_1 and x_2 should have the same dimension.");

        long n = mat_x1.cols();
        long m = mat_x2.cols();
        ERL_ASSERTM(k_mat.rows() >= n, "k_mat.rows() = %ld, it should be >= %ld.", k_mat.rows(), n);
        ERL_ASSERTM(k_mat.cols() >= m, "k_mat.cols() = %ld, it should be >= %ld.", k_mat.cols(), m);

        double a = 1. / m_setting_->scale;
        for (long i = 0; i < n; ++i) {
            for (long j = 0; j < m; ++j) { k_mat(i, j) = m_setting_->alpha * InlineOu(a, (mat_x1.col(i) - mat_x2.col(j)).norm()); }
        }
        return {n, m};
    }

    std::pair<long, long>
    OrnsteinUhlenbeck::ComputeKtrainWithGradient(
        Eigen::Ref<Eigen::MatrixXd>,
        const Eigen::Ref<const Eigen::MatrixXd> &,
        const Eigen::Ref<const Eigen::VectorXb> &) const {
        throw NotImplemented(__PRETTY_FUNCTION__);
    }

    std::pair<long, long>
    OrnsteinUhlenbeck::ComputeKtrainWithGradient(
        Eigen::Ref<Eigen::MatrixXd>,
        const Eigen::Ref<const Eigen::MatrixXd> &,
        const Eigen::Ref<const Eigen::VectorXb> &,
        const Eigen::Ref<const Eigen::VectorXd> &,
        const Eigen::Ref<const Eigen::VectorXd> &,
        const Eigen::Ref<const Eigen::VectorXd> &) const {
        throw NotImplemented(__PRETTY_FUNCTION__);
    }

    std::pair<long, long>
    OrnsteinUhlenbeck::ComputeKtestWithGradient(
        Eigen::Ref<Eigen::MatrixXd>,
        const Eigen::Ref<const Eigen::MatrixXd> &,
        const Eigen::Ref<const Eigen::VectorXb> &,
        const Eigen::Ref<const Eigen::MatrixXd> &) const {
        throw NotImplemented(__PRETTY_FUNCTION__);
    }

    OrnsteinUhlenbeck::OrnsteinUhlenbeck(std::shared_ptr<Setting> setting)
        : Covariance(std::move(setting)) {
        ERL_ASSERTM(m_setting_->type == Type::kOrnsteinUhlenbeck, "setting->type should be kOrnsteinUhlenbeck.");
    }

}  // namespace erl::covariance
