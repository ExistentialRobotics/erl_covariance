#include "erl_covariance/custom_kernel_v3.hpp"

static inline double
InlineExpr(const double &a, const Eigen::VectorXd &weights, const Eigen::Ref<const Eigen::Vector3d> &x_1, const Eigen::Ref<const Eigen::Vector3d> &x_2) {
    Eigen::Vector3d diff = x_1 - x_2;
    diff.z() = std::abs(diff.z());
    diff.z() = std::min(diff.z(), 2 * M_PI - diff.z());
    return std::exp(-a * (weights.x() * std::sqrt(diff.x() * diff.x() + diff.y() * diff.y()) + weights.y() * diff.z()));
}

namespace erl::covariance {
    std::shared_ptr<CustomKernelV3>
    CustomKernelV3::Create(std::shared_ptr<Setting> setting) {
        if (setting == nullptr) {
            setting = std::make_shared<CustomKernelV3::Setting>();
            setting->type = Type::kCustomKernelV3;
        }
        return std::shared_ptr<CustomKernelV3>(new CustomKernelV3(std::move(setting)));
    }

    Eigen::MatrixXd
    CustomKernelV3::ComputeKtrain(const Eigen::Ref<const Eigen::MatrixXd> &mat_x) const {
        auto n = mat_x.cols();
        ERL_DEBUG_ASSERT(mat_x.rows() == 3, "Each column of mat_x should be 3D vector [m_x_, m_y_, angle].");
        ERL_DEBUG_ASSERT(m_setting_->weights.size() == 2, "Number of weights should be 2. Set GetSetting()->weights at first.");

        auto a = 1. / m_setting_->scale;
        Eigen::MatrixXd k_mat(n, n);

        for (long i = 0; i < n; ++i) {
            for (long j = i; j < n; ++j) {
                if (i == j) {
                    k_mat(i, i) = m_setting_->alpha;
                } else {
                    k_mat(i, j) = m_setting_->alpha * InlineExpr(a, m_setting_->weights, mat_x.col(i), mat_x.col(j));
                    k_mat(j, i) = k_mat(i, j);
                }
            }
        }

        return k_mat;
    }

    Eigen::MatrixXd
    CustomKernelV3::ComputeKtrain(const Eigen::Ref<const Eigen::MatrixXd> &mat_x, const Eigen::Ref<const Eigen::VectorXd> &vec_sigma_y) const {
        auto n = mat_x.cols();
        ERL_DEBUG_ASSERT(mat_x.rows() == 3, "Each column of mat_x should be 3D vector [m_x_, m_y_, angle].");
        ERL_DEBUG_ASSERT(n == vec_sigma_y.size(), "#elements of vec_sigma_y does not equal to #columns of mat_x.");
        ERL_DEBUG_ASSERT(m_setting_->weights.size() == 2, "Number of weights should be 2. Set GetSetting()->weights at first.");

        auto a = 1. / m_setting_->scale;
        Eigen::MatrixXd k_mat(n, n);

        for (long i = 0; i < n; ++i) {
            for (long j = i; j < n; ++j) {
                if (i == j) {
                    k_mat(i, i) = m_setting_->alpha + vec_sigma_y[i];
                } else {
                    k_mat(i, j) = m_setting_->alpha * InlineExpr(a, m_setting_->weights, mat_x.col(i), mat_x.col(j));
                    k_mat(j, i) = k_mat(i, j);
                }
            }
        }

        return k_mat;
    }

    Eigen::MatrixXd
    CustomKernelV3::ComputeKtest(const Eigen::Ref<const Eigen::MatrixXd> &mat_x1, const Eigen::Ref<const Eigen::MatrixXd> &mat_x2) const {
        ERL_ASSERTM(mat_x1.rows() == 3, "Each column of mat_x1 should be 4D vector [m_x_, m_y_, angle].");
        ERL_ASSERTM(mat_x2.rows() == 3, "Each column of mat_x2 should be 4D vector [m_x_, m_y_, angle].");
        ERL_ASSERTM(m_setting_->weights.size() == 2, "Number of weights should be 2. Set GetSetting()->weights at first.");

        auto n = mat_x1.cols();
        auto m = mat_x2.cols();

        auto a = 1. / m_setting_->scale;
        Eigen::MatrixXd k_mat(n, m);

        for (long i = 0; i < n; ++i) {
            for (long j = 0; j < m; ++j) { k_mat(i, j) = m_setting_->alpha * InlineExpr(a, m_setting_->weights, mat_x1.col(i), mat_x2.col(j)); }
        }

        return k_mat;
    }

    Eigen::MatrixXd
    CustomKernelV3::ComputeKtrainWithGradient(const Eigen::Ref<const Eigen::MatrixXd> &, const Eigen::Ref<const Eigen::VectorXb> &) const {
        throw NotImplemented(__PRETTY_FUNCTION__);
    }

    Eigen::MatrixXd
    CustomKernelV3::ComputeKtrainWithGradient(
        const Eigen::Ref<const Eigen::MatrixXd> &,
        const Eigen::Ref<const Eigen::VectorXb> &,
        const Eigen::Ref<const Eigen::VectorXd> &,
        const Eigen::Ref<const Eigen::VectorXd> &,
        const Eigen::Ref<const Eigen::VectorXd> &) const {
        throw NotImplemented(__PRETTY_FUNCTION__);
    }

    Eigen::MatrixXd
    CustomKernelV3::ComputeKtestWithGradient(
        const Eigen::Ref<const Eigen::MatrixXd> &,
        const Eigen::Ref<const Eigen::VectorXb> &,
        const Eigen::Ref<const Eigen::MatrixXd> &) const {
        throw NotImplemented(__PRETTY_FUNCTION__);
    }

    CustomKernelV3::CustomKernelV3(std::shared_ptr<Setting> setting)
        : Covariance(std::move(setting)) {
        ERL_ASSERTM(m_setting_->type == Type::kCustomKernelV3, "setting->type should be CUSTOM_KERNEL_V3.");
        if (m_setting_->weights.size() == 0) { m_setting_->weights.setOnes(2); }
    }
}  // namespace erl::covariance
