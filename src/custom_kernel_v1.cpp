#include "erl_covariance/custom_kernel_v1.hpp"

static inline double
InlineExpr(const double &a, const Eigen::VectorXd &weights, const Eigen::Ref<const Eigen::Vector4d> &x_1, const Eigen::Ref<const Eigen::Vector4d> &x_2) {
    Eigen::Vector4d squared_diff = (x_1 - x_2).array().square();
    return std::exp(-a * (weights.x() * std::sqrt(squared_diff.x() + squared_diff.y()) + weights.y() * std::sqrt(squared_diff.z() + squared_diff.w())));
}

namespace erl::covariance {
    std::shared_ptr<CustomKernelV1>
    CustomKernelV1::Create(std::shared_ptr<Setting> setting) {
        if (setting == nullptr) {
            setting = std::make_shared<CustomKernelV1::Setting>();
            setting->type = Type::kCustomKernelV1;
        }
        return std::shared_ptr<CustomKernelV1>(new CustomKernelV1(std::move(setting)));
    }

    std::pair<long, long>
    CustomKernelV1::ComputeKtrain(Eigen::Ref<Eigen::MatrixXd> k_mat, const Eigen::Ref<const Eigen::MatrixXd> &mat_x) const {
        ERL_ASSERTM(mat_x.rows() == 4, "Each column of mat_x should be 4D vector [x, y, vx, vy].");
        ERL_ASSERTM(m_setting_->weights.size() == 2, "Number of weights should be 2. Set GetSetting()->weights at first.");
        long n = mat_x.cols();
        ERL_ASSERTM(k_mat.rows() >= n, "k_mat.rows() = %ld, it should be >= %ld.", k_mat.rows(), n);
        ERL_ASSERTM(k_mat.cols() >= n, "k_mat.cols() = %ld, it should be >= %ld.", k_mat.cols(), n);

        double a = 1. / m_setting_->scale;
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
        return {n, n};
    }

    std::pair<long, long>
    CustomKernelV1::ComputeKtrain(
        Eigen::Ref<Eigen::MatrixXd> k_mat,
        const Eigen::Ref<const Eigen::MatrixXd> &mat_x,
        const Eigen::Ref<const Eigen::VectorXd> &vec_sigma_y) const {
        ERL_ASSERTM(mat_x.rows() == 4, "Each column of mat_x should be 4D vector [x, y, vx, vy].");
        ERL_ASSERTM(m_setting_->weights.size() == 2, "Number of weights should be 2. Set GetSetting()->weights at first.");
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
                    k_mat(i, j) = m_setting_->alpha * InlineExpr(a, m_setting_->weights, mat_x.col(i), mat_x.col(j));
                    k_mat(j, i) = k_mat(i, j);
                }
            }
        }
        return {n, n};
    }

    std::pair<long, long>
    CustomKernelV1::ComputeKtest(
        Eigen::Ref<Eigen::MatrixXd> k_mat,
        const Eigen::Ref<const Eigen::MatrixXd> &mat_x1,
        const Eigen::Ref<const Eigen::MatrixXd> &mat_x2) const {
        ERL_ASSERTM(mat_x1.rows() == 4, "Each column of mat_x1 should be 4D vector [x, y, vx, vy].");
        ERL_ASSERTM(mat_x2.rows() == 4, "Each column of mat_x2 should be 4D vector [x, y, vx, vy].");
        ERL_ASSERTM(m_setting_->weights.size() == 2, "Number of weights should be 2. Set GetSetting()->weights at first.");

        long n = mat_x1.cols();
        long m = mat_x2.cols();
        ERL_ASSERTM(k_mat.rows() >= n, "k_mat.rows() = %ld, it should be >= %ld.", k_mat.rows(), n);
        ERL_ASSERTM(k_mat.cols() >= m, "k_mat.cols() = %ld, it should be >= %ld.", k_mat.cols(), m);

        double a = 1. / m_setting_->scale;
        for (long i = 0; i < n; ++i) {
            for (long j = 0; j < m; ++j) { k_mat(i, j) = m_setting_->alpha * InlineExpr(a, m_setting_->weights, mat_x1.col(i), mat_x2.col(j)); }
        }
        return {n, m};
    }

    std::pair<long, long>
    CustomKernelV1::ComputeKtrainWithGradient(Eigen::Ref<Eigen::MatrixXd>, const Eigen::Ref<const Eigen::MatrixXd> &, const Eigen::Ref<const Eigen::VectorXb> &)
        const {
        throw NotImplemented(__PRETTY_FUNCTION__);
    }

    std::pair<long, long>
    CustomKernelV1::ComputeKtrainWithGradient(
        Eigen::Ref<Eigen::MatrixXd>,
        const Eigen::Ref<const Eigen::MatrixXd> &,
        const Eigen::Ref<const Eigen::VectorXb> &,
        const Eigen::Ref<const Eigen::VectorXd> &,
        const Eigen::Ref<const Eigen::VectorXd> &,
        const Eigen::Ref<const Eigen::VectorXd> &) const {
        throw NotImplemented(__PRETTY_FUNCTION__);
    }

    std::pair<long, long>
    CustomKernelV1::ComputeKtestWithGradient(
        Eigen::Ref<Eigen::MatrixXd>,
        const Eigen::Ref<const Eigen::MatrixXd> &,
        const Eigen::Ref<const Eigen::VectorXb> &,
        const Eigen::Ref<const Eigen::MatrixXd> &) const {
        throw NotImplemented(__PRETTY_FUNCTION__);
    }

    CustomKernelV1::CustomKernelV1(std::shared_ptr<Setting> setting)
        : Covariance(std::move(setting)) {
        ERL_ASSERTM(m_setting_->type == Type::kCustomKernelV1, "setting->type should be kCustomKernelV1.");
        if (m_setting_->weights.size() == 0) { m_setting_->weights.setOnes(2); }
    }
}  // namespace erl::covariance
