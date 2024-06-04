#include "erl_covariance/custom_kernel_v2.hpp"

static double
InlineExpr(const double a, const Eigen::VectorXd &weights, const Eigen::Ref<const Eigen::Vector4d> &x_1, const Eigen::Ref<const Eigen::Vector4d> &x_2) {
    const double d0 = x_1[0] - x_2[0];
    const double d1 = x_1[1] - x_2[1];
    const double d2 = x_1[2] - x_2[2];
    const double d3 = x_1[3] - x_2[3];
    return std::exp(-a * std::sqrt(weights[0] * d0 * d0 + weights[1] * d1 * d1 + weights[2] * d2 * d2 + weights[3] * d3 * d3));
}

namespace erl::covariance {
    std::pair<long, long>
    CustomKernelV2::ComputeKtrain(Eigen::Ref<Eigen::MatrixXd> k_mat, const Eigen::Ref<const Eigen::MatrixXd> &mat_x) const {
        ERL_DEBUG_ASSERT(mat_x.rows() == 4, "Each column of mat_x should be 4D vector [x, y, vx, vy].");
        ERL_DEBUG_ASSERT(m_setting_->weights.size() == 4, "Number of weights should be 4. Set GetSetting()->weights at first.");
        long n = mat_x.cols();
        ERL_DEBUG_ASSERT(k_mat.rows() >= n, "k_mat.rows() = {}, it should be >= {}.", k_mat.rows(), n);
        ERL_DEBUG_ASSERT(k_mat.cols() >= n, "k_mat.cols() = {}, it should be >= {}.", k_mat.cols(), n);

        const double a = 1. / m_setting_->scale;
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
    CustomKernelV2::ComputeKtrain(
        Eigen::Ref<Eigen::MatrixXd> k_mat,
        const Eigen::Ref<const Eigen::MatrixXd> &mat_x,
        const Eigen::Ref<const Eigen::VectorXd> &vec_var_y) const {
        ERL_DEBUG_ASSERT(mat_x.rows() == 4, "Each column of mat_x should be 4D vector [x, y, vx, vy].");
        ERL_DEBUG_ASSERT(m_setting_->weights.size() == 4, "Number of weights should be 4. Set GetSetting()->weights at first.");
        long n = mat_x.cols();
        ERL_DEBUG_ASSERT(k_mat.rows() >= n, "k_mat.rows() = {}, it should be >= {}.", k_mat.rows(), n);
        ERL_DEBUG_ASSERT(k_mat.cols() >= n, "k_mat.cols() = {}, it should be >= {}.", k_mat.cols(), n);
        ERL_DEBUG_ASSERT(n == vec_var_y.size(), "#elements of vec_sigma_y does not equal to #columns of mat_x.");

        const double a = 1. / m_setting_->scale;
        for (long i = 0; i < n; ++i) {
            for (long j = i; j < n; ++j) {
                if (i == j) {
                    k_mat(i, i) = m_setting_->alpha + vec_var_y[i];
                } else {
                    k_mat(i, j) = m_setting_->alpha * InlineExpr(a, m_setting_->weights, mat_x.col(i), mat_x.col(j));
                    k_mat(j, i) = k_mat(i, j);
                }
            }
        }
        return {n, n};
    }

    std::pair<long, long>
    CustomKernelV2::ComputeKtest(
        Eigen::Ref<Eigen::MatrixXd> k_mat,
        const Eigen::Ref<const Eigen::MatrixXd> &mat_x1,
        const Eigen::Ref<const Eigen::MatrixXd> &mat_x2) const {
        ERL_DEBUG_ASSERT(mat_x1.rows() == 4, "Each column of mat_x1 should be 4D vector [x, y, vx, vy].");
        ERL_DEBUG_ASSERT(mat_x2.rows() == 4, "Each column of mat_x2 should be 4D vector [x, y, vx, vy].");
        ERL_DEBUG_ASSERT(m_setting_->weights.size() == 4, "Number of weights should be 4. Set GetSetting()->weights at first.");

        long n = mat_x1.cols();
        long m = mat_x2.cols();
        ERL_DEBUG_ASSERT(k_mat.rows() >= n, "k_mat.rows() = {}, it should be >= {}.", k_mat.rows(), n);
        ERL_DEBUG_ASSERT(k_mat.cols() >= m, "k_mat.cols() = {}, it should be >= {}.", k_mat.cols(), m);

        const double a = 1. / m_setting_->scale;
        for (long i = 0; i < n; ++i) {
            for (long j = 0; j < m; ++j) { k_mat(i, j) = m_setting_->alpha * InlineExpr(a, m_setting_->weights, mat_x1.col(i), mat_x2.col(j)); }
        }
        return {n, m};
    }

    std::pair<long, long>
    CustomKernelV2::ComputeKtrainWithGradient(Eigen::Ref<Eigen::MatrixXd>, const Eigen::Ref<const Eigen::MatrixXd> &, const Eigen::Ref<const Eigen::VectorXb> &)
        const {
        throw NotImplemented(__PRETTY_FUNCTION__);
    }

    std::pair<long, long>
    CustomKernelV2::ComputeKtrainWithGradient(
        Eigen::Ref<Eigen::MatrixXd>,
        const Eigen::Ref<const Eigen::MatrixXd> &,
        const Eigen::Ref<const Eigen::VectorXb> &,
        const Eigen::Ref<const Eigen::VectorXd> &,
        const Eigen::Ref<const Eigen::VectorXd> &,
        const Eigen::Ref<const Eigen::VectorXd> &) const {
        throw NotImplemented(__PRETTY_FUNCTION__);
    }

    std::pair<long, long>
    CustomKernelV2::ComputeKtestWithGradient(
        Eigen::Ref<Eigen::MatrixXd>,
        const Eigen::Ref<const Eigen::MatrixXd> &,
        const Eigen::Ref<const Eigen::VectorXb> &,
        const Eigen::Ref<const Eigen::MatrixXd> &) const {
        throw NotImplemented(__PRETTY_FUNCTION__);
    }
}  // namespace erl::covariance
