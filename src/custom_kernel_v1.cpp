#include "erl_covariance/custom_kernel_v1.hpp"

static double
InlineExpr(const double a, const double *weights, const double *x1, const double *x2) {
    const double d0 = x1[0] - x2[0];
    const double d1 = x1[1] - x2[1];
    const double d2 = x1[2] - x2[2];
    const double d3 = x1[3] - x2[3];
    return std::exp(-a * (weights[0] * std::sqrt(d0 * d0 + d1 * d1) + weights[1] * std::sqrt(d2 * d2 + d3 * d3)));
}

namespace erl::covariance {
    std::pair<long, long>
    CustomKernelV1::ComputeKtrain(
        const Eigen::Ref<const Eigen::MatrixXd> &mat_x,
        const long num_samples,
        Eigen::MatrixXd &mat_k,
        Eigen::VectorXd & /*vec_alpha*/) const {
        ERL_DEBUG_ASSERT(mat_x.rows() == 4, "Each column of mat_x should be 4D vector [x, y, vx, vy].");
        ERL_DEBUG_ASSERT(m_setting_->weights.size() == 2, "Number of weights should be 2. Set GetSetting()->weights at first.");
        ERL_DEBUG_ASSERT(mat_k.rows() >= num_samples, "k_mat.rows() = {}, it should be >= {}.", mat_k.rows(), num_samples);
        ERL_DEBUG_ASSERT(mat_k.cols() >= num_samples, "k_mat.cols() = {}, it should be >= {}.", mat_k.cols(), num_samples);

        const double a = 1. / m_setting_->scale;
        const double alpha = m_setting_->alpha;
        const double *weights = m_setting_->weights.data();
        for (long i = 0; i < num_samples; ++i) {
            for (long j = i; j < num_samples; ++j) {
                if (i == j) {
                    mat_k(i, i) = alpha;
                } else {
                    mat_k(i, j) = alpha * InlineExpr(a, weights, mat_x.col(i).data(), mat_x.col(j).data());
                    mat_k(j, i) = mat_k(i, j);
                }
            }
        }
        return {num_samples, num_samples};
    }

    std::pair<long, long>
    CustomKernelV1::ComputeKtrain(
        const Eigen::Ref<const Eigen::MatrixXd> &mat_x,
        const Eigen::Ref<const Eigen::VectorXd> &vec_var_y,
        const long num_samples,
        Eigen::MatrixXd &k_mat,
        Eigen::VectorXd & /*vec_alpha*/) const {
        ERL_DEBUG_ASSERT(mat_x.rows() == 4, "Each column of mat_x should be 4D vector [x, y, vx, vy].");
        ERL_DEBUG_ASSERT(m_setting_->weights.size() == 2, "Number of weights should be 2. Set GetSetting()->weights at first.");
        ERL_DEBUG_ASSERT(k_mat.rows() >= num_samples, "k_mat.rows() = {}, it should be >= {}.", k_mat.rows(), num_samples);
        ERL_DEBUG_ASSERT(k_mat.cols() >= num_samples, "k_mat.cols() = {}, it should be >= {}.", k_mat.cols(), num_samples);
        ERL_DEBUG_ASSERT(vec_var_y.size() >= num_samples, "vec_var_y does not have enough elements, it should be >= {}.", num_samples);

        const double a = 1. / m_setting_->scale;
        const double alpha = m_setting_->alpha;
        const double *weights = m_setting_->weights.data();
        for (long i = 0; i < num_samples; ++i) {
            for (long j = i; j < num_samples; ++j) {
                if (i == j) {
                    k_mat(i, i) = alpha + vec_var_y[i];
                } else {
                    k_mat(i, j) = alpha * InlineExpr(a, weights, mat_x.col(i).data(), mat_x.col(j).data());
                    k_mat(j, i) = k_mat(i, j);
                }
            }
        }
        return {num_samples, num_samples};
    }

    std::pair<long, long>
    CustomKernelV1::ComputeKtest(
        const Eigen::Ref<const Eigen::MatrixXd> &mat_x1,
        const long num_samples1,
        const Eigen::Ref<const Eigen::MatrixXd> &mat_x2,
        const long num_samples2,
        Eigen::MatrixXd &k_mat) const {
        ERL_DEBUG_ASSERT(mat_x1.rows() == 4, "Each column of mat_x1 should be 4D vector [x, y, vx, vy].");
        ERL_DEBUG_ASSERT(mat_x2.rows() == 4, "Each column of mat_x2 should be 4D vector [x, y, vx, vy].");
        ERL_DEBUG_ASSERT(m_setting_->weights.size() == 2, "Number of weights should be 2. Set GetSetting()->weights at first.");
        ERL_DEBUG_ASSERT(k_mat.rows() >= num_samples1, "k_mat.rows() = {}, it should be >= {}.", k_mat.rows(), num_samples1);
        ERL_DEBUG_ASSERT(k_mat.cols() >= num_samples2, "k_mat.cols() = {}, it should be >= {}.", k_mat.cols(), num_samples2);

        const double a = 1. / m_setting_->scale;
        const double alpha = m_setting_->alpha;
        const double *weights = m_setting_->weights.data();
        for (long i = 0; i < num_samples1; ++i) {
            for (long j = 0; j < num_samples2; ++j) { k_mat(i, j) = alpha * InlineExpr(a, weights, mat_x1.col(i).data(), mat_x2.col(j).data()); }
        }
        return {num_samples1, num_samples2};
    }

    std::pair<long, long>
    CustomKernelV1::ComputeKtrainWithGradient(const Eigen::Ref<const Eigen::MatrixXd> &, long, Eigen::VectorXl &, Eigen::MatrixXd &, Eigen::VectorXd &) const {
        throw NotImplemented(__PRETTY_FUNCTION__);
    }

    std::pair<long, long>
    CustomKernelV1::ComputeKtrainWithGradient(
        const Eigen::Ref<const Eigen::MatrixXd> &,
        long,
        Eigen::VectorXl &,
        const Eigen::Ref<const Eigen::VectorXd> &,
        const Eigen::Ref<const Eigen::VectorXd> &,
        const Eigen::Ref<const Eigen::VectorXd> &,
        Eigen::MatrixXd &,
        Eigen::VectorXd &) const {
        throw NotImplemented(__PRETTY_FUNCTION__);
    }

    std::pair<long, long>
    CustomKernelV1::ComputeKtestWithGradient(
        const Eigen::Ref<const Eigen::MatrixXd> &,
        long,
        const Eigen::Ref<const Eigen::VectorXl> &,
        const Eigen::Ref<const Eigen::MatrixXd> &,
        long,
        Eigen::MatrixXd &) const {
        throw NotImplemented(__PRETTY_FUNCTION__);
    }
}  // namespace erl::covariance
