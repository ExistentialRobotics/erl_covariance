#pragma once

#include "covariance.hpp"

namespace erl::covariance {
    /**
     * a * exp(-(w1 * ||x1-x2||_2 + w2 * |theta1 - theta2|) / l)
     */
    class CustomKernelV3 : public Covariance {

    public:
        explicit CustomKernelV3(std::shared_ptr<Setting> setting)
            : Covariance(std::move(setting)) {
            if (m_setting_->weights.size() == 0) { m_setting_->weights.setOnes(2); }
            m_setting_->x_dim = 3;
        }

        [[nodiscard]] std::string
        GetCovarianceType() const override {
            return "CustomKernelV3";
        }

        [[nodiscard]] std::pair<long, long>
        ComputeKtrain(const Eigen::Ref<const Eigen::MatrixXd> &mat_x, long num_samples, Eigen::MatrixXd &mat_k, Eigen::VectorXd &vec_alpha) const override;

        [[nodiscard]] std::pair<long, long>
        ComputeKtrain(
            const Eigen::Ref<const Eigen::MatrixXd> &mat_x,
            const Eigen::Ref<const Eigen::VectorXd> &vec_var_y,
            long num_samples,
            Eigen::MatrixXd &mat_k,
            Eigen::VectorXd &vec_alpha) const override;

        [[nodiscard]] std::pair<long, long>
        ComputeKtest(
            const Eigen::Ref<const Eigen::MatrixXd> &mat_x1,
            long num_samples1,
            const Eigen::Ref<const Eigen::MatrixXd> &mat_x2,
            long num_samples2,
            Eigen::MatrixXd &mat_k) const override;

        [[nodiscard]] std::pair<long, long>
        ComputeKtrainWithGradient(
            const Eigen::Ref<const Eigen::MatrixXd> &mat_x,
            long num_samples,
            Eigen::VectorXl &vec_grad_flags,
            Eigen::MatrixXd &mat_k,
            Eigen::VectorXd &vec_alpha) const override;

        [[nodiscard]] std::pair<long, long>
        ComputeKtrainWithGradient(
            const Eigen::Ref<const Eigen::MatrixXd> &mat_x,
            long num_samples,
            Eigen::VectorXl &vec_grad_flags,
            const Eigen::Ref<const Eigen::VectorXd> &vec_var_x,
            const Eigen::Ref<const Eigen::VectorXd> &vec_var_y,
            const Eigen::Ref<const Eigen::VectorXd> &vec_var_grad,
            Eigen::MatrixXd &mat_k,
            Eigen::VectorXd &vec_alpha) const override;

        [[nodiscard]] std::pair<long, long>
        ComputeKtestWithGradient(
            const Eigen::Ref<const Eigen::MatrixXd> &mat_x1,
            long num_samples1,
            const Eigen::Ref<const Eigen::VectorXl> &vec_grad1_flags,
            const Eigen::Ref<const Eigen::MatrixXd> &mat_x2,
            long num_samples2,
            Eigen::MatrixXd &mat_k) const override;
    };

    ERL_REGISTER_COVARIANCE(CustomKernelV3);
}  // namespace erl::covariance
