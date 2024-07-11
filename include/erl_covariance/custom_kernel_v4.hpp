#pragma once

#include "covariance.hpp"

namespace erl::covariance {
    /**
     * a * exp(-(||p1-p2||_{w,2}) / l), where w is the weights, p1=[x1, theta1], p2=[x2, theta2]
     */
    class CustomKernelV4 : public Covariance {

    public:
        [[nodiscard]] std::shared_ptr<Covariance>
        Create(std::shared_ptr<Setting> setting) const override {
            if (setting == nullptr) { setting = std::make_shared<Setting>(); }
            return std::make_shared<CustomKernelV4>(std::move(setting));
        }

        explicit CustomKernelV4(std::shared_ptr<Setting> setting)
            : Covariance(std::move(setting)) {
            if (m_setting_->weights.size() == 0) { m_setting_->weights.setOnes(3); }
            m_setting_->x_dim = 3;
        }

        [[nodiscard]] std::string
        GetCovarianceType() const override {
            return "CustomKernelV4";
        }

        [[nodiscard]] std::pair<long, long>
        ComputeKtrain(const Eigen::Ref<const Eigen::MatrixXd> &mat_x, long num_samples, Eigen::MatrixXd &k_mat) const final;

        [[nodiscard]] std::pair<long, long>
        ComputeKtrain(
            const Eigen::Ref<const Eigen::MatrixXd> &mat_x,
            const Eigen::Ref<const Eigen::VectorXd> &vec_var_y,
            long num_samples,
            Eigen::MatrixXd &k_mat) const final;

        [[nodiscard]] std::pair<long, long>
        ComputeKtest(
            const Eigen::Ref<const Eigen::MatrixXd> &mat_x1,
            long num_samples1,
            const Eigen::Ref<const Eigen::MatrixXd> &mat_x2,
            long num_samples2,
            Eigen::MatrixXd &k_mat) const final;

        [[nodiscard]] std::pair<long, long>
        ComputeKtrainWithGradient(const Eigen::Ref<const Eigen::MatrixXd> &mat_x, long num_samples, Eigen::VectorXl &vec_grad_flags, Eigen::MatrixXd &k_mat)
            const final;

        [[nodiscard]] std::pair<long, long>
        ComputeKtrainWithGradient(
            const Eigen::Ref<const Eigen::MatrixXd> &mat_x,
            long num_samples,
            Eigen::VectorXl &vec_grad_flags,
            const Eigen::Ref<const Eigen::VectorXd> &vec_var_x,
            const Eigen::Ref<const Eigen::VectorXd> &vec_var_y,
            const Eigen::Ref<const Eigen::VectorXd> &vec_var_grad,
            Eigen::MatrixXd &k_mat) const final;

        [[nodiscard]] std::pair<long, long>
        ComputeKtestWithGradient(
            const Eigen::Ref<const Eigen::MatrixXd> &mat_x1,
            long num_samples1,
            const Eigen::Ref<const Eigen::VectorXl> &vec_grad1_flags,
            const Eigen::Ref<const Eigen::MatrixXd> &mat_x2,
            long num_samples2,
            Eigen::MatrixXd &k_mat) const final;
    };

    ERL_REGISTER_COVARIANCE(CustomKernelV4);
}  // namespace erl::covariance
