#pragma once

#include "covariance.hpp"

namespace erl::covariance {
    /**
     * a * exp(-(w1 * ||x1-x2||_2 + w2 * |theta1 - theta2|) / l)
     */
    class CustomKernelV3 : public Covariance {

    public:
        static std::shared_ptr<CustomKernelV3>
        Create(std::shared_ptr<Setting> setting = nullptr);

        [[nodiscard]] std::pair<long, long>
        ComputeKtrain(Eigen::Ref<Eigen::MatrixXd> k_mat, const Eigen::Ref<const Eigen::MatrixXd> &mat_x) const final;

        [[nodiscard]] std::pair<long, long>
        ComputeKtrain(Eigen::Ref<Eigen::MatrixXd> k_mat, const Eigen::Ref<const Eigen::MatrixXd> &mat_x, const Eigen::Ref<const Eigen::VectorXd> &vec_sigma_y)
            const final;

        [[nodiscard]] std::pair<long, long>
        ComputeKtest(Eigen::Ref<Eigen::MatrixXd> k_mat, const Eigen::Ref<const Eigen::MatrixXd> &mat_x1, const Eigen::Ref<const Eigen::MatrixXd> &mat_x2)
            const final;

        [[nodiscard]] std::pair<long, long>
        ComputeKtrainWithGradient(
            Eigen::Ref<Eigen::MatrixXd> k_mat,
            const Eigen::Ref<const Eigen::MatrixXd> &mat_x,
            const Eigen::Ref<const Eigen::VectorXb> &vec_grad_flags) const final;

        [[nodiscard]] std::pair<long, long>
        ComputeKtrainWithGradient(
            Eigen::Ref<Eigen::MatrixXd> k_mat,
            const Eigen::Ref<const Eigen::MatrixXd> &mat_x,
            const Eigen::Ref<const Eigen::VectorXb> &vec_grad_flags,
            const Eigen::Ref<const Eigen::VectorXd> &vec_sigma_x,
            const Eigen::Ref<const Eigen::VectorXd> &vec_sigma_y,
            const Eigen::Ref<const Eigen::VectorXd> &vec_sigma_grad) const final;

        [[nodiscard]] std::pair<long, long>
        ComputeKtestWithGradient(
            Eigen::Ref<Eigen::MatrixXd> k_mat,
            const Eigen::Ref<const Eigen::MatrixXd> &mat_x1,
            const Eigen::Ref<const Eigen::VectorXb> &vec_grad1_flags,
            const Eigen::Ref<const Eigen::MatrixXd> &mat_x2) const final;

    private:
        explicit CustomKernelV3(std::shared_ptr<Setting> setting);
    };
}  // namespace erl::covariance
