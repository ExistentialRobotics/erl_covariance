#pragma once

#include "covariance.hpp"

namespace erl::covariance {
    /**
     * a * exp(-(||p1-p2||_{w,2}) / l), where w is the weight, p1=[x1,v1], p2=[x2,v2]
     */
    class CustomKernelV2 : public Covariance {

    public:
        static std::shared_ptr<CustomKernelV2>
        Create(std::shared_ptr<Setting> setting = nullptr);

        [[nodiscard]] Eigen::MatrixXd
        ComputeKtrain(const Eigen::Ref<const Eigen::MatrixXd> &mat_x) const final;

        [[nodiscard]] Eigen::MatrixXd
        ComputeKtrain(const Eigen::Ref<const Eigen::MatrixXd> &mat_x, const Eigen::Ref<const Eigen::VectorXd> &vec_sigma_y) const final;

        [[nodiscard]] Eigen::MatrixXd
        ComputeKtest(const Eigen::Ref<const Eigen::MatrixXd> &mat_x1, const Eigen::Ref<const Eigen::MatrixXd> &mat_x2) const final;

        [[nodiscard]] Eigen::MatrixXd
        ComputeKtrainWithGradient(const Eigen::Ref<const Eigen::MatrixXd> &mat_x, const Eigen::Ref<const Eigen::VectorXb> &vec_grad_flags) const final;

        [[nodiscard]] Eigen::MatrixXd
        ComputeKtrainWithGradient(
            const Eigen::Ref<const Eigen::MatrixXd> &mat_x,
            const Eigen::Ref<const Eigen::VectorXb> &vec_grad_flags,
            const Eigen::Ref<const Eigen::VectorXd> &vec_sigma_x,
            const Eigen::Ref<const Eigen::VectorXd> &vec_sigma_y,
            const Eigen::Ref<const Eigen::VectorXd> &vec_sigma_grad) const final;

        [[nodiscard]] Eigen::MatrixXd
        ComputeKtestWithGradient(
            const Eigen::Ref<const Eigen::MatrixXd> &mat_x1,
            const Eigen::Ref<const Eigen::VectorXb> &vec_grad1_flags,
            const Eigen::Ref<const Eigen::MatrixXd> &mat_x2) const final;

    private:
        explicit CustomKernelV2(std::shared_ptr<Setting> setting);
    };
}  // namespace erl::covariance
