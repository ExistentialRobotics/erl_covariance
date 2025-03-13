#pragma once

#include "covariance.hpp"

namespace erl::covariance {

    template<typename Dtype, int Dim>
    class Matern32 : public Covariance<Dtype> {
    public:
        using Super = Covariance<Dtype>;
        using Setting = typename Super::Setting;
        using MatrixX = Eigen::MatrixX<Dtype>;
        using VectorX = Eigen::VectorX<Dtype>;

        explicit Matern32(std::shared_ptr<Setting> setting)
            : Super(std::move(setting)) {
            if (Dim != Eigen::Dynamic) { Super::m_setting_->x_dim = Dim; }  // set x_dim
        }

        [[nodiscard]] std::string
        GetCovarianceType() const override {
            return type_name<Matern32>();
        }

        [[nodiscard]] std::pair<long, long>
        ComputeKtrain(const Eigen::Ref<const MatrixX> &mat_x, long num_samples, MatrixX &mat_k, VectorX & /*vec_alpha*/) const override;

        [[nodiscard]] std::pair<long, long>
        ComputeKtrain(
            const Eigen::Ref<const MatrixX> &mat_x,
            const Eigen::Ref<const VectorX> &vec_var_y,
            long num_samples,
            MatrixX &mat_k,
            VectorX & /*vec_alpha*/) const override;

        [[nodiscard]] std::pair<long, long>
        ComputeKtest(const Eigen::Ref<const MatrixX> &mat_x1, long num_samples1, const Eigen::Ref<const MatrixX> &mat_x2, long num_samples2, MatrixX &mat_k)
            const override;

        [[nodiscard]] std::pair<long, long>
        ComputeKtrainWithGradient(
            const Eigen::Ref<const MatrixX> &mat_x,
            long num_samples,
            Eigen::VectorXl &vec_grad_flags,
            MatrixX &mat_k,
            VectorX & /*vec_alpha*/) const override;

        [[nodiscard]] std::pair<long, long>
        ComputeKtrainWithGradient(
            const Eigen::Ref<const MatrixX> &mat_x,
            long num_samples,
            Eigen::VectorXl &vec_grad_flags,
            const Eigen::Ref<const VectorX> &vec_var_x,
            const Eigen::Ref<const VectorX> &vec_var_y,
            const Eigen::Ref<const VectorX> &vec_var_grad,
            MatrixX &mat_k,
            VectorX & /*vec_alpha*/) const override;

        [[nodiscard]] std::pair<long, long>
        ComputeKtestWithGradient(
            const Eigen::Ref<const MatrixX> &mat_x1,
            long num_samples1,
            const Eigen::Ref<const Eigen::VectorXl> &vec_grad1_flags,
            const Eigen::Ref<const MatrixX> &mat_x2,
            long num_samples2,
            bool predict_gradient,
            MatrixX &mat_k) const override;
    };

    using Matern32_1d = Matern32<double, 1>;
    using Matern32_2d = Matern32<double, 2>;
    using Matern32_3d = Matern32<double, 3>;
    using Matern32_Xd = Matern32<double, Eigen::Dynamic>;

    using Matern32_1f = Matern32<float, 1>;
    using Matern32_2f = Matern32<float, 2>;
    using Matern32_3f = Matern32<float, 3>;
    using Matern32_Xf = Matern32<float, Eigen::Dynamic>;

}  // namespace erl::covariance

#include "matern32.tpp"
