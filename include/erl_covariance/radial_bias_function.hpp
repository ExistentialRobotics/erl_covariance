#pragma once

#include "covariance.hpp"

namespace erl::covariance {

    template<int Dim, typename Dtype>
    class RadialBiasFunction : public Covariance<Dtype> {
        // ref: https://en.wikipedia.org/wiki/Radial_basis_function_kernel
    public:
        using Super = Covariance<Dtype>;
        using Setting = typename Super::Setting;
        using Matrix = typename Super::Matrix;
        using Vector = typename Super::Vector;

        explicit RadialBiasFunction(std::shared_ptr<Setting> setting)
            : Super(std::move(setting)) {
            if (Dim != Eigen::Dynamic) { Super::m_setting_->x_dim = Dim; }  // set x_dim
        }

        [[nodiscard]] std::string
        GetCovarianceType() const override {
            return type_name<RadialBiasFunction>();
        }

        [[nodiscard]] std::pair<long, long>
        ComputeKtrain(const Eigen::Ref<const Matrix> &mat_x, long num_samples, Matrix &mat_k, Vector & /*vec_alpha*/) const override;

        [[nodiscard]] std::pair<long, long>
        ComputeKtrain(const Eigen::Ref<const Matrix> &mat_x, const Eigen::Ref<const Vector> &vec_var_y, long num_samples, Matrix &mat_k, Vector & /*vec_alpha*/)
            const override;

        [[nodiscard]] std::pair<long, long>
        ComputeKtest(const Eigen::Ref<const Matrix> &mat_x1, long num_samples1, const Eigen::Ref<const Matrix> &mat_x2, long num_samples2, Matrix &mat_k)
            const override;

        [[nodiscard]] std::pair<long, long>
        ComputeKtrainWithGradient(
            const Eigen::Ref<const Matrix> &mat_x,
            long num_samples,
            Eigen::VectorXl &vec_grad_flags,
            Matrix &mat_k,
            Vector & /*vec_alpha*/) const override;

        [[nodiscard]] std::pair<long, long>
        ComputeKtrainWithGradient(
            const Eigen::Ref<const Matrix> &mat_x,
            long num_samples,
            Eigen::VectorXl &vec_grad_flags,
            const Eigen::Ref<const Vector> &vec_var_x,
            const Eigen::Ref<const Vector> &vec_var_y,
            const Eigen::Ref<const Vector> &vec_var_grad,
            Matrix &mat_k,
            Vector & /*vec_alpha*/) const override;

        [[nodiscard]] std::pair<long, long>
        ComputeKtestWithGradient(
            const Eigen::Ref<const Matrix> &mat_x1,
            long num_samples1,
            const Eigen::Ref<const Eigen::VectorXl> &vec_grad1_flags,
            const Eigen::Ref<const Matrix> &mat_x2,
            long num_samples2,
            bool predict_gradient,
            Matrix &mat_k) const override;
    };

#include "radial_bias_function.tpp"

    using RadialBiasFunction1d = RadialBiasFunction<1, double>;
    using RadialBiasFunction2d = RadialBiasFunction<2, double>;
    using RadialBiasFunction3d = RadialBiasFunction<3, double>;
    using RadialBiasFunctionXd = RadialBiasFunction<Eigen::Dynamic, double>;

    using RadialBiasFunction1f = RadialBiasFunction<1, float>;
    using RadialBiasFunction2f = RadialBiasFunction<2, float>;
    using RadialBiasFunction3f = RadialBiasFunction<3, float>;
    using RadialBiasFunctionXf = RadialBiasFunction<Eigen::Dynamic, float>;

}  // namespace erl::covariance
