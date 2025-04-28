#pragma once

#include "covariance.hpp"

namespace erl::covariance {

    template<typename Dtype, int /*Dim*/>
    class RadialBiasFunction : public Covariance<Dtype> {
        // ref: https://en.wikipedia.org/wiki/Radial_basis_function_kernel
    public:
        using Super = Covariance<Dtype>;
        using Setting = typename Super::Setting;
        using MatrixX = Eigen::MatrixX<Dtype>;
        using SparseMatrix = Eigen::SparseMatrix<Dtype>;
        using VectorX = Eigen::VectorX<Dtype>;

        explicit RadialBiasFunction(std::shared_ptr<Setting> setting);

        [[nodiscard]] std::string
        GetCovarianceType() const override;

        [[nodiscard]] std::pair<long, long>
        ComputeKtrain(const Eigen::Ref<const MatrixX> &mat_x, long num_samples, MatrixX &mat_k, MatrixX & /*mat_alpha*/) override;

        [[nodiscard]] std::pair<long, long>
        ComputeKtrain(const Eigen::Ref<const MatrixX> &mat_x, long num_samples, MatrixX &mat_k);

        [[nodiscard]] std::pair<long, long>
        ComputeKtrain(
            const Eigen::Ref<const MatrixX> &mat_x,
            const Eigen::Ref<const VectorX> &vec_var_y,
            long num_samples,
            MatrixX &mat_k,
            MatrixX & /*mat_alpha*/) override;

        [[nodiscard]] std::pair<long, long>
        ComputeKtrain(const Eigen::Ref<const MatrixX> &mat_x, const Eigen::Ref<const VectorX> &vec_var_y, long num_samples, MatrixX &mat_k);

        [[nodiscard]] std::pair<long, long>
        ComputeKtest(const Eigen::Ref<const MatrixX> &mat_x1, long num_samples1, const Eigen::Ref<const MatrixX> &mat_x2, long num_samples2, MatrixX &mat_k)
            const override;

        [[nodiscard]] std::pair<long, long>
        ComputeKtestSparse(
            const Eigen::Ref<const MatrixX> &mat_x1,
            long num_samples1,
            const Eigen::Ref<const MatrixX> &mat_x2,
            long num_samples2,
            Dtype zero_threshold,
            SparseMatrix &mat_k) const override;

        [[nodiscard]] std::pair<long, long>
        ComputeKtrainWithGradient(
            const Eigen::Ref<const MatrixX> &mat_x,
            long num_samples,
            Eigen::VectorXl &vec_grad_flags,
            MatrixX &mat_k,
            MatrixX & /*mat_alpha*/) override;

        [[nodiscard]] std::pair<long, long>
        ComputeKtrainWithGradient(const Eigen::Ref<const MatrixX> &mat_x, long num_samples, Eigen::VectorXl &vec_grad_flags, MatrixX &mat_k);

        [[nodiscard]] std::pair<long, long>
        ComputeKtrainWithGradient(
            const Eigen::Ref<const MatrixX> &mat_x,
            long num_samples,
            Eigen::VectorXl &vec_grad_flags,
            const Eigen::Ref<const VectorX> &vec_var_x,
            const Eigen::Ref<const VectorX> &vec_var_y,
            const Eigen::Ref<const VectorX> &vec_var_grad,
            MatrixX &mat_k,
            MatrixX & /*mat_alpha*/) override;

        [[nodiscard]] std::pair<long, long>
        ComputeKtrainWithGradient(
            const Eigen::Ref<const MatrixX> &mat_x,
            long num_samples,
            Eigen::VectorXl &vec_grad_flags,
            const Eigen::Ref<const VectorX> &vec_var_x,
            const Eigen::Ref<const VectorX> &vec_var_y,
            const Eigen::Ref<const VectorX> &vec_var_grad,
            MatrixX &mat_k);

        [[nodiscard]] std::pair<long, long>
        ComputeKtestWithGradient(
            const Eigen::Ref<const MatrixX> &mat_x1,
            long num_samples1,
            const Eigen::Ref<const Eigen::VectorXl> &vec_grad1_flags,
            const Eigen::Ref<const MatrixX> &mat_x2,
            long num_samples2,
            bool predict_gradient,
            MatrixX &mat_k) const override;

        [[nodiscard]] std::pair<long, long>
        ComputeKtestWithGradientSparse(
            const Eigen::Ref<const MatrixX> &mat_x1,
            long num_samples1,
            const Eigen::Ref<const Eigen::VectorXl> &vec_grad1_flags,
            const Eigen::Ref<const MatrixX> &mat_x2,
            long num_samples2,
            bool predict_gradient,
            Dtype zero_threshold,
            SparseMatrix &mat_k) const override;
    };

    using RadialBiasFunction1d = RadialBiasFunction<double, 1>;
    using RadialBiasFunction2d = RadialBiasFunction<double, 2>;
    using RadialBiasFunction3d = RadialBiasFunction<double, 3>;
    using RadialBiasFunctionXd = RadialBiasFunction<double, Eigen::Dynamic>;

    using RadialBiasFunction1f = RadialBiasFunction<float, 1>;
    using RadialBiasFunction2f = RadialBiasFunction<float, 2>;
    using RadialBiasFunction3f = RadialBiasFunction<float, 3>;
    using RadialBiasFunctionXf = RadialBiasFunction<float, Eigen::Dynamic>;

}  // namespace erl::covariance

#include "radial_bias_function.tpp"
