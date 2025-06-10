#pragma once

#include "covariance.hpp"

namespace erl::covariance {

    template<typename Dtype, int /*Dim*/>
    class RationalQuadratic : public Covariance<Dtype> {
        // ref:
        // https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.RationalQuadratic.html
    public:
        using Super = Covariance<Dtype>;
        using Setting = typename Super::Setting;
        using MatrixX = Eigen::MatrixX<Dtype>;
        using VectorX = Eigen::VectorX<Dtype>;

        explicit RationalQuadratic(std::shared_ptr<Setting> setting);

        [[nodiscard]] std::string
        GetCovarianceType() const override;

        [[nodiscard]] std::pair<long, long>
        ComputeKtrain(
            const Eigen::Ref<const MatrixX> &mat_x,
            long num_samples,
            MatrixX &k_mat,
            MatrixX & /*mat_alpha*/) override;

        [[nodiscard]] std::pair<long, long>
        ComputeKtrain(const Eigen::Ref<const MatrixX> &mat_x, long num_samples, MatrixX &mat_k);

        [[nodiscard]] std::pair<long, long>
        ComputeKtrain(
            const Eigen::Ref<const MatrixX> &mat_x,
            const Eigen::Ref<const VectorX> &vec_var_y,
            long num_samples,
            MatrixX &k_mat,
            MatrixX & /*mat_alpha*/) override;

        [[nodiscard]] std::pair<long, long>
        ComputeKtrain(
            const Eigen::Ref<const MatrixX> &mat_x,
            const Eigen::Ref<const VectorX> &vec_var_y,
            long num_samples,
            MatrixX &mat_k);

        [[nodiscard]] std::pair<long, long>
        ComputeKtest(
            const Eigen::Ref<const MatrixX> &mat_x1,
            long num_samples1,
            const Eigen::Ref<const MatrixX> &mat_x2,
            long num_samples2,
            MatrixX &k_mat) const override;

        [[nodiscard]] std::pair<long, long>
        ComputeKtrainWithGradient(
            const Eigen::Ref<const MatrixX> &mat_x,
            long num_samples,
            Eigen::VectorXl &vec_grad_flags,
            MatrixX &k_mat,
            MatrixX & /*mat_alpha*/) override;

        [[nodiscard]] std::pair<long, long>
        ComputeKtrainWithGradient(
            const Eigen::Ref<const MatrixX> &mat_x,
            long num_samples,
            Eigen::VectorXl &vec_grad_flags,
            MatrixX &mat_k);

        [[nodiscard]] std::pair<long, long>
        ComputeKtrainWithGradient(
            const Eigen::Ref<const MatrixX> &mat_x,
            long num_samples,
            Eigen::VectorXl &vec_grad_flags,
            const Eigen::Ref<const VectorX> &vec_var_x,
            const Eigen::Ref<const VectorX> &vec_var_y,
            const Eigen::Ref<const VectorX> &vec_var_grad,
            MatrixX &k_mat,
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
            MatrixX &k_mat) const override;
    };

    using RationalQuadratic1d = RationalQuadratic<double, 1>;
    using RationalQuadratic2d = RationalQuadratic<double, 2>;
    using RationalQuadratic3d = RationalQuadratic<double, 3>;
    using RationalQuadraticXd = RationalQuadratic<double, Eigen::Dynamic>;

    using RationalQuadratic1f = RationalQuadratic<float, 1>;
    using RationalQuadratic2f = RationalQuadratic<float, 2>;
    using RationalQuadratic3f = RationalQuadratic<float, 3>;
    using RationalQuadraticXf = RationalQuadratic<float, Eigen::Dynamic>;

    extern template class RationalQuadratic<double, 1>;
    extern template class RationalQuadratic<double, 2>;
    extern template class RationalQuadratic<double, 3>;
    extern template class RationalQuadratic<double, Eigen::Dynamic>;

    extern template class RationalQuadratic<float, 1>;
    extern template class RationalQuadratic<float, 2>;
    extern template class RationalQuadratic<float, 3>;
    extern template class RationalQuadratic<float, Eigen::Dynamic>;

}  // namespace erl::covariance
