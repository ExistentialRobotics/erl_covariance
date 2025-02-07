#pragma once

#include "covariance.hpp"

namespace erl::covariance {

    template<int Dim, typename Dtype>
    class RationalQuadratic : public Covariance<Dtype> {
        // ref: https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.RationalQuadratic.html
    public:
        using Super = Covariance<Dtype>;
        using Setting = typename Super::Setting;
        using Matrix = typename Super::Matrix;
        using Vector = typename Super::Vector;

        explicit RationalQuadratic(std::shared_ptr<Setting> setting)
            : Super(std::move(setting)) {
            ERL_DEBUG_ASSERT(Dim == Eigen::Dynamic || Super::m_setting_->x_dim == Dim, "setting->x_dim should be {}.", Dim);
            ERL_WARN_ONCE_COND(Dim == Eigen::Dynamic, "Dim is Eigen::Dynamic, it may cause performance issue.");
            Super::m_setting_->x_dim = Dim;
        }

        [[nodiscard]] std::string
        GetCovarianceType() const override {
            return type_name<RationalQuadratic>();
        }

        [[nodiscard]] std::pair<long, long>
        ComputeKtrain(const Eigen::Ref<const Matrix> &mat_x, long num_samples, Matrix &k_mat, Vector & /*vec_alpha*/) const override;

        [[nodiscard]] std::pair<long, long>
        ComputeKtrain(const Eigen::Ref<const Matrix> &mat_x, const Eigen::Ref<const Vector> &vec_var_y, long num_samples, Matrix &k_mat, Vector & /*vec_alpha*/)
            const override;

        [[nodiscard]] std::pair<long, long>
        ComputeKtest(const Eigen::Ref<const Matrix> &mat_x1, long num_samples1, const Eigen::Ref<const Matrix> &mat_x2, long num_samples2, Matrix &k_mat)
            const override;

        [[nodiscard]] std::pair<long, long>
        ComputeKtrainWithGradient(
            const Eigen::Ref<const Matrix> &mat_x,
            long num_samples,
            Eigen::VectorXl &vec_grad_flags,
            Matrix &k_mat,
            Vector & /*vec_alpha*/) const override;

        [[nodiscard]] std::pair<long, long>
        ComputeKtrainWithGradient(
            const Eigen::Ref<const Matrix> &mat_x,
            long num_samples,
            Eigen::VectorXl &vec_grad_flags,
            const Eigen::Ref<const Vector> &vec_var_x,
            const Eigen::Ref<const Vector> &vec_var_y,
            const Eigen::Ref<const Vector> &vec_var_grad,
            Matrix &k_mat,
            Vector & /*vec_alpha*/) const override;

        [[nodiscard]] std::pair<long, long>
        ComputeKtestWithGradient(
            const Eigen::Ref<const Matrix> &mat_x1,
            long num_samples1,
            const Eigen::Ref<const Eigen::VectorXl> &vec_grad1_flags,
            const Eigen::Ref<const Matrix> &mat_x2,
            long num_samples2,
            bool predict_gradient,
            Matrix &k_mat) const override;
    };

#include "rational_quadratic.tpp"

    using RationalQuadratic1d = RationalQuadratic<1, double>;
    using RationalQuadratic2d = RationalQuadratic<2, double>;
    using RationalQuadratic3d = RationalQuadratic<3, double>;
    using RationalQuadraticXd = RationalQuadratic<Eigen::Dynamic, double>;

    using RationalQuadratic1f = RationalQuadratic<1, float>;
    using RationalQuadratic2f = RationalQuadratic<2, float>;
    using RationalQuadratic3f = RationalQuadratic<3, float>;
    using RationalQuadraticXf = RationalQuadratic<Eigen::Dynamic, float>;

}  // namespace erl::covariance
