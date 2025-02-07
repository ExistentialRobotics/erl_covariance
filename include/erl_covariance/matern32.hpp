#pragma once

#include "covariance.hpp"

namespace erl::covariance {

    template<int Dim, typename Dtype>
    class Matern32 : public Covariance<Dtype> {
    public:
        using Super = Covariance<Dtype>;
        using Setting = typename Super::Setting;
        using Matrix = typename Super::Matrix;
        using Vector = typename Super::Vector;

        explicit Matern32(std::shared_ptr<Setting> setting)
            : Super(std::move(setting)) {
            if (Dim != Eigen::Dynamic) { Super::m_setting_->x_dim = Dim; }  // set x_dim
        }

        [[nodiscard]] std::string
        GetCovarianceType() const override {
            return type_name<Matern32>();
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

#include "matern32.tpp"

    using Matern32_1d = Matern32<1, double>;
    using Matern32_2d = Matern32<2, double>;
    using Matern32_3d = Matern32<3, double>;
    using Matern32_Xd = Matern32<Eigen::Dynamic, double>;

    using Matern32_1f = Matern32<1, float>;
    using Matern32_2f = Matern32<2, float>;
    using Matern32_3f = Matern32<3, float>;
    using Matern32_Xf = Matern32<Eigen::Dynamic, float>;

    ERL_REGISTER_COVARIANCE(Matern32_1d);
    ERL_REGISTER_COVARIANCE(Matern32_2d);
    ERL_REGISTER_COVARIANCE(Matern32_3d);
    ERL_REGISTER_COVARIANCE(Matern32_Xd);

    ERL_REGISTER_COVARIANCE(Matern32_1f);
    ERL_REGISTER_COVARIANCE(Matern32_2f);
    ERL_REGISTER_COVARIANCE(Matern32_3f);
    ERL_REGISTER_COVARIANCE(Matern32_Xf);

}  // namespace erl::covariance
