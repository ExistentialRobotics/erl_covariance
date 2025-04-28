#pragma once

#include "covariance.hpp"

namespace erl::covariance {

    template<typename Dtype, int Dim>
    class OrnsteinUhlenbeck : public Covariance<Dtype> {
        // ref1: https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
        // ref2: https://www.cs.cmu.edu/~epxing/Class/10708-15/notes/10708_scribe_lecture21.pdf

    public:
        using Super = Covariance<Dtype>;
        using Setting = typename Super::Setting;
        using MatrixX = Eigen::MatrixX<Dtype>;
        using VectorX = Eigen::VectorX<Dtype>;

        explicit OrnsteinUhlenbeck(std::shared_ptr<Setting> setting);

        [[nodiscard]] std::string
        GetCovarianceType() const override {
            return type_name<OrnsteinUhlenbeck>();
        }

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
        ComputeKtrainWithGradient(const Eigen::Ref<const MatrixX> &, long, Eigen::VectorXl &, MatrixX &, MatrixX & /*mat_alpha*/) override;

        [[nodiscard]] std::pair<long, long>
        ComputeKtrainWithGradient(const Eigen::Ref<const MatrixX> &mat_x, long num_samples, Eigen::VectorXl &vec_grad_flags, MatrixX &mat_k);

        [[nodiscard]] std::pair<long, long>
        ComputeKtrainWithGradient(
            const Eigen::Ref<const MatrixX> &,
            long,
            Eigen::VectorXl &,
            const Eigen::Ref<const VectorX> &,
            const Eigen::Ref<const VectorX> &,
            const Eigen::Ref<const VectorX> &,
            MatrixX &,
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
            const Eigen::Ref<const MatrixX> &,
            long,
            const Eigen::Ref<const Eigen::VectorXl> &,
            const Eigen::Ref<const MatrixX> &,
            long,
            bool,
            MatrixX &) const override;
    };

    using OrnsteinUhlenbeck1d = OrnsteinUhlenbeck<double, 1>;
    using OrnsteinUhlenbeck2d = OrnsteinUhlenbeck<double, 2>;
    using OrnsteinUhlenbeck3d = OrnsteinUhlenbeck<double, 3>;
    using OrnsteinUhlenbeckXd = OrnsteinUhlenbeck<double, Eigen::Dynamic>;

    using OrnsteinUhlenbeck1f = OrnsteinUhlenbeck<float, 1>;
    using OrnsteinUhlenbeck2f = OrnsteinUhlenbeck<float, 2>;
    using OrnsteinUhlenbeck3f = OrnsteinUhlenbeck<float, 3>;
    using OrnsteinUhlenbeckXf = OrnsteinUhlenbeck<float, Eigen::Dynamic>;

}  // namespace erl::covariance

#include "ornstein_uhlenbeck.tpp"
