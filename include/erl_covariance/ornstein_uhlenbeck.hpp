#pragma once

#include "covariance.hpp"

namespace erl::covariance {

    template<int Dim, typename Dtype>
    class OrnsteinUhlenbeck : public Covariance<Dtype> {
        // ref1: https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
        // ref2: https://www.cs.cmu.edu/~epxing/Class/10708-15/notes/10708_scribe_lecture21.pdf

    public:
        using Super = Covariance<Dtype>;
        using Setting = typename Super::Setting;
        using MatrixX = Eigen::MatrixX<Dtype>;
        using VectorX = Eigen::VectorX<Dtype>;

        explicit OrnsteinUhlenbeck(std::shared_ptr<Setting> setting)
            : Super(std::move(setting)) {
            if (Dim != Eigen::Dynamic) { Super::m_setting_->x_dim = Dim; }  // set x_dim
        }

        [[nodiscard]] std::string
        GetCovarianceType() const override {
            return type_name<OrnsteinUhlenbeck>();
        }

        [[nodiscard]] std::pair<long, long>
        ComputeKtrain(const Eigen::Ref<const MatrixX> &mat_x, long num_samples, MatrixX &k_mat, VectorX & /*vec_y*/) const override;

        [[nodiscard]] std::pair<long, long>
        ComputeKtrain(const Eigen::Ref<const MatrixX> &mat_x, const Eigen::Ref<const VectorX> &vec_var_y, long num_samples, MatrixX &k_mat, VectorX & /*vec_y*/)
            const override;

        [[nodiscard]] std::pair<long, long>
        ComputeKtest(const Eigen::Ref<const MatrixX> &mat_x1, long num_samples1, const Eigen::Ref<const MatrixX> &mat_x2, long num_samples2, MatrixX &k_mat)
            const override;

        [[nodiscard]] std::pair<long, long>
        ComputeKtrainWithGradient(const Eigen::Ref<const MatrixX> &, long, Eigen::VectorXl &, MatrixX &, VectorX &) const override;

        [[nodiscard]] std::pair<long, long>
        ComputeKtrainWithGradient(
            const Eigen::Ref<const MatrixX> &,
            long,
            Eigen::VectorXl &,
            const Eigen::Ref<const VectorX> &,
            const Eigen::Ref<const VectorX> &,
            const Eigen::Ref<const VectorX> &,
            MatrixX &,
            VectorX &) const override;

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

    using OrnsteinUhlenbeck1d = OrnsteinUhlenbeck<1, double>;
    using OrnsteinUhlenbeck2d = OrnsteinUhlenbeck<2, double>;
    using OrnsteinUhlenbeck3d = OrnsteinUhlenbeck<3, double>;
    using OrnsteinUhlenbeckXd = OrnsteinUhlenbeck<Eigen::Dynamic, double>;

    using OrnsteinUhlenbeck1f = OrnsteinUhlenbeck<1, float>;
    using OrnsteinUhlenbeck2f = OrnsteinUhlenbeck<2, float>;
    using OrnsteinUhlenbeck3f = OrnsteinUhlenbeck<3, float>;
    using OrnsteinUhlenbeckXf = OrnsteinUhlenbeck<Eigen::Dynamic, float>;

}  // namespace erl::covariance

#include "ornstein_uhlenbeck.tpp"
