#pragma once

#include "reduced_rank_covariance.hpp"

namespace erl::covariance {

    template<int Dim, typename Dtype>
    class ReducedRankMatern32 : public ReducedRankCovariance<Dtype> {
    public:
        using Super = ReducedRankCovariance<Dtype>;
        using Setting = typename Super::Setting;
        using MatrixX = Eigen::MatrixX<Dtype>;
        using VectorX = Eigen::VectorX<Dtype>;

        explicit ReducedRankMatern32(std::shared_ptr<Setting> setting)
            : Super(std::move(setting)) {}

        [[nodiscard]] std::string
        GetCovarianceType() const override {
            return type_name<ReducedRankMatern32>();
        }

        [[nodiscard]] VectorX
        ComputeSpectralDensities(const VectorX &freq_squared_norm) const override {
            const Dtype l_inv = 1.0 / Super::m_setting_->scale;
            const Dtype l2_inv = l_inv * l_inv;
            const Dtype beta = 3.0 * l2_inv;
            VectorX s(freq_squared_norm.size());
            Dtype *s_ptr = s.data();
            if (Dim == 1) {
                const Dtype alpha = 12.0 * std::sqrt(3.0) * l2_inv * l_inv;
                for (long i = 0; i < s.size(); ++i) {
                    Dtype s_i = beta + freq_squared_norm[i];
                    s_ptr[i] = alpha / (s_i * s_i);
                }
            } else if (Dim == 2) {
                const Dtype alpha = 18.0 * std::sqrt(3.0) * M_PI * l2_inv * l_inv;
                for (long i = 0; i < s.size(); ++i) { s_ptr[i] = alpha * std::pow(beta + freq_squared_norm[i], -2.5); }
            } else if (Dim == 3) {
                const Dtype alpha = 96.0 * M_PI * std::sqrt(3) * l2_inv * l_inv;
                for (long i = 0; i < s.size(); ++i) {
                    Dtype s_i = beta + freq_squared_norm[i];
                    s_ptr[i] = alpha / (s_i * s_i * s_i);
                }
            } else {
                const long dims = Super::m_setting_->x_dim;
                ERL_DEBUG_ASSERT(dims > 0, "x_dim from setting must be greater than 0 when Dim is Eigen::Dynamic");
                const Dtype alpha =
                    std::pow(2.0, dims + 1) * std::pow(M_PI, (dims - 1) / 2.0) * 3.0 * std::sqrt(3.0) * std::tgamma((3.0 + dims) / 2.0) * l2_inv * l_inv;
                for (long i = 0; i < s.size(); ++i) { s_ptr[i] = alpha * std::pow(beta + freq_squared_norm[i], -(dims + 3.0) / 2.0); }
            }
            return s;
        }
    };

    using ReducedRankMatern32_1d = ReducedRankMatern32<1, double>;
    using ReducedRankMatern32_2d = ReducedRankMatern32<2, double>;
    using ReducedRankMatern32_3d = ReducedRankMatern32<3, double>;
    using ReducedRankMatern32_Xd = ReducedRankMatern32<Eigen::Dynamic, double>;

    using ReducedRankMatern32_1f = ReducedRankMatern32<1, float>;
    using ReducedRankMatern32_2f = ReducedRankMatern32<2, float>;
    using ReducedRankMatern32_3f = ReducedRankMatern32<3, float>;
    using ReducedRankMatern32_Xf = ReducedRankMatern32<Eigen::Dynamic, float>;

}  // namespace erl::covariance
