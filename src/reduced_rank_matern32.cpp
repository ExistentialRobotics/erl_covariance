#include "erl_covariance/reduced_rank_matern32.hpp"

namespace erl::covariance {
    template<typename Dtype, int Dim>
    ReducedRankMatern32<Dtype, Dim>::ReducedRankMatern32(std::shared_ptr<Setting> setting)
        : Super(std::move(setting)) {}

    template<typename Dtype, int Dim>
    std::string
    ReducedRankMatern32<Dtype, Dim>::GetCovarianceType() const {
        return type_name<ReducedRankMatern32>();
    }

    template<typename Dtype, int Dim>
    [[nodiscard]] std::string
    ReducedRankMatern32<Dtype, Dim>::GetCovarianceName() const {
        return "ReducedRankMatern32";
    }

    template<typename Dtype, int Dim>
    typename ReducedRankMatern32<Dtype, Dim>::VectorX
    ReducedRankMatern32<Dtype, Dim>::ComputeSpectralDensities(
        const VectorX &freq_squared_norm) const {
        const Dtype l_inv = 1.0f / Super::m_setting_->scale;
        const Dtype l2_inv = l_inv * l_inv;
        const Dtype beta = 3.0f * l2_inv;
        VectorX s(freq_squared_norm.size());
        Dtype *s_ptr = s.data();
        if (Dim == 1) {
            const Dtype alpha = 12.0f * std::sqrt(3.0f) * l2_inv * l_inv;
            for (long i = 0; i < s.size(); ++i) {
                Dtype s_i = beta + freq_squared_norm[i];
                s_ptr[i] = alpha / (s_i * s_i);
            }
        } else if (Dim == 2) {
            const Dtype alpha = 18.0f * std::sqrt(3.0f) * M_PI * l2_inv * l_inv;
            for (long i = 0; i < s.size(); ++i) {
                s_ptr[i] = alpha * std::pow(beta + freq_squared_norm[i], -2.5f);
            }
        } else if (Dim == 3) {
            const Dtype alpha = 96.0f * M_PI * std::sqrt(3.0f) * l2_inv * l_inv;
            for (long i = 0; i < s.size(); ++i) {
                Dtype s_i = beta + freq_squared_norm[i];
                s_ptr[i] = alpha / (s_i * s_i * s_i);
            }
        } else {
            const long dims = Super::m_setting_->x_dim;
            ERL_DEBUG_ASSERT(
                dims > 0,
                "x_dim from setting must be greater than 0 when Dim is Eigen::Dynamic");
            const Dtype alpha = std::pow(2.0f, dims + 1) * std::pow(M_PI, (dims - 1) / 2.0f) *
                                3.0f * std::sqrt(3.0f) * std::tgamma((3.0f + dims) / 2.0f) *
                                l2_inv * l_inv;
            for (long i = 0; i < s.size(); ++i) {
                s_ptr[i] = alpha * std::pow(beta + freq_squared_norm[i], -(dims + 3.0f) / 2.0f);
            }
        }
        return s;
    }

    template class ReducedRankMatern32<double, 1>;
    template class ReducedRankMatern32<double, 2>;
    template class ReducedRankMatern32<double, 3>;
    template class ReducedRankMatern32<double, Eigen::Dynamic>;

    template class ReducedRankMatern32<float, 1>;
    template class ReducedRankMatern32<float, 2>;
    template class ReducedRankMatern32<float, 3>;
    template class ReducedRankMatern32<float, Eigen::Dynamic>;
}  // namespace erl::covariance
