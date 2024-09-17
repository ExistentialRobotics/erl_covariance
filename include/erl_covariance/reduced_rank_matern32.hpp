#pragma once

#include "reduced_rank_covariance.hpp"

namespace erl::covariance {

    template<long Dim>
    class ReducedRankMatern32 : public ReducedRankCovariance {
    public:
        explicit ReducedRankMatern32(std::shared_ptr<Setting> setting)
            : ReducedRankCovariance(std::move(setting)) {}

        [[nodiscard]] std::string
        GetCovarianceType() const override {
            if (Dim == Eigen::Dynamic) { return "ReducedRankMatern32_Xd"; }
            return "ReducedRankMatern32_" + std::to_string(Dim) + "D";
        }

        [[nodiscard]] Eigen::VectorXd
        ComputeSpectralDensities(const Eigen::VectorXd &freq_squared_norm) const override {
            const double l_inv = 1.0 / m_setting_->scale;
            const double l2_inv = l_inv * l_inv;
            const double beta = 3.0 * l2_inv;
            Eigen::VectorXd s(freq_squared_norm.size());
            double *s_ptr = s.data();
            if (Dim == 1) {
                const double alpha = 12.0 * std::sqrt(3.0) * l2_inv * l_inv;
                for (long i = 0; i < s.size(); ++i) {
                    double s_i = beta + freq_squared_norm[i];
                    s_ptr[i] = alpha / (s_i * s_i);
                }
            } else if (Dim == 2) {
                const double alpha = 18.0 * std::sqrt(3.0) * M_PI * l2_inv * l_inv;
                for (long i = 0; i < s.size(); ++i) { s_ptr[i] = alpha * std::pow(beta + freq_squared_norm[i], -2.5); }
            } else if (Dim == 3) {
                const double alpha = 96.0 * M_PI * std::sqrt(3) * l2_inv * l_inv;
                for (long i = 0; i < s.size(); ++i) {
                    double s_i = beta + freq_squared_norm[i];
                    s_ptr[i] = alpha / (s_i * s_i * s_i);
                }
            } else {
                const long dims = ReducedRankCovariance::m_setting_->x_dim;
                ERL_DEBUG_ASSERT(dims > 0, "x_dim from setting must be greater than 0 when Dim is Eigen::Dynamic");
                const double alpha =
                    std::pow(2.0, dims + 1) * std::pow(M_PI, (dims - 1) / 2.0) * 3.0 * std::sqrt(3.0) * std::tgamma((3.0 + dims) / 2.0) * l2_inv * l_inv;
                for (long i = 0; i < s.size(); ++i) { s_ptr[i] = alpha * std::pow(beta + freq_squared_norm[i], -(dims + 3.0) / 2.0); }
            }
            return s;
        }
    };

    using ReducedRankMatern32_1D = ReducedRankMatern32<1>;
    using ReducedRankMatern32_2D = ReducedRankMatern32<2>;
    using ReducedRankMatern32_3D = ReducedRankMatern32<3>;
    using ReducedRankMatern32_Xd = ReducedRankMatern32<Eigen::Dynamic>;

    ERL_REGISTER_COVARIANCE(ReducedRankMatern32_1D);
    ERL_REGISTER_COVARIANCE(ReducedRankMatern32_2D);
    ERL_REGISTER_COVARIANCE(ReducedRankMatern32_3D);
    ERL_REGISTER_COVARIANCE(ReducedRankMatern32_Xd);

}  // namespace erl::covariance
