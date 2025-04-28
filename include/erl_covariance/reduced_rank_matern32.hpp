#pragma once

#include "reduced_rank_covariance.hpp"

namespace erl::covariance {

    template<typename Dtype, int /*Dim*/>
    class ReducedRankMatern32 : public ReducedRankCovariance<Dtype> {
    public:
        using Super = ReducedRankCovariance<Dtype>;
        using Setting = typename Super::Setting;
        using MatrixX = Eigen::MatrixX<Dtype>;
        using VectorX = Eigen::VectorX<Dtype>;

        explicit ReducedRankMatern32(std::shared_ptr<Setting> setting);

        [[nodiscard]] std::string
        GetCovarianceType() const override;

        [[nodiscard]] VectorX
        ComputeSpectralDensities(const VectorX &freq_squared_norm) const override;
    };

    using ReducedRankMatern32_1d = ReducedRankMatern32<double, 1>;
    using ReducedRankMatern32_2d = ReducedRankMatern32<double, 2>;
    using ReducedRankMatern32_3d = ReducedRankMatern32<double, 3>;
    using ReducedRankMatern32_Xd = ReducedRankMatern32<double, Eigen::Dynamic>;

    using ReducedRankMatern32_1f = ReducedRankMatern32<float, 1>;
    using ReducedRankMatern32_2f = ReducedRankMatern32<float, 2>;
    using ReducedRankMatern32_3f = ReducedRankMatern32<float, 3>;
    using ReducedRankMatern32_Xf = ReducedRankMatern32<float, Eigen::Dynamic>;

}  // namespace erl::covariance

#include "reduced_rank_matern32.tpp"
