#include "erl_covariance/init.hpp"

#include "erl_covariance/matern32.hpp"
#include "erl_covariance/ornstein_uhlenbeck.hpp"
#include "erl_covariance/radial_bias_function.hpp"
#include "erl_covariance/rational_quadratic.hpp"
#include "erl_covariance/reduced_rank_covariance.hpp"
#include "erl_covariance/reduced_rank_matern32.hpp"

namespace erl::covariance {

#define REGISTER(x) (void) x::Register<x>()

    bool initialized = Init();

    bool
    Init() {
        static bool initialized_ = false;
        if (initialized_) { return true; }
        REGISTER(Covariance<double>::Setting);
        REGISTER(Covariance<float>::Setting);
        REGISTER(Matern32_1d);
        REGISTER(Matern32_2d);
        REGISTER(Matern32_3d);
        REGISTER(Matern32_Xd);
        REGISTER(Matern32_1f);
        REGISTER(Matern32_2f);
        REGISTER(Matern32_3f);
        REGISTER(Matern32_Xf);
        REGISTER(OrnsteinUhlenbeck1d);
        REGISTER(OrnsteinUhlenbeck2d);
        REGISTER(OrnsteinUhlenbeck3d);
        REGISTER(OrnsteinUhlenbeckXd);
        REGISTER(OrnsteinUhlenbeck1f);
        REGISTER(OrnsteinUhlenbeck2f);
        REGISTER(OrnsteinUhlenbeck3f);
        REGISTER(OrnsteinUhlenbeckXf);
        REGISTER(RadialBiasFunction1d);
        REGISTER(RadialBiasFunction2d);
        REGISTER(RadialBiasFunction3d);
        REGISTER(RadialBiasFunctionXd);
        REGISTER(RadialBiasFunction1f);
        REGISTER(RadialBiasFunction2f);
        REGISTER(RadialBiasFunction3f);
        REGISTER(RadialBiasFunctionXf);
        REGISTER(RationalQuadratic1d);
        REGISTER(RationalQuadratic2d);
        REGISTER(RationalQuadratic3d);
        REGISTER(RationalQuadraticXd);
        REGISTER(RationalQuadratic1f);
        REGISTER(RationalQuadratic2f);
        REGISTER(RationalQuadratic3f);
        REGISTER(RationalQuadraticXf);
        REGISTER(ReducedRankCovariance<double>::Setting);
        REGISTER(ReducedRankCovariance<float>::Setting);
        REGISTER(ReducedRankMatern32_1d);
        REGISTER(ReducedRankMatern32_2d);
        REGISTER(ReducedRankMatern32_3d);
        REGISTER(ReducedRankMatern32_Xd);
        REGISTER(ReducedRankMatern32_1f);
        REGISTER(ReducedRankMatern32_2f);
        REGISTER(ReducedRankMatern32_3f);
        REGISTER(ReducedRankMatern32_Xf);
        ERL_INFO("erl_covariance initialized");
        initialized_ = true;
        return true;
    }
}  // namespace erl::covariance
