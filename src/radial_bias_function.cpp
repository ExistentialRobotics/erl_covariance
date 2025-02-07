#include "erl_covariance/radial_bias_function.hpp"  // included to register the class

namespace erl::covariance {
    ERL_REGISTER_COVARIANCE(RadialBiasFunction1d);
    ERL_REGISTER_COVARIANCE(RadialBiasFunction2d);
    ERL_REGISTER_COVARIANCE(RadialBiasFunction3d);
    ERL_REGISTER_COVARIANCE(RadialBiasFunctionXd);

    ERL_REGISTER_COVARIANCE(RadialBiasFunction1f);
    ERL_REGISTER_COVARIANCE(RadialBiasFunction2f);
    ERL_REGISTER_COVARIANCE(RadialBiasFunction3f);
    ERL_REGISTER_COVARIANCE(RadialBiasFunctionXf);

}  // namespace erl::covariance
