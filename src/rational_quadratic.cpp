#include "erl_covariance/rational_quadratic.hpp"

namespace erl::covariance {
    ERL_REGISTER_COVARIANCE(RationalQuadratic1d);
    ERL_REGISTER_COVARIANCE(RationalQuadratic2d);
    ERL_REGISTER_COVARIANCE(RationalQuadratic3d);
    ERL_REGISTER_COVARIANCE(RationalQuadraticXd);

    ERL_REGISTER_COVARIANCE(RationalQuadratic1f);
    ERL_REGISTER_COVARIANCE(RationalQuadratic2f);
    ERL_REGISTER_COVARIANCE(RationalQuadratic3f);
    ERL_REGISTER_COVARIANCE(RationalQuadraticXf);
}  // namespace erl::covariance
