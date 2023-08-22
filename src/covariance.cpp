#include "erl_covariance/covariance.hpp"
#include "erl_covariance/custom_kernel_v1.hpp"
#include "erl_covariance/custom_kernel_v2.hpp"
#include "erl_covariance/custom_kernel_v3.hpp"
#include "erl_covariance/custom_kernel_v4.hpp"
#include "erl_covariance/matern32.hpp"
#include "erl_covariance/ornstein_uhlenbeck.hpp"
#include "erl_covariance/radial_bias_function.hpp"
#include "erl_covariance/rational_quadratic.hpp"

namespace erl::covariance {
    Covariance::Covariance(std::shared_ptr<Setting> setting)
        : m_setting_(std::move(setting)) {}

    std::shared_ptr<Covariance>
    Covariance::Create(const std::shared_ptr<Setting> &setting) {
        ERL_DEBUG_ASSERT(setting != nullptr, "setting should not be nullptr");
        switch (setting->type) {
            case Type::kOrnsteinUhlenbeck:
                if (setting->x_dim == 1) {
                    return OrnsteinUhlenbeck<1>::Create(std::static_pointer_cast<OrnsteinUhlenbeck<1>::Setting>(setting));
                } else if (setting->x_dim == 2) {
                    return OrnsteinUhlenbeck<2>::Create(std::static_pointer_cast<OrnsteinUhlenbeck<2>::Setting>(setting));
                } else if (setting->x_dim == 3) {
                    return OrnsteinUhlenbeck<3>::Create(std::static_pointer_cast<OrnsteinUhlenbeck<3>::Setting>(setting));
                } else if (setting->x_dim == 4) {
                    return OrnsteinUhlenbeck<4>::Create(std::static_pointer_cast<OrnsteinUhlenbeck<4>::Setting>(setting));
                } else {
                    return OrnsteinUhlenbeck<Eigen::Dynamic>::Create(std::static_pointer_cast<OrnsteinUhlenbeck<Eigen::Dynamic>::Setting>(setting));
                }
            case Type::kMatern32:
                if (setting->x_dim == 1) {
                    return Matern32<1>::Create(std::static_pointer_cast<Matern32<1>::Setting>(setting));
                } else if (setting->x_dim == 2) {
                    return Matern32<2>::Create(std::static_pointer_cast<Matern32<2>::Setting>(setting));
                } else if (setting->x_dim == 3) {
                    return Matern32<3>::Create(std::static_pointer_cast<Matern32<3>::Setting>(setting));
                } else if (setting->x_dim == 4) {
                    return Matern32<4>::Create(std::static_pointer_cast<Matern32<4>::Setting>(setting));
                } else {
                    return Matern32<Eigen::Dynamic>::Create(std::static_pointer_cast<Matern32<Eigen::Dynamic>::Setting>(setting));
                }
            case Type::kRadialBiasFunction:
                if (setting->x_dim == 1) {
                    return RadialBiasFunction<1>::Create(std::static_pointer_cast<RadialBiasFunction<1>::Setting>(setting));
                } else if (setting->x_dim == 2) {
                    return RadialBiasFunction<2>::Create(std::static_pointer_cast<RadialBiasFunction<2>::Setting>(setting));
                } else if (setting->x_dim == 3) {
                    return RadialBiasFunction<3>::Create(std::static_pointer_cast<RadialBiasFunction<3>::Setting>(setting));
                } else if (setting->x_dim == 4) {
                    return RadialBiasFunction<4>::Create(std::static_pointer_cast<RadialBiasFunction<4>::Setting>(setting));
                } else {
                    return RadialBiasFunction<Eigen::Dynamic>::Create(std::static_pointer_cast<RadialBiasFunction<Eigen::Dynamic>::Setting>(setting));
                }
            case Type::kRationalQuadratic:
                if (setting->x_dim == 1) {
                    return RationalQuadratic<1>::Create(std::static_pointer_cast<RationalQuadratic<1>::Setting>(setting));
                } else if (setting->x_dim == 2) {
                    return RationalQuadratic<2>::Create(std::static_pointer_cast<RationalQuadratic<2>::Setting>(setting));
                } else if (setting->x_dim == 3) {
                    return RationalQuadratic<3>::Create(std::static_pointer_cast<RationalQuadratic<3>::Setting>(setting));
                } else if (setting->x_dim == 4) {
                    return RationalQuadratic<4>::Create(std::static_pointer_cast<RationalQuadratic<4>::Setting>(setting));
                } else {
                    return RationalQuadratic<Eigen::Dynamic>::Create(std::static_pointer_cast<RationalQuadratic<Eigen::Dynamic>::Setting>(setting));
                }
            case Type::kCustomKernelV1:
                return CustomKernelV1::Create(std::static_pointer_cast<CustomKernelV1::Setting>(setting));
            case Type::kCustomKernelV2:
                return CustomKernelV2::Create(std::static_pointer_cast<CustomKernelV2::Setting>(setting));
            case Type::kCustomKernelV3:
                return CustomKernelV3::Create(std::static_pointer_cast<CustomKernelV3::Setting>(setting));
            case Type::kCustomKernelV4:
                return CustomKernelV4::Create(std::static_pointer_cast<CustomKernelV4::Setting>(setting));
            case Type::kUnknown:
            default:
                throw std::logic_error("Covariance type is kUnknown, which is unexpected.");
        }
    }
}  // namespace erl::covariance
