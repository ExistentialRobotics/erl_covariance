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
                return OrnsteinUhlenbeck::Create(std::static_pointer_cast<OrnsteinUhlenbeck::Setting>(setting));
            case Type::kMatern32:
                return Matern32::Create(std::static_pointer_cast<Matern32::Setting>(setting));
            case Type::kRadialBiasFunction:
                return RadialBiasFunction::Create(std::static_pointer_cast<RadialBiasFunction::Setting>(setting));
            case Type::kRationalQuadratic:
                return RationalQuadratic::Create(std::static_pointer_cast<RationalQuadratic::Setting>(setting));
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
