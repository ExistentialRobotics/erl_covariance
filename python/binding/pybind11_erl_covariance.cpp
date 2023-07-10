#include "erl_covariance/pybind11_erl_covariance.hpp"

#include "erl_common/pybind11.hpp"
#include "erl_common/string_utils.hpp"
#include "erl_covariance/custom_kernel_v1.hpp"
#include "erl_covariance/custom_kernel_v2.hpp"
#include "erl_covariance/custom_kernel_v3.hpp"
#include "erl_covariance/custom_kernel_v4.hpp"
#include "erl_covariance/matern32.hpp"
#include "erl_covariance/ornstein_uhlenbeck.hpp"
#include "erl_covariance/radial_bias_function.hpp"
#include "erl_covariance/rational_quadratic.hpp"

using namespace erl::common;
using namespace erl::covariance;

void
BindCovariance(py::module &m) {
    auto submodule = m.def_submodule("covariance", "Interface of erl_covariance.");

    auto py_covariance = py::class_<Covariance, std::shared_ptr<Covariance>>(submodule, ERL_AS_STRING(Covariance));

    py::enum_<Covariance::Type>(py_covariance, "Type", py::arithmetic(), "Type of submodule kernel.")
        .value(Covariance::GetTypeName(Covariance::Type::kOrnsteinUhlenbeck), Covariance::Type::kOrnsteinUhlenbeck)
        .value(Covariance::GetTypeName(Covariance::Type::kMatern32), Covariance::Type::kMatern32)
        .value(Covariance::GetTypeName(Covariance::Type::kRadialBiasFunction), Covariance::Type::kRadialBiasFunction)
        .value(Covariance::GetTypeName(Covariance::Type::kRationalQuadratic), Covariance::Type::kRationalQuadratic)
        .value(Covariance::GetTypeName(Covariance::Type::kCustomKernelV1), Covariance::Type::kCustomKernelV1)
        .value(Covariance::GetTypeName(Covariance::Type::kCustomKernelV2), Covariance::Type::kCustomKernelV2)
        .value(Covariance::GetTypeName(Covariance::Type::kCustomKernelV3), Covariance::Type::kCustomKernelV3)
        .value(Covariance::GetTypeName(Covariance::Type::kCustomKernelV4), Covariance::Type::kCustomKernelV4)
        .export_values();

    py::class_<Covariance::Setting, YamlableBase, std::shared_ptr<Covariance::Setting>>(py_covariance, "Setting")
        .def(py::init())
        .def(py::init<Covariance::Type>())
        .def_readwrite("type", &Covariance::Setting::type)
        .def_readwrite("alpha", &Covariance::Setting::alpha)
        .def_readwrite("scale", &Covariance::Setting::scale)
        .def_readwrite("scale_mix", &Covariance::Setting::scale_mix)
        .def_readwrite("weights", &Covariance::Setting::weights);

    py_covariance.def(py::init<>(&Covariance::Create), py::arg("setting"))
        .def_property_readonly("setting", &Covariance::GetSetting)
        .def("compute_ktrain", py::overload_cast<const Eigen::Ref<const Eigen::MatrixXd> &>(&Covariance::ComputeKtrain, py::const_), py::arg("mat_x"))
        .def(
            "compute_ktrain",
            py::overload_cast<const Eigen::Ref<const Eigen::MatrixXd> &, const Eigen::Ref<const Eigen::VectorXd> &>(&Covariance::ComputeKtrain, py::const_),
            py::arg("mat_x"),
            py::arg("vec_sigma_y"))
        .def("compute_ktest", &Covariance::ComputeKtest, py::arg("mat_x1"), py::arg("mat_x2"))
        .def(
            "compute_ktrain_with_gradient",
            py::overload_cast<const Eigen::Ref<const Eigen::MatrixXd> &, const Eigen::Ref<const Eigen::VectorXb> &>(
                &Covariance::ComputeKtrainWithGradient,
                py::const_),
            py::arg("mat_x"),
            py::arg("vec_grad_flags"))
        .def(
            "compute_ktrain_with_gradient",
            py::overload_cast<
                const Eigen::Ref<const Eigen::MatrixXd> &,
                const Eigen::Ref<const Eigen::VectorXb> &,
                const Eigen::Ref<const Eigen::VectorXd> &,
                const Eigen::Ref<const Eigen::VectorXd> &,
                const Eigen::Ref<const Eigen::VectorXd> &>(&Covariance::ComputeKtrainWithGradient, py::const_),
            py::arg("mat_x"),
            py::arg("vec_grad_flags"),
            py::arg("vec_sigma_x"),
            py::arg("vec_sigma_y"),
            py::arg("vec_sigma_grad"))
        .def("compute_ktest_with_gradient", &Covariance::ComputeKtestWithGradient, py::arg("mat_x1"), py::arg("vec_grad_flags"), py::arg("mat_x2"));

    py::class_<OrnsteinUhlenbeck, Covariance, std::shared_ptr<OrnsteinUhlenbeck>>(submodule, ERL_AS_STRING(OrnsteinUhlenbeck))
        .def(py::init(py::overload_cast<>(&OrnsteinUhlenbeck::Create)))
        .def(py::init(py::overload_cast<std::shared_ptr<Covariance::Setting>>(&OrnsteinUhlenbeck::Create)), py::arg("setting"));
    py::class_<Matern32, Covariance, std::shared_ptr<Matern32>>(submodule, ERL_AS_STRING(Matern32))
        .def(py::init(py::overload_cast<>(&Matern32::Create)))
        .def(py::init(py::overload_cast<std::shared_ptr<Covariance::Setting>>(&Matern32::Create)), py::arg("setting"));
    py::class_<RadialBiasFunction, Covariance, std::shared_ptr<RadialBiasFunction>>(submodule, ERL_AS_STRING(RadialBiasFunction))
        .def(py::init(py::overload_cast<>(&RadialBiasFunction::Create)))
        .def(py::init(py::overload_cast<std::shared_ptr<Covariance::Setting>>(&RadialBiasFunction::Create)), py::arg("setting"));
    py::class_<RationalQuadratic, Covariance, std::shared_ptr<RationalQuadratic>>(submodule, ERL_AS_STRING(RationalQuadratic))
        .def(py::init(py::overload_cast<>(&RationalQuadratic::Create)))
        .def(py::init(py::overload_cast<std::shared_ptr<Covariance::Setting>>(&RationalQuadratic::Create)), py::arg("setting"));
    py::class_<CustomKernelV1, Covariance, std::shared_ptr<CustomKernelV1>>(submodule, ERL_AS_STRING(CustomKernelV1))
        .def(py::init(py::overload_cast<>(&CustomKernelV1::Create)))
        .def(py::init(py::overload_cast<std::shared_ptr<Covariance::Setting>>(&CustomKernelV1::Create)), py::arg("setting"));
    py::class_<CustomKernelV2, Covariance, std::shared_ptr<CustomKernelV2>>(submodule, ERL_AS_STRING(CustomKernelV2))
        .def(py::init(py::overload_cast<>(&CustomKernelV2::Create)))
        .def(py::init(py::overload_cast<std::shared_ptr<Covariance::Setting>>(&CustomKernelV2::Create)), py::arg("setting"));
    py::class_<CustomKernelV3, Covariance, std::shared_ptr<CustomKernelV3>>(submodule, ERL_AS_STRING(CustomKernelV3))
        .def(py::init(py::overload_cast<>(&CustomKernelV3::Create)))
        .def(py::init(py::overload_cast<std::shared_ptr<Covariance::Setting>>(&CustomKernelV3::Create)), py::arg("setting"));
    py::class_<CustomKernelV4, Covariance, std::shared_ptr<CustomKernelV4>>(submodule, ERL_AS_STRING(CustomKernelV4))
        .def(py::init(py::overload_cast<>(&CustomKernelV4::Create)))
        .def(py::init(py::overload_cast<std::shared_ptr<Covariance::Setting>>(&CustomKernelV4::Create)), py::arg("setting"));
}
