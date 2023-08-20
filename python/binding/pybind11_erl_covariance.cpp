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
    auto py_covariance = py::class_<Covariance, std::shared_ptr<Covariance>>(m, ERL_AS_STRING(Covariance));

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
        .def(
            "compute_ktrain",
            [](const Covariance &self, const Eigen::Ref<const Eigen::MatrixXd> &mat_x) -> Eigen::MatrixXd {
                long n = mat_x.cols();
                if (n == 0) { return {}; }
                std::pair<long, long> size = Covariance::GetMinimumKtrainSize(n, 0, 0);
                Eigen::MatrixXd k_mat(size.first, size.second);
                (void) self.ComputeKtrain(k_mat, mat_x);
                return k_mat;
            },
            py::arg("mat_x"))
        .def(
            "compute_ktrain",
            [](const Covariance &self,
               const Eigen::Ref<const Eigen::MatrixXd> &mat_x,
               const Eigen::Ref<const Eigen::VectorXd> &vec_sigma_y) -> Eigen::MatrixXd {
                long n = mat_x.cols();
                if (n == 0) { return {}; }
                std::pair<long, long> size = Covariance::GetMinimumKtrainSize(n, 0, 0);
                Eigen::MatrixXd k_mat(size.first, size.second);
                (void) self.ComputeKtrain(k_mat, mat_x, vec_sigma_y);
                return k_mat;
            },
            py::arg("mat_x"),
            py::arg("vec_sigma_y"))
        .def(
            "compute_ktest",
            [](const Covariance &self, const Eigen::Ref<const Eigen::MatrixXd> &mat_x1, const Eigen::Ref<const Eigen::MatrixXd> &mat_x2) -> Eigen::MatrixXd {
                long n = mat_x1.cols();
                long m = mat_x2.cols();
                if (n == 0 || m == 0) { return {}; }
                std::pair<long, long> size = Covariance::GetMinimumKtestSize(n, 0, 0, m);
                Eigen::MatrixXd k_mat(size.first, size.second);
                (void) self.ComputeKtest(k_mat, mat_x1, mat_x2);
                return k_mat;
            },
            py::arg("mat_x1"),
            py::arg("mat_x2"))
        .def(
            "compute_ktrain_with_gradient",
            [](const Covariance &self,
               const Eigen::Ref<const Eigen::MatrixXd> &mat_x,
               const Eigen::Ref<const Eigen::VectorXb> &vec_grad_flags) -> Eigen::MatrixXd {
                long n = mat_x.cols();
                if (n == 0) { return {}; }
                long dim = mat_x.rows();
                long n_grad = vec_grad_flags.cast<long>().sum();
                std::pair<long, long> size = Covariance::GetMinimumKtrainSize(n, n_grad, dim);
                Eigen::MatrixXd k_mat(size.first, size.second);
                (void) self.ComputeKtrainWithGradient(k_mat, mat_x, vec_grad_flags);
                return k_mat;
            },
            py::arg("mat_x"),
            py::arg("vec_grad_flags"))
        .def(
            "compute_ktrain_with_gradient",
            [](const Covariance &self,
               const Eigen::Ref<const Eigen::MatrixXd> &mat_x,
               const Eigen::Ref<const Eigen::VectorXb> &vec_grad_flags,
               const Eigen::Ref<const Eigen::VectorXd> &vec_sigma_x,
               const Eigen::Ref<const Eigen::VectorXd> &vec_sigma_y,
               const Eigen::Ref<const Eigen::VectorXd> &vec_sigma_grad) -> Eigen::MatrixXd {
                long n = mat_x.cols();
                if (n == 0) { return {}; }
                long dim = mat_x.rows();
                long n_grad = vec_grad_flags.cast<long>().sum();
                std::pair<long, long> size = Covariance::GetMinimumKtrainSize(n, n_grad, dim);
                Eigen::MatrixXd k_mat(size.first, size.second);
                (void) self.ComputeKtrainWithGradient(k_mat, mat_x, vec_grad_flags, vec_sigma_x, vec_sigma_y, vec_sigma_grad);
                return k_mat;
            },
            py::arg("mat_x"),
            py::arg("vec_grad_flags"),
            py::arg("vec_sigma_x"),
            py::arg("vec_sigma_y"),
            py::arg("vec_sigma_grad"))
        .def(
            "compute_ktest_with_gradient",
            [](const Covariance &self,
               const Eigen::Ref<const Eigen::MatrixXd> &mat_x1,
               const Eigen::Ref<const Eigen::VectorXb> &vec_grad1_flags,
               const Eigen::Ref<const Eigen::MatrixXd> &mat_x2) -> Eigen::MatrixXd {
                long n = mat_x1.cols();
                long m = mat_x2.cols();
                if (n == 0 || m == 0) { return {}; }
                long dim = mat_x1.rows();
                long n_grad = vec_grad1_flags.cast<long>().sum();
                std::pair<long, long> size = Covariance::GetMinimumKtestSize(n, n_grad, dim, m);
                Eigen::MatrixXd k_mat(size.first, size.second);
                (void) self.ComputeKtestWithGradient(k_mat, mat_x1, vec_grad1_flags, mat_x2);
                return k_mat;
            },
            py::arg("mat_x1"),
            py::arg("vec_grad1_flags"),
            py::arg("mat_x2"));

    py::class_<OrnsteinUhlenbeck, Covariance, std::shared_ptr<OrnsteinUhlenbeck>>(m, ERL_AS_STRING(OrnsteinUhlenbeck))
        .def(py::init(py::overload_cast<std::shared_ptr<Covariance::Setting>>(&OrnsteinUhlenbeck::Create)), py::arg("setting") = nullptr);

    py::class_<Matern32<1>, Covariance, std::shared_ptr<Matern32<1>>>(m, "Matern32_1D")
        .def(py::init(py::overload_cast<std::shared_ptr<Covariance::Setting>>(&Matern32<1>::Create)), py::arg("setting") = nullptr);
    py::class_<Matern32<2>, Covariance, std::shared_ptr<Matern32<2>>>(m, "Matern32_2D")
        .def(py::init(py::overload_cast<std::shared_ptr<Covariance::Setting>>(&Matern32<2>::Create)), py::arg("setting") = nullptr);
    py::class_<Matern32<3>, Covariance, std::shared_ptr<Matern32<3>>>(m, "Matern32_3D")
        .def(py::init(py::overload_cast<std::shared_ptr<Covariance::Setting>>(&Matern32<3>::Create)), py::arg("setting") = nullptr);
    py::class_<Matern32<Eigen::Dynamic>, Covariance, std::shared_ptr<Matern32<Eigen::Dynamic>>>(m, "Matern32_xD")

        .def(py::init(py::overload_cast<std::shared_ptr<Covariance::Setting>>(&Matern32<Eigen::Dynamic>::Create)), py::arg("setting") = nullptr);
    py::class_<RadialBiasFunction<1>, Covariance, std::shared_ptr<RadialBiasFunction<1>>>(m, "RadialBiasFunction_1D")
        .def(py::init(py::overload_cast<std::shared_ptr<Covariance::Setting>>(&RadialBiasFunction<1>::Create)), py::arg("setting") = nullptr);
    py::class_<RadialBiasFunction<2>, Covariance, std::shared_ptr<RadialBiasFunction<2>>>(m, "RadialBiasFunction_2D")
        .def(py::init(py::overload_cast<std::shared_ptr<Covariance::Setting>>(&RadialBiasFunction<2>::Create)), py::arg("setting") = nullptr);
    py::class_<RadialBiasFunction<3>, Covariance, std::shared_ptr<RadialBiasFunction<3>>>(m, "RadialBiasFunction_3D")
        .def(py::init(py::overload_cast<std::shared_ptr<Covariance::Setting>>(&RadialBiasFunction<3>::Create)), py::arg("setting") = nullptr);
    py::class_<RadialBiasFunction<Eigen::Dynamic>, Covariance, std::shared_ptr<RadialBiasFunction<Eigen::Dynamic>>>(m, "RadialBiasFunction_xD")
        .def(py::init(py::overload_cast<std::shared_ptr<Covariance::Setting>>(&RadialBiasFunction<Eigen::Dynamic>::Create)), py::arg("setting") = nullptr);

    py::class_<RationalQuadratic<1>, Covariance, std::shared_ptr<RationalQuadratic<1>>>(m, "RationalQuadratic_1D")
        .def(py::init(py::overload_cast<std::shared_ptr<Covariance::Setting>>(&RationalQuadratic<1>::Create)), py::arg("setting") = nullptr);
    py::class_<RationalQuadratic<2>, Covariance, std::shared_ptr<RationalQuadratic<2>>>(m, "RationalQuadratic_2D")
        .def(py::init(py::overload_cast<std::shared_ptr<Covariance::Setting>>(&RationalQuadratic<2>::Create)), py::arg("setting") = nullptr);
    py::class_<RationalQuadratic<3>, Covariance, std::shared_ptr<RationalQuadratic<3>>>(m, "RationalQuadratic_3D")
        .def(py::init(py::overload_cast<std::shared_ptr<Covariance::Setting>>(&RationalQuadratic<3>::Create)), py::arg("setting") = nullptr);
    py::class_<RationalQuadratic<Eigen::Dynamic>, Covariance, std::shared_ptr<RationalQuadratic<Eigen::Dynamic>>>(m, "RationalQuadratic_xD")
        .def(py::init(py::overload_cast<std::shared_ptr<Covariance::Setting>>(&RationalQuadratic<Eigen::Dynamic>::Create)), py::arg("setting") = nullptr);

    py::class_<CustomKernelV1, Covariance, std::shared_ptr<CustomKernelV1>>(m, ERL_AS_STRING(CustomKernelV1))
        .def(py::init(py::overload_cast<std::shared_ptr<Covariance::Setting>>(&CustomKernelV1::Create)), py::arg("setting") = nullptr);
    py::class_<CustomKernelV2, Covariance, std::shared_ptr<CustomKernelV2>>(m, ERL_AS_STRING(CustomKernelV2))
        .def(py::init(py::overload_cast<std::shared_ptr<Covariance::Setting>>(&CustomKernelV2::Create)), py::arg("setting") = nullptr);
    py::class_<CustomKernelV3, Covariance, std::shared_ptr<CustomKernelV3>>(m, ERL_AS_STRING(CustomKernelV3))
        .def(py::init(py::overload_cast<std::shared_ptr<Covariance::Setting>>(&CustomKernelV3::Create)), py::arg("setting") = nullptr);
    py::class_<CustomKernelV4, Covariance, std::shared_ptr<CustomKernelV4>>(m, ERL_AS_STRING(CustomKernelV4))
        .def(py::init(py::overload_cast<std::shared_ptr<Covariance::Setting>>(&CustomKernelV4::Create)), py::arg("setting") = nullptr);
}

PYBIND11_MODULE(PYBIND_MODULE_NAME, m) {
    m.doc() = "Python 3 Interface of erl_covariance";
    BindCovariance(m);
}
