#include "erl_common/pybind11.hpp"
#include "erl_common/string_utils.hpp"
#include "erl_covariance/covariance.hpp"

using namespace erl::common;
using namespace erl::covariance;

void
BindCovariance(const py::module &m) {
    auto py_covariance = py::class_<Covariance, std::shared_ptr<Covariance>>(m, "Covariance");

    py::class_<Covariance::Setting, YamlableBase, std::shared_ptr<Covariance::Setting>>(py_covariance, "Setting")
        .def(py::init())
        .def_readwrite("alpha", &Covariance::Setting::alpha)
        .def_readwrite("scale", &Covariance::Setting::scale)
        .def_readwrite("scale_mix", &Covariance::Setting::scale_mix)
        .def_readwrite("weights", &Covariance::Setting::weights);

    py_covariance.def_property_readonly("setting", &Covariance::GetSetting)
        .def(
            "compute_ktrain",
            [](const Covariance &self, const Eigen::Ref<const Eigen::MatrixXd> &mat_x) -> Eigen::MatrixXd {
                const long n = mat_x.cols();
                if (n == 0) { return {}; }
                const auto [rows, cols] = Covariance::GetMinimumKtrainSize(n, 0, 0);
                Eigen::MatrixXd k_mat(rows, cols);
                (void) self.ComputeKtrain(k_mat, mat_x);
                return k_mat;
            },
            py::arg("mat_x"))
        .def(
            "compute_ktrain",
            [](const Covariance &self,
               const Eigen::Ref<const Eigen::MatrixXd> &mat_x,
               const Eigen::Ref<const Eigen::VectorXd> &vec_sigma_y) -> Eigen::MatrixXd {
                const long n = mat_x.cols();
                if (n == 0) { return {}; }
                const auto [rows, cols] = Covariance::GetMinimumKtrainSize(n, 0, 0);
                Eigen::MatrixXd k_mat(rows, cols);
                (void) self.ComputeKtrain(k_mat, mat_x, vec_sigma_y);
                return k_mat;
            },
            py::arg("mat_x"),
            py::arg("vec_sigma_y"))
        .def(
            "compute_ktest",
            [](const Covariance &self, const Eigen::Ref<const Eigen::MatrixXd> &mat_x1, const Eigen::Ref<const Eigen::MatrixXd> &mat_x2) -> Eigen::MatrixXd {
                const long n1 = mat_x1.cols();
                const long n2 = mat_x2.cols();
                if (n1 == 0 || n2 == 0) { return {}; }
                const auto [rows, cols] = Covariance::GetMinimumKtestSize(n1, 0, 0, n2);
                Eigen::MatrixXd k_mat(rows, cols);
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
                const long n = mat_x.cols();
                if (n == 0) { return {}; }
                const long dim = mat_x.rows();
                const long n_grad = vec_grad_flags.cast<long>().sum();
                const auto [rows, cols] = Covariance::GetMinimumKtrainSize(n, n_grad, dim);
                Eigen::MatrixXd k_mat(rows, cols);
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
                const long n = mat_x.cols();
                if (n == 0) { return {}; }
                const long dim = mat_x.rows();
                const long n_grad = vec_grad_flags.cast<long>().sum();
                const auto [rows, cols] = Covariance::GetMinimumKtrainSize(n, n_grad, dim);
                Eigen::MatrixXd k_mat(rows, cols);
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
                const long n1 = mat_x1.cols();
                const long n2 = mat_x2.cols();
                if (n1 == 0 || n2 == 0) { return {}; }
                const long dim = mat_x1.rows();
                const long n_grad = vec_grad1_flags.cast<long>().sum();
                const auto [rows, cols] = Covariance::GetMinimumKtestSize(n1, n_grad, dim, n2);
                Eigen::MatrixXd k_mat(rows, cols);
                (void) self.ComputeKtestWithGradient(k_mat, mat_x1, vec_grad1_flags, mat_x2);
                return k_mat;
            },
            py::arg("mat_x1"),
            py::arg("vec_grad1_flags"),
            py::arg("mat_x2"));
}

PYBIND11_MODULE(PYBIND_MODULE_NAME, m) {
    m.doc() = "Python 3 Interface of erl_covariance";
    BindCovariance(m);
}
