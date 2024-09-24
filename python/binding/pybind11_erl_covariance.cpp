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
        .def_readwrite("x_dim", &Covariance::Setting::x_dim)
        .def_readwrite("alpha", &Covariance::Setting::alpha)
        .def_readwrite("scale", &Covariance::Setting::scale)
        .def_readwrite("scale_mix", &Covariance::Setting::scale_mix)
        .def_readwrite("weights", &Covariance::Setting::weights);

    py_covariance.def_property_readonly("type", &Covariance::GetCovarianceType)
        .def_property_readonly("setting", &Covariance::GetSetting)
        .def(
            "get_minimum_ktrain_size",
            &Covariance::GetMinimumKtrainSize,
            py::arg("num_samples"),
            py::arg("num_samples_with_gradient"),
            py::arg("num_gradient_dimensions"))
        .def(
            "get_minimum_ktest_size",
            &Covariance::GetMinimumKtestSize,
            py::arg("num_train_samples"),
            py::arg("num_train_samples_with_gradient"),
            py::arg("num_gradient_dimensions"),
            py::arg("num_test_queries"),
            py::arg("predict_gradient"))
        .def(
            "compute_ktrain",
            [](const Covariance &self, const Eigen::Ref<const Eigen::MatrixXd> &mat_x, const long num_samples, Eigen::VectorXd alpha_vec) {
                const long n = mat_x.cols();
                if (n == 0) { return py::make_tuple(py::none(), py::none()); }
                const auto [rows, cols] = self.GetMinimumKtrainSize(n, 0, 0);
                Eigen::MatrixXd k_mat(rows, cols);
                ERL_ASSERTM(alpha_vec.size() == cols, "alpha_vec has wrong size.");
                (void) self.ComputeKtrain(mat_x, num_samples, k_mat, alpha_vec);
                return py::make_tuple(k_mat, alpha_vec);
            },
            py::arg("mat_x"),
            py::arg("num_samples"),
            py::arg("alpha_vec"))
        .def(
            "compute_ktrain",
            [](const Covariance &self,
               const Eigen::Ref<const Eigen::MatrixXd> &mat_x,
               const Eigen::Ref<const Eigen::VectorXd> &vec_var_y,
               const long num_samples,
               Eigen::VectorXd alpha_vec) {
                const long n = mat_x.cols();
                if (n == 0) { return py::make_tuple(py::none(), py::none()); }
                const auto [rows, cols] = self.GetMinimumKtrainSize(n, 0, 0);
                Eigen::MatrixXd k_mat(rows, cols);
                ERL_ASSERTM(alpha_vec.size() == cols, "alpha_vec has wrong size. It should be {}.", cols);
                (void) self.ComputeKtrain(mat_x, vec_var_y, num_samples, k_mat, alpha_vec);
                return py::make_tuple(k_mat, alpha_vec);
            },
            py::arg("mat_x"),
            py::arg("vec_var_y"),
            py::arg("num_samples"),
            py::arg("alpha_vec"))
        .def(
            "compute_ktest",
            [](const Covariance &self,
               const Eigen::Ref<const Eigen::MatrixXd> &mat_x1,
               const long num_samples1,
               const Eigen::Ref<const Eigen::MatrixXd> &mat_x2,
               const long num_samples2) -> Eigen::MatrixXd {
                const long n1 = mat_x1.cols();
                const long n2 = mat_x2.cols();
                if (n1 == 0 || n2 == 0) { return {}; }
                const auto [rows, cols] = self.GetMinimumKtestSize(n1, 0, 0, n2, false);
                Eigen::MatrixXd k_mat(rows, cols);
                (void) self.ComputeKtest(mat_x1, num_samples1, mat_x2, num_samples2, k_mat);
                return k_mat;
            },
            py::arg("mat_x1"),
            py::arg("num_samples1"),
            py::arg("mat_x2"),
            py::arg("num_samples2"))
        .def(
            "compute_ktrain_with_gradient",
            [](const Covariance &self,
               const Eigen::Ref<const Eigen::MatrixXd> &mat_x,
               const long num_samples,
               Eigen::VectorXl vec_grad_flags,
               Eigen::VectorXd alpha_vec) {
                const long n = mat_x.cols();
                if (n == 0) { return py::make_tuple(py::none(), py::none()); }
                const long dim = mat_x.rows();
                const long n_grad = vec_grad_flags.cast<long>().sum();
                const auto [rows, cols] = self.GetMinimumKtrainSize(n, n_grad, dim);
                Eigen::MatrixXd k_mat(rows, cols);
                ERL_ASSERTM(alpha_vec.size() == cols, "alpha_vec has wrong size. It should be {}.", cols);
                (void) self.ComputeKtrainWithGradient(mat_x, num_samples, vec_grad_flags, k_mat, alpha_vec);
                return py::make_tuple(k_mat, alpha_vec);
            },
            py::arg("mat_x"),
            py::arg("num_samples"),
            py::arg("vec_grad_flags"),
            py::arg("alpha_vec"))
        .def(
            "compute_ktrain_with_gradient",
            [](const Covariance &self,
               const Eigen::Ref<const Eigen::MatrixXd> &mat_x,
               const long num_samples,
               Eigen::VectorXl vec_grad_flags,
               const Eigen::Ref<const Eigen::VectorXd> &vec_var_x,
               const Eigen::Ref<const Eigen::VectorXd> &vec_var_y,
               const Eigen::Ref<const Eigen::VectorXd> &vec_var_grad,
               Eigen::VectorXd alpha_vec) {
                const long n = mat_x.cols();
                if (n == 0) { return py::make_tuple(py::none(), py::none()); }
                const long dim = mat_x.rows();
                const long n_grad = vec_grad_flags.cast<long>().sum();
                const auto [rows, cols] = self.GetMinimumKtrainSize(n, n_grad, dim);
                Eigen::MatrixXd k_mat(rows, cols);
                ERL_ASSERTM(alpha_vec.size() == cols, "alpha_vec has wrong size. It should be {}.", cols);
                (void) self.ComputeKtrainWithGradient(mat_x, num_samples, vec_grad_flags, vec_var_x, vec_var_y, vec_var_grad, k_mat, alpha_vec);
                return py::make_tuple(k_mat, alpha_vec);
            },
            py::arg("mat_x"),
            py::arg("num_samples"),
            py::arg("vec_grad_flags"),
            py::arg("vec_var_x"),
            py::arg("vec_var_y"),
            py::arg("vec_var_grad"),
            py::arg("alpha_vec"))
        .def(
            "compute_ktest_with_gradient",
            [](const Covariance &self,
               const Eigen::Ref<const Eigen::MatrixXd> &mat_x1,
               const long num_samples1,
               const Eigen::Ref<const Eigen::VectorXl> &vec_grad1_flags,
               const Eigen::Ref<const Eigen::MatrixXd> &mat_x2,
               const long num_samples2,
               const bool predict_gradient) -> Eigen::MatrixXd {
                const long n1 = mat_x1.cols();
                const long n2 = mat_x2.cols();
                if (n1 == 0 || n2 == 0) { return {}; }
                const long dim = mat_x1.rows();
                const long n_grad = vec_grad1_flags.cast<long>().sum();
                const auto [rows, cols] = self.GetMinimumKtestSize(n1, n_grad, dim, n2, predict_gradient);
                Eigen::MatrixXd k_mat(rows, cols);
                (void) self.ComputeKtestWithGradient(mat_x1, num_samples1, vec_grad1_flags, mat_x2, num_samples2, predict_gradient, k_mat);
                return k_mat;
            },
            py::arg("mat_x1"),
            py::arg("num_samples1"),
            py::arg("vec_grad1_flags"),
            py::arg("mat_x2"),
            py::arg("num_samples2"),
            py::arg("predict_gradient"));
}

PYBIND11_MODULE(PYBIND_MODULE_NAME, m) {
    m.doc() = "Python 3 Interface of erl_covariance";
    BindCovariance(m);
}
