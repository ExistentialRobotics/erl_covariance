#include "erl_common/pybind11.hpp"
#include "erl_common/pybind11_yaml.hpp"
#include "erl_common/string_utils.hpp"
#include "erl_covariance/covariance.hpp"

using namespace erl::common;
using namespace erl::covariance;

template<typename Dtype>
void
BindCovarianceImpl(const py::module &m, const char *name) {
    using T = Covariance<Dtype>;
    using MatrixX = Eigen::MatrixX<Dtype>;
    using VectorX = Eigen::VectorX<Dtype>;

    auto py_covariance = py::class_<T, std::shared_ptr<T>>(m, name);

    py::class_<typename T::Setting, YamlableBase, std::shared_ptr<typename T::Setting>>(
        py_covariance,
        "Setting")
        .def(py::init())
        .def_readwrite("x_dim", &T::Setting::x_dim)
        .def_readwrite("scale", &T::Setting::scale)
        .def_readwrite("scale_mix", &T::Setting::scale_mix)
        .def_readwrite("weights", &T::Setting::weights);

    py_covariance.def_property_readonly("type", &T::GetCovarianceType)
        .def_property_readonly("setting", &T::GetSetting)
        .def(
            "get_minimum_ktrain_size",
            &T::GetMinimumKtrainSize,
            py::arg("num_samples"),
            py::arg("num_samples_with_gradient"),
            py::arg("num_gradient_dimensions"))
        .def(
            "get_minimum_ktest_size",
            &T::GetMinimumKtestSize,
            py::arg("num_train_samples"),
            py::arg("num_train_samples_with_gradient"),
            py::arg("num_gradient_dimensions"),
            py::arg("num_test_queries"),
            py::arg("predict_gradient"))
        .def(
            "compute_ktrain",
            [](T &self,
               const Eigen::Ref<const MatrixX> &mat_x,
               const long num_samples,
               const MatrixX &mat_alpha) {
                const long n = mat_x.cols();
                if (n == 0) { return py::dict(); }
                const auto [rows, cols] = self.GetMinimumKtrainSize(n, 0, 0);
                ERL_ASSERTM(mat_alpha.rows() == cols, "mat_alpha.rows() should be {}.", cols);
                MatrixX mat_k(rows, cols);
                MatrixX mat_alpha_out = mat_alpha;
                (void) self.ComputeKtrain(mat_x, num_samples, mat_k, mat_alpha_out);
                py::dict result;
                result["mat_k"] = mat_k;
                result["mat_alpha"] = mat_alpha_out;
                return result;
            },
            py::arg("mat_x"),
            py::arg("num_samples"),
            py::arg("mat_alpha"))
        .def(
            "compute_ktrain",
            [](T &self,
               const Eigen::Ref<const MatrixX> &mat_x,
               const Eigen::Ref<const VectorX> &vec_var_y,
               const long num_samples,
               const MatrixX &mat_alpha) {
                const long n = mat_x.cols();
                if (n == 0) { return py::dict(); }
                const auto [rows, cols] = self.GetMinimumKtrainSize(n, 0, 0);
                ERL_ASSERTM(mat_alpha.rows() == cols, "mat_alpha.rows() should be {}.", cols);
                MatrixX mat_k(rows, cols);
                MatrixX mat_alpha_out = mat_alpha;
                (void) self.ComputeKtrain(mat_x, vec_var_y, num_samples, mat_k, mat_alpha_out);
                py::dict result;
                result["mat_k"] = mat_k;
                result["mat_alpha"] = mat_alpha_out;
                return result;
            },
            py::arg("mat_x"),
            py::arg("vec_var_y"),
            py::arg("num_samples"),
            py::arg("mat_alpha"))
        .def(
            "compute_ktest",
            [](const T &self,
               const Eigen::Ref<const MatrixX> &mat_x1,
               const long num_samples1,
               const Eigen::Ref<const MatrixX> &mat_x2,
               const long num_samples2) -> MatrixX {
                const long n1 = mat_x1.cols();
                const long n2 = mat_x2.cols();
                if (n1 == 0 || n2 == 0) { return {}; }
                const auto [rows, cols] = self.GetMinimumKtestSize(n1, 0, 0, n2, false);
                MatrixX k_mat(rows, cols);
                (void) self.ComputeKtest(mat_x1, num_samples1, mat_x2, num_samples2, k_mat);
                return k_mat;
            },
            py::arg("mat_x1"),
            py::arg("num_samples1"),
            py::arg("mat_x2"),
            py::arg("num_samples2"))
        .def(
            "compute_ktrain_with_gradient",
            [](T &self,
               const Eigen::Ref<const MatrixX> &mat_x,
               const long num_samples,
               const Eigen::VectorXl &vec_grad_flags,
               const MatrixX &mat_alpha) {
                const long n = mat_x.cols();
                if (n == 0) { return py::dict(); }
                const long dim = mat_x.rows();
                const long n_grad = vec_grad_flags.cast<long>().sum();
                const auto [rows, cols] = self.GetMinimumKtrainSize(n, n_grad, dim);
                Eigen::VectorXl vec_grad_flags_out = vec_grad_flags;
                MatrixX mat_k(rows, cols);
                MatrixX mat_alpha_out = mat_alpha;
                ERL_ASSERTM(mat_alpha.rows() == cols, "mat_alpha.rows() should be {}.", cols);
                (void) self.ComputeKtrainWithGradient(
                    mat_x,
                    num_samples,
                    vec_grad_flags_out,
                    mat_k,
                    mat_alpha_out);
                py::dict result;
                result["mat_k"] = mat_k;
                result["vec_grad_flags"] = vec_grad_flags_out;
                result["mat_alpha"] = mat_alpha_out;
                return result;
            },
            py::arg("mat_x"),
            py::arg("num_samples"),
            py::arg("vec_grad_flags"),
            py::arg("mat_alpha"))
        .def(
            "compute_ktrain_with_gradient",
            [](T &self,
               const Eigen::Ref<const MatrixX> &mat_x,
               const long num_samples,
               const Eigen::VectorXl &vec_grad_flags,
               const Eigen::Ref<const VectorX> &vec_var_x,
               const Eigen::Ref<const VectorX> &vec_var_y,
               const Eigen::Ref<const VectorX> &vec_var_grad,
               const MatrixX &mat_alpha) {
                const long n = mat_x.cols();
                if (n == 0) { return py::dict(); }
                const long dim = mat_x.rows();
                const long n_grad = vec_grad_flags.cast<long>().sum();
                const auto [rows, cols] = self.GetMinimumKtrainSize(n, n_grad, dim);
                Eigen::VectorXl vec_grad_flags_out = vec_grad_flags;
                MatrixX mat_k(rows, cols);
                MatrixX mat_alpha_out = mat_alpha;
                ERL_ASSERTM(mat_alpha.rows() == cols, "mat_alpha.rows() should be {}.", cols);
                (void) self.ComputeKtrainWithGradient(
                    mat_x,
                    num_samples,
                    vec_grad_flags_out,
                    vec_var_x,
                    vec_var_y,
                    vec_var_grad,
                    mat_k,
                    mat_alpha_out);
                py::dict result;
                result["mat_k"] = mat_k;
                result["vec_grad_flags"] = vec_grad_flags_out;
                result["mat_alpha"] = mat_alpha_out;
                return result;
            },
            py::arg("mat_x"),
            py::arg("num_samples"),
            py::arg("vec_grad_flags"),
            py::arg("vec_var_x"),
            py::arg("vec_var_y"),
            py::arg("vec_var_grad"),
            py::arg("mat_alpha"))
        .def(
            "compute_ktest_with_gradient",
            [](const T &self,
               const Eigen::Ref<const MatrixX> &mat_x1,
               const long num_samples1,
               const Eigen::Ref<const Eigen::VectorXl> &vec_grad1_flags,
               const Eigen::Ref<const MatrixX> &mat_x2,
               const long num_samples2,
               const bool predict_gradient) -> MatrixX {
                const long n1 = mat_x1.cols();
                const long n2 = mat_x2.cols();
                if (n1 == 0 || n2 == 0) { return {}; }
                const long dim = mat_x1.rows();
                const long n_grad = vec_grad1_flags.cast<long>().sum();
                const auto [rows, cols] =
                    self.GetMinimumKtestSize(n1, n_grad, dim, n2, predict_gradient);
                MatrixX k_mat(rows, cols);
                (void) self.ComputeKtestWithGradient(
                    mat_x1,
                    num_samples1,
                    vec_grad1_flags,
                    mat_x2,
                    num_samples2,
                    predict_gradient,
                    k_mat);
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
    BindYamlableBase(m);
    BindCovarianceImpl<double>(m, "CovarianceD");
    BindCovarianceImpl<float>(m, "CovarianceF");
}
