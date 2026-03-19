#include "erl_covariance/ornstein_uhlenbeck.hpp"

#include "erl_common/exception.hpp"

#include <cmath>

namespace erl::covariance {
    template<typename Dtype, int Dim>
    OrnsteinUhlenbeck<Dtype, Dim>::OrnsteinUhlenbeck(std::shared_ptr<Setting> setting)
        : Super(std::move(setting)) {
        if (Dim != Eigen::Dynamic) {
            ERL_WARN_ONCE_COND(
                Super::m_setting_->x_dim != Dim,
                "x_dim will change from {} to {}.",
                Super::m_setting_->x_dim,
                Dim);
            Super::m_setting_->x_dim = Dim;
        } else {
            ERL_DEBUG_ASSERT(Super::m_setting_->x_dim == Dim, "x_dim should be {}.", Dim);
        }
    }

    template<typename Dtype, int Dim>
    [[nodiscard]] std::string
    OrnsteinUhlenbeck<Dtype, Dim>::GetCovarianceType() const {
        return type_name<OrnsteinUhlenbeck>();
    }

    template<typename Dtype, int Dim>
    [[nodiscard]] std::string
    OrnsteinUhlenbeck<Dtype, Dim>::GetCovarianceName() const {
        return "OrnsteinUhlenbeck";
    }

    template<typename Dtype, int Dim>
    Dtype
    OrnsteinUhlenbeck<Dtype, Dim>::GetHessianScaleFactor() const {
        ERL_FATAL("Ornstein-Uhlenbeck covariance function does not support Hessian computation.");
        return 0.0f;
    }

    template<typename Dtype, int Dim>
    std::pair<long, long>
    OrnsteinUhlenbeck<Dtype, Dim>::ComputeKtrain(
        const Eigen::Ref<const MatrixX> &mat_x,
        const long num_samples,
        MatrixX &mat_k,
        MatrixX & /*mat_alpha*/) {

        ERL_DEBUG_ASSERT(
            mat_k.rows() >= num_samples,
            "mat_k.rows() = {}, it should be >= {}.",
            mat_k.rows(),
            num_samples);
        ERL_DEBUG_ASSERT(
            mat_k.cols() >= num_samples,
            "mat_k.cols() = {}, it should be >= {}.",
            mat_k.cols(),
            num_samples);
        const long dim = (Dim == Eigen::Dynamic) ? mat_x.rows() : Dim;
        const Dtype a = -1.0f / Super::m_setting_->scale;
        for (long i = 0; i < num_samples; ++i) {
            for (long j = i; j < num_samples; ++j) {
                if (i == j) {
                    mat_k(i, i) = 1.0f;
                } else {
                    Dtype r = 0.0f;
                    for (long k = 0; k < dim; ++k) {
                        const Dtype dx = mat_x(k, i) - mat_x(k, j);
                        r += dx * dx;
                    }
                    r = std::sqrt(r);  // (mat_x.col(i) - mat_x.col(j)).norm();
                    mat_k(i, j) = std::exp(a * r);
                    mat_k(j, i) = mat_k(i, j);
                }
            }
        }
        return {num_samples, num_samples};
    }

    template<typename Dtype, int Dim>
    std::pair<long, long>
    OrnsteinUhlenbeck<Dtype, Dim>::ComputeKtrain(
        const Eigen::Ref<const MatrixX> &mat_x,
        const long num_samples,
        const Dtype exp_bias,
        MatrixX &mat_k,
        MatrixX & /*mat_alpha*/) {

        ERL_DEBUG_ASSERT(
            mat_k.rows() >= num_samples,
            "mat_k.rows() = {}, it should be >= {}.",
            mat_k.rows(),
            num_samples);
        ERL_DEBUG_ASSERT(
            mat_k.cols() >= num_samples,
            "mat_k.cols() = {}, it should be >= {}.",
            mat_k.cols(),
            num_samples);
        const long dim = (Dim == Eigen::Dynamic) ? mat_x.rows() : Dim;
        const Dtype a = -1.0f / Super::m_setting_->scale;
        for (long i = 0; i < num_samples; ++i) {
            for (long j = i; j < num_samples; ++j) {
                if (i == j) {
                    mat_k(i, i) = 1.0f;
                } else {
                    Dtype r = 0.0f;
                    for (long k = 0; k < dim; ++k) {
                        const Dtype dx = mat_x(k, i) - mat_x(k, j);
                        r += dx * dx;
                    }
                    r = std::sqrt(r);  // (mat_x.col(i) - mat_x.col(j)).norm();
                    mat_k(i, j) = std::exp(a * r + exp_bias);
                    mat_k(j, i) = mat_k(i, j);
                }
            }
        }
        return {num_samples, num_samples};
    }

    template<typename Dtype, int Dim>
    std::pair<long, long>
    OrnsteinUhlenbeck<Dtype, Dim>::ComputeKtrain(
        const Eigen::Ref<const MatrixX> &mat_x,
        const Eigen::Ref<const VectorX> &vec_var_y,
        const long num_samples,
        MatrixX &mat_k,
        MatrixX & /*mat_alpha*/) {

        ERL_DEBUG_ASSERT(
            mat_k.rows() >= num_samples,
            "mat_k.rows() = {}, it should be >= {}.",
            mat_k.rows(),
            num_samples);
        ERL_DEBUG_ASSERT(
            mat_k.cols() >= num_samples,
            "mat_k.cols() = {}, it should be >= {}.",
            mat_k.cols(),
            num_samples);
        ERL_DEBUG_ASSERT(
            vec_var_y.size() >= num_samples,
            "vec_var_y.size() = {}, it should be >= {}.",
            vec_var_y.size(),
            num_samples);
        const long dim = (Dim == Eigen::Dynamic) ? mat_x.rows() : Dim;
        const Dtype a = -1.0f / Super::m_setting_->scale;
        const long stride = mat_k.outerStride();
        for (long j = 0; j < num_samples; ++j) {
            Dtype *mat_k_j_ptr = mat_k.col(j).data();
            const Dtype *xj_ptr = mat_x.col(j).data();
            mat_k_j_ptr[j] = 1.0f + vec_var_y[j];     // mat_k(j, j)
            Dtype *k_ji_ptr = &mat_k(j, j) + stride;  // mat_k(j, i)
            for (long i = j + 1; i < num_samples; ++i, k_ji_ptr += stride) {
                const Dtype *xi_ptr = mat_x.col(i).data();
                Dtype r = 0.0f;
                for (long k = 0; k < dim; ++k) {
                    const Dtype dx = xi_ptr[k] - xj_ptr[k];
                    r += dx * dx;
                }
                r = std::sqrt(r);              // (mat_x.col(i) - mat_x.col(j)).norm();
                Dtype &k_ij = mat_k_j_ptr[i];  // mat_k(i, j)
                k_ij = std::exp(a * r);
                *k_ji_ptr = k_ij;  // mat_k(j, i) = k_ij;
            }
        }
        return {num_samples, num_samples};
    }

    template<typename Dtype, int Dim>
    std::pair<long, long>
    OrnsteinUhlenbeck<Dtype, Dim>::ComputeKtrain(
        const Eigen::Ref<const MatrixX> &mat_x,
        const Eigen::Ref<const VectorX> &vec_var_y,
        const long num_samples,
        const Dtype exp_bias,
        MatrixX &mat_k,
        MatrixX & /*mat_alpha*/) {

        ERL_DEBUG_ASSERT(
            mat_k.rows() >= num_samples,
            "mat_k.rows() = {}, it should be >= {}.",
            mat_k.rows(),
            num_samples);
        ERL_DEBUG_ASSERT(
            mat_k.cols() >= num_samples,
            "mat_k.cols() = {}, it should be >= {}.",
            mat_k.cols(),
            num_samples);
        ERL_DEBUG_ASSERT(
            vec_var_y.size() >= num_samples,
            "vec_var_y.size() = {}, it should be >= {}.",
            vec_var_y.size(),
            num_samples);
        const long dim = (Dim == Eigen::Dynamic) ? mat_x.rows() : Dim;
        const Dtype a = -1.0f / Super::m_setting_->scale;
        const Dtype b = std::exp(exp_bias);
        const long stride = mat_k.outerStride();
        for (long j = 0; j < num_samples; ++j) {
            Dtype *mat_k_j_ptr = mat_k.col(j).data();
            const Dtype *xj_ptr = mat_x.col(j).data();
            mat_k_j_ptr[j] = b + vec_var_y[j];        // mat_k(j, j)
            Dtype *k_ji_ptr = &mat_k(j, j) + stride;  // mat_k(j, i)
            for (long i = j + 1; i < num_samples; ++i, k_ji_ptr += stride) {
                const Dtype *xi_ptr = mat_x.col(i).data();
                Dtype r = 0.0f;
                for (long k = 0; k < dim; ++k) {
                    const Dtype dx = xi_ptr[k] - xj_ptr[k];
                    r += dx * dx;
                }
                r = std::sqrt(r);              // (mat_x.col(i) - mat_x.col(j)).norm();
                Dtype &k_ij = mat_k_j_ptr[i];  // mat_k(i, j)
                k_ij = std::exp(a * r + exp_bias);
                *k_ji_ptr = k_ij;  // mat_k(j, i) = k_ij;
            }
        }
        return {num_samples, num_samples};
    }

    template<typename Dtype, int Dim>
    std::pair<long, long>
    OrnsteinUhlenbeck<Dtype, Dim>::ComputeKtest(
        const Eigen::Ref<const MatrixX> &mat_x1,
        const long num_samples,
        const Eigen::Ref<const MatrixX> &mat_x2,
        const long num_queries,
        MatrixX &mat_k) const {

        ERL_DEBUG_ASSERT(
            mat_x1.rows() == mat_x2.rows(),
            "Sample vectors stored in x1 and x2 should have the same dimension.");
        ERL_DEBUG_ASSERT(
            mat_k.rows() >= num_samples,
            "mat_k.rows() = {}, it should be >= {}.",
            mat_k.rows(),
            num_samples);
        ERL_DEBUG_ASSERT(
            mat_k.cols() >= num_queries,
            "mat_k.cols() = {}, it should be >= {}.",
            mat_k.cols(),
            num_queries);
        const long dim = (Dim == Eigen::Dynamic) ? mat_x1.rows() : Dim;
        const Dtype a = -1.0f / Super::m_setting_->scale;
        for (long j = 0; j < num_queries; ++j) {
            const Dtype *x2_ptr = mat_x2.col(j).data();
            Dtype *col_j_ptr = mat_k.col(j).data();
            for (long i = 0; i < num_samples; ++i) {
                const Dtype *x1_ptr = mat_x1.col(i).data();
                Dtype r = 0.0f;
                for (long k = 0; k < dim; ++k) {
                    const Dtype dx = x1_ptr[k] - x2_ptr[k];
                    r += dx * dx;
                }
                r = std::sqrt(r);  // (mat_x1.col(i) - mat_x2.col(j)).norm();
                col_j_ptr[i] = std::exp(a * r);
            }
        }
        return {num_samples, num_queries};
    }

    template<typename Dtype, int Dim>
    std::pair<long, long>
    OrnsteinUhlenbeck<Dtype, Dim>::ComputeKtest(
        const Eigen::Ref<const MatrixX> &mat_x1,
        long num_samples,
        const Eigen::Ref<const MatrixX> &mat_x2,
        long num_queries,
        Dtype exp_bias,
        MatrixX &mat_k) const {

        ERL_DEBUG_ASSERT(
            mat_x1.rows() == mat_x2.rows(),
            "Sample vectors stored in x1 and x2 should have the same dimension.");
        ERL_DEBUG_ASSERT(
            mat_k.rows() >= num_samples,
            "mat_k.rows() = {}, it should be >= {}.",
            mat_k.rows(),
            num_samples);
        ERL_DEBUG_ASSERT(
            mat_k.cols() >= num_queries,
            "mat_k.cols() = {}, it should be >= {}.",
            mat_k.cols(),
            num_queries);
        const long dim = (Dim == Eigen::Dynamic) ? mat_x1.rows() : Dim;
        const Dtype a = -1.0f / Super::m_setting_->scale;
        for (long j = 0; j < num_queries; ++j) {
            const Dtype *x2_ptr = mat_x2.col(j).data();
            Dtype *col_j_ptr = mat_k.col(j).data();
            for (long i = 0; i < num_samples; ++i) {
                const Dtype *x1_ptr = mat_x1.col(i).data();
                Dtype r = 0.0f;
                for (long k = 0; k < dim; ++k) {
                    const Dtype dx = x1_ptr[k] - x2_ptr[k];
                    r += dx * dx;
                }
                r = std::sqrt(r);  // (mat_x1.col(i) - mat_x2.col(j)).norm();
                col_j_ptr[i] = std::exp(a * r + exp_bias);
            }
        }
        return {num_samples, num_queries};
    }

    template<typename Dtype, int Dim>
    std::pair<long, long>
    OrnsteinUhlenbeck<Dtype, Dim>::ComputeKtrainWithGradient(
        const Eigen::Ref<const MatrixX> & /*mat_x*/,
        long /*num_samples*/,
        Eigen::VectorXl & /*vec_grad_flags*/,
        MatrixX & /*mat_k*/,
        MatrixX & /*mat_alpha*/) {
        throw NotImplemented(__PRETTY_FUNCTION__);
    }

    template<typename Dtype, int Dim>
    std::pair<long, long>
    OrnsteinUhlenbeck<Dtype, Dim>::ComputeKtrainWithGradient(
        const Eigen::Ref<const MatrixX> & /*mat_x*/,
        long /*num_samples*/,
        Dtype /*exp_bias*/,
        Eigen::VectorXl & /*vec_grad_flags*/,
        MatrixX & /*mat_k*/,
        MatrixX & /*mat_alpha*/) {
        throw NotImplemented(__PRETTY_FUNCTION__);
    }

    template<typename Dtype, int Dim>
    std::pair<long, long>
    OrnsteinUhlenbeck<Dtype, Dim>::ComputeKtrainWithGradient(
        const Eigen::Ref<const MatrixX> & /*mat_x*/,
        const long /*num_samples*/,
        Eigen::VectorXl & /*vec_grad_flags*/,
        const Eigen::Ref<const VectorX> & /*vec_var_x*/,
        const Eigen::Ref<const VectorX> & /*vec_var_y*/,
        const Eigen::Ref<const VectorX> & /*vec_var_grad*/,
        MatrixX & /*mat_k*/,
        MatrixX & /*mat_alpha*/) {
        throw NotImplemented(__PRETTY_FUNCTION__);
    }

    template<typename Dtype, int Dim>
    std::pair<long, long>
    OrnsteinUhlenbeck<Dtype, Dim>::ComputeKtrainWithGradient(
        const Eigen::Ref<const MatrixX> & /*mat_x*/,
        const long /*num_samples*/,
        const Dtype /*exp_bias*/,
        Eigen::VectorXl & /*vec_grad_flags*/,
        const Eigen::Ref<const VectorX> & /*vec_var_x*/,
        const Eigen::Ref<const VectorX> & /*vec_var_y*/,
        const Eigen::Ref<const VectorX> & /*vec_var_grad*/,
        MatrixX & /*mat_k*/,
        MatrixX & /*mat_alpha*/) {
        throw NotImplemented(__PRETTY_FUNCTION__);
    }

    template<typename Dtype, int Dim>
    std::pair<long, long>
    OrnsteinUhlenbeck<Dtype, Dim>::ComputeKtestWithGradient(
        const Eigen::Ref<const MatrixX> & /*mat_x1*/,
        long /*num_samples*/,
        const Eigen::Ref<const Eigen::VectorXl> & /*vec_grad1_flags*/,
        const Eigen::Ref<const MatrixX> & /*mat_x2*/,
        long /*num_queries*/,
        bool /*predict_gradient*/,
        MatrixX & /*mat_k*/) const {
        throw NotImplemented(__PRETTY_FUNCTION__);
    }

    template<typename Dtype, int Dim>
    std::pair<long, long>
    OrnsteinUhlenbeck<Dtype, Dim>::ComputeKtestWithGradient(
        const Eigen::Ref<const MatrixX> & /*mat_x1*/,
        long /*num_samples*/,
        const Eigen::Ref<const Eigen::VectorXl> & /*vec_grad1_flags*/,
        const Eigen::Ref<const MatrixX> & /*mat_x2*/,
        long /*num_queries*/,
        bool /*predict_gradient*/,
        Dtype /*exp_bias*/,
        MatrixX & /*mat_k*/) const {
        throw NotImplemented(__PRETTY_FUNCTION__);
    }

    template class OrnsteinUhlenbeck<double, 1>;
    template class OrnsteinUhlenbeck<double, 2>;
    template class OrnsteinUhlenbeck<double, 3>;
    template class OrnsteinUhlenbeck<double, Eigen::Dynamic>;

    template class OrnsteinUhlenbeck<float, 1>;
    template class OrnsteinUhlenbeck<float, 2>;
    template class OrnsteinUhlenbeck<float, 3>;
    template class OrnsteinUhlenbeck<float, Eigen::Dynamic>;

}  // namespace erl::covariance
