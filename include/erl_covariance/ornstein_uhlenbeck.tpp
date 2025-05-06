#pragma once

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
    std::pair<long, long>
    OrnsteinUhlenbeck<Dtype, Dim>::ComputeKtrain(
        const Eigen::Ref<const MatrixX> &mat_x,
        const long num_samples,
        MatrixX &mat_k,
        MatrixX &) {
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
        long dim;
        if constexpr (Dim == Eigen::Dynamic) {
            dim = mat_x.rows();
        } else {
            dim = Dim;
        }

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
                    r = std::sqrt(r);               // (mat_x.col(i) - mat_x.col(j)).norm();
                    mat_k(i, j) = std::exp(a * r);  // using single precision to improve performance
                    mat_k(j, i) = mat_k(i, j);
                }
            }
        }
        return {num_samples, num_samples};
    }

    template<typename Dtype, int Dim>
    [[nodiscard]] std::pair<long, long>
    OrnsteinUhlenbeck<Dtype, Dim>::ComputeKtrain(
        const Eigen::Ref<const MatrixX> &mat_x,
        const long num_samples,
        MatrixX &mat_k) {
        MatrixX mat_alpha;
        return ComputeKtrain(mat_x, num_samples, mat_k, mat_alpha);
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
        long dim;
        if constexpr (Dim == Eigen::Dynamic) {
            dim = mat_x.rows();
        } else {
            dim = Dim;
        }

        const Dtype a = -1.0f / Super::m_setting_->scale;
        for (long j = 0; j < num_samples; ++j) {
            Dtype *k_mat_j_ptr = mat_k.col(j).data();   // use raw pointer to improve performance
            const Dtype *xj_ptr = mat_x.col(j).data();  // use raw pointer to improve performance
            k_mat_j_ptr[j] = 1.0f + vec_var_y[j];       // mat_k(j, j)
            for (long i = j + 1; i < num_samples; ++i) {
                Dtype r = 0.0f;
                const Dtype *xi_ptr =
                    mat_x.col(i).data();  // use raw pointer to improve performance
                for (long k = 0; k < dim; ++k) {
                    const Dtype dx = xi_ptr[k] - xj_ptr[k];
                    r += dx * dx;
                }
                r = std::sqrt(r);              // (mat_x.col(i) - mat_x.col(j)).norm();
                Dtype &k_ij = k_mat_j_ptr[i];  // use reference to improve performance
                k_ij = std::exp(a * r);
                mat_k(j, i) = k_ij;
            }
        }
        return {num_samples, num_samples};
    }

    template<typename Dtype, int Dim>
    [[nodiscard]] std::pair<long, long>
    OrnsteinUhlenbeck<Dtype, Dim>::ComputeKtrain(
        const Eigen::Ref<const MatrixX> &mat_x,
        const Eigen::Ref<const VectorX> &vec_var_y,
        const long num_samples,
        MatrixX &mat_k) {
        MatrixX mat_alpha;
        return ComputeKtrain(mat_x, vec_var_y, num_samples, mat_k, mat_alpha);
    }

    template<typename Dtype, int Dim>
    std::pair<long, long>
    OrnsteinUhlenbeck<Dtype, Dim>::ComputeKtest(
        const Eigen::Ref<const MatrixX> &mat_x1,
        const long num_samples1,
        const Eigen::Ref<const MatrixX> &mat_x2,
        const long num_samples2,
        MatrixX &mat_k) const {
        ERL_DEBUG_ASSERT(
            mat_x1.rows() == mat_x2.rows(),
            "Sample vectors stored in x_1 and x_2 should have the same dimension.");
        ERL_DEBUG_ASSERT(
            mat_k.rows() >= num_samples1,
            "mat_k.rows() = {}, it should be >= {}.",
            mat_k.rows(),
            num_samples1);
        ERL_DEBUG_ASSERT(
            mat_k.cols() >= num_samples2,
            "mat_k.cols() = {}, it should be >= {}.",
            mat_k.cols(),
            num_samples2);
        long dim;
        if constexpr (Dim == Eigen::Dynamic) {
            dim = mat_x1.rows();
        } else {
            dim = Dim;
        }
        const Dtype a = -1.0f / Super::m_setting_->scale;
        for (long j = 0; j < num_samples2; ++j) {
            const Dtype *x2_ptr = mat_x2.col(j).data();  // use raw pointer to improve performance
            Dtype *col_j_ptr = mat_k.col(j).data();      // use raw pointer to improve performance
            for (long i = 0; i < num_samples1; ++i) {
                Dtype r = 0.0f;
                const Dtype *x1_ptr =
                    mat_x1.col(i).data();  // use raw pointer to improve performance
                for (long k = 0; k < dim; ++k) {
                    const Dtype dx = x1_ptr[k] - x2_ptr[k];
                    r += dx * dx;
                }
                r = std::sqrt(r);  // (mat_x1.col(i) - mat_x2.col(j)).norm();
                col_j_ptr[i] = std::exp(a * r);
            }
        }
        return {num_samples1, num_samples2};
    }

    template<typename Dtype, int Dim>
    std::pair<long, long>
    OrnsteinUhlenbeck<Dtype, Dim>::ComputeKtrainWithGradient(
        const Eigen::Ref<const MatrixX> &,
        long,
        Eigen::VectorXl &,
        MatrixX &,
        MatrixX &) {
        throw NotImplemented(__PRETTY_FUNCTION__);
    }

    template<typename Dtype, int Dim>
    [[nodiscard]] std::pair<long, long>
    OrnsteinUhlenbeck<Dtype, Dim>::ComputeKtrainWithGradient(
        const Eigen::Ref<const MatrixX> &mat_x,
        const long num_samples,
        Eigen::VectorXl &vec_grad_flags,
        MatrixX &mat_k) {
        MatrixX mat_alpha;
        return ComputeKtrainWithGradient(mat_x, num_samples, vec_grad_flags, mat_k, mat_alpha);
    }

    template<typename Dtype, int Dim>
    std::pair<long, long>
    OrnsteinUhlenbeck<Dtype, Dim>::ComputeKtrainWithGradient(
        const Eigen::Ref<const MatrixX> &,
        long,
        Eigen::VectorXl &,
        const Eigen::Ref<const VectorX> &,
        const Eigen::Ref<const VectorX> &,
        const Eigen::Ref<const VectorX> &,
        MatrixX &,
        MatrixX &) {
        throw NotImplemented(__PRETTY_FUNCTION__);
    }

    template<typename Dtype, int Dim>
    [[nodiscard]] std::pair<long, long>
    OrnsteinUhlenbeck<Dtype, Dim>::ComputeKtrainWithGradient(
        const Eigen::Ref<const MatrixX> &mat_x,
        const long num_samples,
        Eigen::VectorXl &vec_grad_flags,
        const Eigen::Ref<const VectorX> &vec_var_x,
        const Eigen::Ref<const VectorX> &vec_var_y,
        const Eigen::Ref<const VectorX> &vec_var_grad,
        MatrixX &mat_k) {
        MatrixX mat_alpha;
        return ComputeKtrainWithGradient(
            mat_x,
            num_samples,
            vec_grad_flags,
            vec_var_x,
            vec_var_y,
            vec_var_grad,
            mat_k,
            mat_alpha);
    }

    template<typename Dtype, int Dim>
    std::pair<long, long>
    OrnsteinUhlenbeck<Dtype, Dim>::ComputeKtestWithGradient(
        const Eigen::Ref<const MatrixX> &,
        long,
        const Eigen::Ref<const Eigen::VectorXl> &,
        const Eigen::Ref<const MatrixX> &,
        long,
        bool,
        MatrixX &) const {
        throw NotImplemented(__PRETTY_FUNCTION__);
    }
}  // namespace erl::covariance
