#pragma once

#include "erl_common/exception.hpp"

#include <cmath>

namespace erl::covariance {
    template<int Dim, typename Dtype>
    std::pair<long, long>
    OrnsteinUhlenbeck<Dim, Dtype>::ComputeKtrain(const Eigen::Ref<const MatrixX> &mat_x, const long num_samples, MatrixX &k_mat, VectorX & /*vec_y*/) const {
        ERL_DEBUG_ASSERT(k_mat.rows() >= num_samples, "k_mat.rows() = {}, it should be >= {}.", k_mat.rows(), num_samples);
        ERL_DEBUG_ASSERT(k_mat.cols() >= num_samples, "k_mat.cols() = {}, it should be >= {}.", k_mat.cols(), num_samples);
        long dim;
        if constexpr (Dim == Eigen::Dynamic) {
            dim = mat_x.rows();
        } else {
            dim = Dim;
        }

        const Dtype a = -1. / Super::m_setting_->scale;
        const Dtype alpha = Super::m_setting_->alpha;
        for (long i = 0; i < num_samples; ++i) {
            for (long j = i; j < num_samples; ++j) {
                if (i == j) {
                    k_mat(i, i) = alpha;
                } else {
                    Dtype r = 0.0;
                    for (long k = 0; k < dim; ++k) {
                        const Dtype dx = mat_x(k, i) - mat_x(k, j);
                        r += dx * dx;
                    }
                    r = std::sqrt(r);                       // (mat_x.col(i) - mat_x.col(j)).norm();
                    k_mat(i, j) = alpha * std::exp(a * r);  // using single precision to improve performance
                    k_mat(j, i) = k_mat(i, j);
                }
            }
        }
        return {num_samples, num_samples};
    }

    template<int Dim, typename Dtype>
    std::pair<long, long>
    OrnsteinUhlenbeck<Dim, Dtype>::ComputeKtrain(
        const Eigen::Ref<const MatrixX> &mat_x,
        const Eigen::Ref<const VectorX> &vec_var_y,
        const long num_samples,
        MatrixX &k_mat,
        VectorX & /*vec_y*/) const {
        ERL_DEBUG_ASSERT(k_mat.rows() >= num_samples, "k_mat.rows() = {}, it should be >= {}.", k_mat.rows(), num_samples);
        ERL_DEBUG_ASSERT(k_mat.cols() >= num_samples, "k_mat.cols() = {}, it should be >= {}.", k_mat.cols(), num_samples);
        ERL_DEBUG_ASSERT(vec_var_y.size() >= num_samples, "vec_var_y does not have enough elements, it should be >= {}.", num_samples);
        long dim;
        if constexpr (Dim == Eigen::Dynamic) {
            dim = mat_x.rows();
        } else {
            dim = Dim;
        }

        const Dtype a = -1. / Super::m_setting_->scale;
        const Dtype alpha = Super::m_setting_->alpha;
        for (long j = 0; j < num_samples; ++j) {
            Dtype *k_mat_j_ptr = k_mat.col(j).data();   // use raw pointer to improve performance
            const Dtype *xj_ptr = mat_x.col(j).data();  // use raw pointer to improve performance
            k_mat_j_ptr[j] = alpha + vec_var_y[j];      // k_mat(j, j)
            for (long i = j + 1; i < num_samples; ++i) {
                Dtype r = 0.0;
                const Dtype *xi_ptr = mat_x.col(i).data();  // use raw pointer to improve performance
                for (long k = 0; k < dim; ++k) {
                    const Dtype dx = xi_ptr[k] - xj_ptr[k];
                    r += dx * dx;
                }
                r = std::sqrt(r);              // (mat_x.col(i) - mat_x.col(j)).norm();
                Dtype &k_ij = k_mat_j_ptr[i];  // use reference to improve performance
                k_ij = alpha * std::exp(a * r);
                k_mat(j, i) = k_ij;
            }
        }
        return {num_samples, num_samples};
    }

    template<int Dim, typename Dtype>
    std::pair<long, long>
    OrnsteinUhlenbeck<Dim, Dtype>::ComputeKtest(
        const Eigen::Ref<const MatrixX> &mat_x1,
        const long num_samples1,
        const Eigen::Ref<const MatrixX> &mat_x2,
        const long num_samples2,
        MatrixX &k_mat) const {

        ERL_DEBUG_ASSERT(mat_x1.rows() == mat_x2.rows(), "Sample vectors stored in x_1 and x_2 should have the same dimension.");
        ERL_DEBUG_ASSERT(k_mat.rows() >= num_samples1, "k_mat.rows() = {}, it should be >= {}.", k_mat.rows(), num_samples1);
        ERL_DEBUG_ASSERT(k_mat.cols() >= num_samples2, "k_mat.cols() = {}, it should be >= {}.", k_mat.cols(), num_samples2);
        long dim;
        if constexpr (Dim == Eigen::Dynamic) {
            dim = mat_x1.rows();
        } else {
            dim = Dim;
        }

        const Dtype a = -1. / Super::m_setting_->scale;
        const Dtype alpha = Super::m_setting_->alpha;
        for (long j = 0; j < num_samples2; ++j) {
            const Dtype *x2_ptr = mat_x2.col(j).data();  // use raw pointer to improve performance
            Dtype *col_j_ptr = k_mat.col(j).data();      // use raw pointer to improve performance
            for (long i = 0; i < num_samples1; ++i) {
                Dtype r = 0.0;
                const Dtype *x1_ptr = mat_x1.col(i).data();  // use raw pointer to improve performance
                for (long k = 0; k < dim; ++k) {
                    const Dtype dx = x1_ptr[k] - x2_ptr[k];
                    r += dx * dx;
                }
                r = std::sqrt(r);  // (mat_x1.col(i) - mat_x2.col(j)).norm();
                col_j_ptr[i] = alpha * std::exp(a * r);
            }
        }
        return {num_samples1, num_samples2};
    }

    template<int Dim, typename Dtype>
    std::pair<long, long>
    OrnsteinUhlenbeck<Dim, Dtype>::ComputeKtrainWithGradient(const Eigen::Ref<const MatrixX> &, long, Eigen::VectorXl &, MatrixX &, VectorX &) const {
        throw NotImplemented(__PRETTY_FUNCTION__);
    }

    template<int Dim, typename Dtype>
    std::pair<long, long>
    OrnsteinUhlenbeck<Dim, Dtype>::ComputeKtrainWithGradient(
        const Eigen::Ref<const MatrixX> &,
        long,
        Eigen::VectorXl &,
        const Eigen::Ref<const VectorX> &,
        const Eigen::Ref<const VectorX> &,
        const Eigen::Ref<const VectorX> &,
        MatrixX &,
        VectorX &) const {
        throw NotImplemented(__PRETTY_FUNCTION__);
    }

    template<int Dim, typename Dtype>
    std::pair<long, long>
    OrnsteinUhlenbeck<Dim, Dtype>::ComputeKtestWithGradient(
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
