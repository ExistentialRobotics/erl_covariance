#pragma once

#include "covariance.hpp"

#include <cmath>

namespace erl::covariance {

    template<long Dim>
    class OrnsteinUhlenbeck : public Covariance {
        // ref1: https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
        // ref2: https://www.cs.cmu.edu/~epxing/Class/10708-15/notes/10708_scribe_lecture21.pdf

    public:
        explicit OrnsteinUhlenbeck(std::shared_ptr<Setting> setting)
            : Covariance(std::move(setting)) {
            if (Dim != Eigen::Dynamic) { m_setting_->x_dim = Dim; }  // set x_dim
        }

        [[nodiscard]] std::string
        GetCovarianceType() const override {
            if (Dim == Eigen::Dynamic) { return "OrnsteinUhlenbeckXd"; }
            return "OrnsteinUhlenbeck" + std::to_string(Dim) + "D";
        }

        [[nodiscard]] std::pair<long, long>
        ComputeKtrain(const Eigen::Ref<const Eigen::MatrixXd> &mat_x, const long num_samples, Eigen::MatrixXd &k_mat, Eigen::VectorXd & /*vec_y*/)
            const override {
            ERL_DEBUG_ASSERT(k_mat.rows() >= num_samples, "k_mat.rows() = {}, it should be >= {}.", k_mat.rows(), num_samples);
            ERL_DEBUG_ASSERT(k_mat.cols() >= num_samples, "k_mat.cols() = {}, it should be >= {}.", k_mat.cols(), num_samples);
            long dim;
            if constexpr (Dim == Eigen::Dynamic) {
                dim = mat_x.rows();
            } else {
                dim = Dim;
            }

            const double a = -1. / m_setting_->scale;
            const double alpha = m_setting_->alpha;
            for (long i = 0; i < num_samples; ++i) {
                for (long j = i; j < num_samples; ++j) {
                    if (i == j) {
                        k_mat(i, i) = alpha;
                    } else {
                        double r = 0.0;
                        for (long k = 0; k < dim; ++k) {
                            const double dx = mat_x(k, i) - mat_x(k, j);
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

        [[nodiscard]] std::pair<long, long>
        ComputeKtrain(
            const Eigen::Ref<const Eigen::MatrixXd> &mat_x,
            const Eigen::Ref<const Eigen::VectorXd> &vec_var_y,
            const long num_samples,
            Eigen::MatrixXd &k_mat,
            Eigen::VectorXd & /*vec_y*/) const override {
            ERL_DEBUG_ASSERT(k_mat.rows() >= num_samples, "k_mat.rows() = {}, it should be >= {}.", k_mat.rows(), num_samples);
            ERL_DEBUG_ASSERT(k_mat.cols() >= num_samples, "k_mat.cols() = {}, it should be >= {}.", k_mat.cols(), num_samples);
            ERL_DEBUG_ASSERT(vec_var_y.size() >= num_samples, "vec_var_y does not have enough elements, it should be >= {}.", num_samples);
            long dim;
            if constexpr (Dim == Eigen::Dynamic) {
                dim = mat_x.rows();
            } else {
                dim = Dim;
            }

            const double a = -1. / m_setting_->scale;
            const double alpha = m_setting_->alpha;
            for (long j = 0; j < num_samples; ++j) {
                double *k_mat_j_ptr = k_mat.col(j).data();   // use raw pointer to improve performance
                const double *xj_ptr = mat_x.col(j).data();  // use raw pointer to improve performance
                k_mat_j_ptr[j] = alpha + vec_var_y[j];       // k_mat(j, j)
                for (long i = j + 1; i < num_samples; ++i) {
                    double r = 0.0;
                    const double *xi_ptr = mat_x.col(i).data();  // use raw pointer to improve performance
                    for (long k = 0; k < dim; ++k) {
                        const double dx = xi_ptr[k] - xj_ptr[k];
                        r += dx * dx;
                    }
                    r = std::sqrt(r);               // (mat_x.col(i) - mat_x.col(j)).norm();
                    double &k_ij = k_mat_j_ptr[i];  // use reference to improve performance
                    k_ij = alpha * std::exp(a * r);
                    k_mat(j, i) = k_ij;
                }
            }
            return {num_samples, num_samples};
        }

        [[nodiscard]] std::pair<long, long>
        ComputeKtest(
            const Eigen::Ref<const Eigen::MatrixXd> &mat_x1,
            const long num_samples1,
            const Eigen::Ref<const Eigen::MatrixXd> &mat_x2,
            const long num_samples2,
            Eigen::MatrixXd &k_mat) const override {

            ERL_DEBUG_ASSERT(mat_x1.rows() == mat_x2.rows(), "Sample vectors stored in x_1 and x_2 should have the same dimension.");
            ERL_DEBUG_ASSERT(k_mat.rows() >= num_samples1, "k_mat.rows() = {}, it should be >= {}.", k_mat.rows(), num_samples1);
            ERL_DEBUG_ASSERT(k_mat.cols() >= num_samples2, "k_mat.cols() = {}, it should be >= {}.", k_mat.cols(), num_samples2);
            long dim;
            if constexpr (Dim == Eigen::Dynamic) {
                dim = mat_x1.rows();
            } else {
                dim = Dim;
            }

            const double a = -1. / m_setting_->scale;
            const double alpha = m_setting_->alpha;
            for (long j = 0; j < num_samples2; ++j) {
                const double *x2_ptr = mat_x2.col(j).data();  // use raw pointer to improve performance
                double *col_j_ptr = k_mat.col(j).data();      // use raw pointer to improve performance
                for (long i = 0; i < num_samples1; ++i) {
                    double r = 0.0;
                    const double *x1_ptr = mat_x1.col(i).data();  // use raw pointer to improve performance
                    for (long k = 0; k < dim; ++k) {
                        const double dx = x1_ptr[k] - x2_ptr[k];
                        r += dx * dx;
                    }
                    r = std::sqrt(r);  // (mat_x1.col(i) - mat_x2.col(j)).norm();
                    col_j_ptr[i] = alpha * std::exp(a * r);
                }
            }
            return {num_samples1, num_samples2};
        }

        [[nodiscard]] std::pair<long, long>
        ComputeKtrainWithGradient(const Eigen::Ref<const Eigen::MatrixXd> &, long, Eigen::VectorXl &, Eigen::MatrixXd &, Eigen::VectorXd &) const override {
            throw NotImplemented(__PRETTY_FUNCTION__);
        }

        [[nodiscard]] std::pair<long, long>
        ComputeKtrainWithGradient(
            const Eigen::Ref<const Eigen::MatrixXd> &,
            long,
            Eigen::VectorXl &,
            const Eigen::Ref<const Eigen::VectorXd> &,
            const Eigen::Ref<const Eigen::VectorXd> &,
            const Eigen::Ref<const Eigen::VectorXd> &,
            Eigen::MatrixXd &,
            Eigen::VectorXd &) const override {
            throw NotImplemented(__PRETTY_FUNCTION__);
        }

        [[nodiscard]] std::pair<long, long>
        ComputeKtestWithGradient(
            const Eigen::Ref<const Eigen::MatrixXd> &,
            long,
            const Eigen::Ref<const Eigen::VectorXl> &,
            const Eigen::Ref<const Eigen::MatrixXd> &,
            long,
            bool,
            Eigen::MatrixXd &) const override {
            throw NotImplemented(__PRETTY_FUNCTION__);
        }
    };

    using OrnsteinUhlenbeck1D = OrnsteinUhlenbeck<1>;
    using OrnsteinUhlenbeck2D = OrnsteinUhlenbeck<2>;
    using OrnsteinUhlenbeck3D = OrnsteinUhlenbeck<3>;
    using OrnsteinUhlenbeckXd = OrnsteinUhlenbeck<Eigen::Dynamic>;

    ERL_REGISTER_COVARIANCE(OrnsteinUhlenbeck1D);
    ERL_REGISTER_COVARIANCE(OrnsteinUhlenbeck2D);
    ERL_REGISTER_COVARIANCE(OrnsteinUhlenbeck3D);
    ERL_REGISTER_COVARIANCE(OrnsteinUhlenbeckXd);
}  // namespace erl::covariance
