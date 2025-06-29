#include "erl_covariance/radial_bias_function.hpp"

namespace erl::covariance {
    template<typename Dtype, int Dim>
    RadialBiasFunction<Dtype, Dim>::RadialBiasFunction(std::shared_ptr<Setting> setting)
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
    std::string
    RadialBiasFunction<Dtype, Dim>::GetCovarianceType() const {
        return type_name<RadialBiasFunction>();
    }

    template<typename Dtype, int Dim>
    std::pair<long, long>
    RadialBiasFunction<Dtype, Dim>::ComputeKtrain(
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

        const Dtype a = 0.5f / (Super::m_setting_->scale * Super::m_setting_->scale);
        const long stride = mat_k.outerStride();
        for (long j = 0; j < num_samples; ++j) {
            Dtype *mat_k_j_ptr = mat_k.col(j).data();
            const Dtype *xj_ptr = mat_x.col(j).data();
            mat_k_j_ptr[j] = 1.0f;  // mat_k(j, j)
            if (j + 1 >= num_samples) { continue; }
            Dtype *k_ji_ptr = &mat_k(j, j + 1);  // mat_k(j, i)
            for (long i = j + 1; i < num_samples; ++i, k_ji_ptr += stride) {
                const Dtype *xi_ptr = mat_x.col(i).data();
                Dtype r = 0.0f;
                if constexpr (Dim == Eigen::Dynamic) {
                    const long dim = mat_x.rows();
                    for (long k = 0; k < dim; ++k) {
                        const Dtype dx = xi_ptr[k] - xj_ptr[k];
                        r += dx * dx;
                    }
                } else {
                    for (long k = 0; k < Dim; ++k) {
                        const Dtype dx = xi_ptr[k] - xj_ptr[k];
                        r += dx * dx;
                    }
                }
                Dtype &k_ij = mat_k_j_ptr[i];
                k_ij = std::exp(-a * r);  // mat_k(i, j)
                *k_ji_ptr = k_ij;         // mat_k(j, i) = k_ij;
            }
        }
        return {num_samples, num_samples};
    }

    template<typename Dtype, int Dim>
    std::pair<long, long>
    RadialBiasFunction<Dtype, Dim>::ComputeKtrain(
        const Eigen::Ref<const MatrixX> &mat_x,
        const long num_samples,
        MatrixX &mat_k) {
        MatrixX mat_alpha;
        return ComputeKtrain(mat_x, num_samples, mat_k, mat_alpha);
    }

    template<typename Dtype, int Dim>
    std::pair<long, long>
    RadialBiasFunction<Dtype, Dim>::ComputeKtrain(
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

        const Dtype a = 0.5f / (Super::m_setting_->scale * Super::m_setting_->scale);
        const long stride = mat_k.outerStride();
        for (long j = 0; j < num_samples; ++j) {
            Dtype *mat_k_j_ptr = mat_k.col(j).data();   // use raw pointer to improve performance
            const Dtype *xj_ptr = mat_x.col(j).data();  // use raw pointer to improve performance
            mat_k_j_ptr[j] = 1.0f + vec_var_y[j];       // mat_k(j, j)
            if (j + 1 >= num_samples) { continue; }
            Dtype *k_ji_ptr = &mat_k(j, j + 1);  // mat_k(j, i)
            for (long i = j + 1; i < num_samples; ++i, k_ji_ptr += stride) {
                const Dtype *xi_ptr = mat_x.col(i).data();
                Dtype r = 0.0f;
                if constexpr (Dim == Eigen::Dynamic) {
                    const long dim = mat_x.rows();
                    for (long k = 0; k < dim; ++k) {
                        const Dtype dx = xi_ptr[k] - xj_ptr[k];
                        r += dx * dx;
                    }
                } else {
                    for (long k = 0; k < Dim; ++k) {
                        const Dtype dx = xi_ptr[k] - xj_ptr[k];
                        r += dx * dx;
                    }
                }
                Dtype &k_ij = mat_k_j_ptr[i];  // mat_k(i, j)
                k_ij = std::exp(-a * r);
                *k_ji_ptr = k_ij;  // mat_k(j, i) = k_ij;
            }
        }
        return {num_samples, num_samples};
    }

    template<typename Dtype, int Dim>
    std::pair<long, long>
    RadialBiasFunction<Dtype, Dim>::ComputeKtrain(
        const Eigen::Ref<const MatrixX> &mat_x,
        const Eigen::Ref<const VectorX> &vec_var_y,
        const long num_samples,
        MatrixX &mat_k) {
        MatrixX mat_alpha;
        return ComputeKtrain(mat_x, vec_var_y, num_samples, mat_k, mat_alpha);
    }

    template<typename Dtype, int Dim>
    std::pair<long, long>
    RadialBiasFunction<Dtype, Dim>::ComputeKtest(
        const Eigen::Ref<const MatrixX> &mat_x1,
        const long num_samples1,
        const Eigen::Ref<const MatrixX> &mat_x2,
        const long num_samples2,
        MatrixX &mat_k) const {
        ERL_DEBUG_ASSERT(
            mat_x1.rows() == mat_x2.rows(),
            "Sample vectors stored in x1 and x2 should have the same dimension.");
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

        const Dtype a = 0.5f / (Super::m_setting_->scale * Super::m_setting_->scale);
        for (long j = 0; j < num_samples2; ++j) {
            const Dtype *x2_ptr = mat_x2.col(j).data();
            Dtype *col_j_ptr = mat_k.col(j).data();
            for (long i = 0; i < num_samples1; ++i) {
                const Dtype *x1_ptr = mat_x1.col(i).data();
                Dtype r = 0.0f;
                if constexpr (Dim == Eigen::Dynamic) {
                    const long dim = mat_x1.rows();
                    for (long k = 0; k < dim; ++k) {
                        const Dtype dx = x1_ptr[k] - x2_ptr[k];
                        r += dx * dx;
                    }
                } else {
                    for (long k = 0; k < Dim; ++k) {
                        const Dtype dx = x1_ptr[k] - x2_ptr[k];
                        r += dx * dx;
                    }
                }
                col_j_ptr[i] = std::exp(-a * r);
            }
        }
        return {num_samples1, num_samples2};
    }

    template<typename Dtype, int Dim>
    std::pair<long, long>
    RadialBiasFunction<Dtype, Dim>::ComputeKtestSparse(
        const Eigen::Ref<const MatrixX> &mat_x1,
        const long num_samples1,
        const Eigen::Ref<const MatrixX> &mat_x2,
        const long num_samples2,
        Dtype zero_threshold,
        SparseMatrix &mat_k) const {

        const Dtype a = 0.5f / (Super::m_setting_->scale * Super::m_setting_->scale);
        const Dtype threshold = std::log(zero_threshold);
        mat_k.setZero();  // clear the sparse matrix
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
        for (long j = 0; j < num_samples2; ++j) {
            const Dtype *x2_ptr = mat_x2.col(j).data();
            for (long i = 0; i < num_samples1; ++i) {
                const Dtype *x1_ptr = mat_x1.col(i).data();
                Dtype r = 0.0f;
                if constexpr (Dim == Eigen::Dynamic) {
                    const long dim = mat_x1.rows();
                    for (long k = 0; k < dim; ++k) {
                        const Dtype dx = x1_ptr[k] - x2_ptr[k];
                        r += dx * dx;
                    }
                } else {
                    for (long k = 0; k < Dim; ++k) {
                        const Dtype dx = x1_ptr[k] - x2_ptr[k];
                        r += dx * dx;
                    }
                }
                r *= -a;
                if (r < threshold) { continue; }
                mat_k.insert(i, j) = std::exp(r);
            }
        }
        return {num_samples1, num_samples2};
    }

    template<typename Dtype, int Dim>
    std::pair<long, long>
    RadialBiasFunction<Dtype, Dim>::ComputeKtrainWithGradient(
        const Eigen::Ref<const MatrixX> &mat_x,
        const long num_samples,
        Eigen::VectorXl &vec_grad_flags,
        MatrixX &mat_k,
        MatrixX & /*mat_alpha*/) {

        long dim;
        if constexpr (Dim == Eigen::Dynamic) {
            dim = mat_x.rows();
        } else {
            dim = Dim;
        }

        ERL_DEBUG_ASSERT(mat_x.rows() == dim, "Each column of mat_x should be {}-D vector.", dim);
        long n_grad = 0;
        long *grad_flags = vec_grad_flags.data();
        for (long i = 0; i < num_samples; ++i) {
            if (long &flag = grad_flags[i]; flag > 0) { flag = num_samples + n_grad++; }
        }
        long n_rows = num_samples + n_grad * dim;
        long n_cols = n_rows;
        ERL_DEBUG_ASSERT(
            mat_k.rows() >= n_rows,
            "mat_k.rows() = {}, it should be >= {}.",
            mat_k.rows(),
            n_rows);
        ERL_DEBUG_ASSERT(
            mat_k.cols() >= n_cols,
            "mat_k.cols() = {}, it should be >= {}.",
            mat_k.cols(),
            n_cols);

        const Dtype l2_inv = 1.0f / (Super::m_setting_->scale * Super::m_setting_->scale);
        const Dtype a = 0.5f * l2_inv;
        // buffer to store the difference between x1_i and x2_j
        Eigen::Vector<Dtype, Dim> diff_ij;          // avoid memory allocation on the heap
        Eigen::Vector<Dtype *, Dim> mat_k_kj_ptrs;  // avoid memory allocation on the heap
        Eigen::Vector<Dtype *, Dim> mat_k_ki_ptrs;  // avoid memory allocation on the heap
        if constexpr (Dim == Eigen::Dynamic) {
            diff_ij.resize(dim);
            mat_k_kj_ptrs.resize(dim);
            mat_k_ki_ptrs.resize(dim);
        }
        for (long j = 0; j < num_samples; ++j) {
            Dtype *mat_k_j_ptr = mat_k.col(j).data();
            mat_k_j_ptr[j] = 1.0f;  // mat_k(j, j)
            if (grad_flags[j]) {
                for (long k = 0, kj = grad_flags[j]; k < dim; ++k, kj += n_grad) {
                    mat_k_kj_ptrs[k] = mat_k.col(kj).data();
                }

                for (long k = 0, kj = grad_flags[j]; k < dim; ++k, kj += n_grad) {
                    mat_k_kj_ptrs[k][kj] = l2_inv;  // mat_k(kj, kj) = cov(df_j/dx_k, f_j)
                    mat_k_kj_ptrs[k][j] = 0.0f;     // mat_k(j, kj) = cov(df_j/dx_k, f_j)
                    mat_k_j_ptr[kj] = 0.0f;         // mat_k(kj, j) = cov(df_j/dx_k, f_j)
                    for (long l = k + 1, lj = kj + n_grad; l < dim; ++l, lj += n_grad) {
                        mat_k_kj_ptrs[l][kj] = 0.0f;  // mat_k(kj, lj) = cov(df_j/dx_k, df_j/dx_l)
                        mat_k_kj_ptrs[k][lj] = 0.0f;  // mat_k(lj, kj) = cov(df_j/dx_l, df_j/dx_k)
                    }
                }
            }

            const Dtype *xj_ptr = mat_x.col(j).data();
            for (long i = j + 1; i < num_samples; ++i) {
                const Dtype *xi_ptr = mat_x.col(i).data();
                Dtype *mat_k_i_ptr = mat_k.col(i).data();
                Dtype r = 0;
                for (long k = 0; k < dim; ++k) {
                    Dtype &dx = diff_ij[k];
                    dx = xi_ptr[k] - xj_ptr[k];
                    r += dx * dx;
                }
                Dtype &k_ij = mat_k_j_ptr[i];  // mat_k(i, j)
                k_ij = std::exp(-a * r);       // cov(f_i, f_j)
                mat_k_i_ptr[j] = k_ij;

                if (grad_flags[j]) {
                    // cov(df_j, f_i) = cov(f_i, df_j)
                    for (long k = 0, kj = grad_flags[j]; k < dim; ++k, kj += n_grad) {
                        Dtype &k_i_kj = mat_k_kj_ptrs[k][i];  // mat_k(i, kj)
                        k_i_kj = l2_inv * diff_ij[k] * k_ij;  // cov(f_i, df_j/dx_k)
                        mat_k_i_ptr[kj] = k_i_kj;             // mat_k(kj, i) = cov(df_j/dx_k, f_i)
                    }

                    if (grad_flags[i]) {
                        for (long k = 0, ki = grad_flags[i]; k < dim; ++k, ki += n_grad) {
                            mat_k_ki_ptrs[k] = mat_k.col(ki).data();
                        }

                        for (long k = 0, ki = grad_flags[i], kj = grad_flags[j]; k < dim;
                             ++k, ki += n_grad, kj += n_grad) {
                            Dtype &k_ki_j = mat_k_j_ptr[ki];  // mat_k(ki, j)
                            // cov(df_i, f_j) = -cov(f_i, df_j) = cov(f_j, df_i)
                            k_ki_j = -mat_k_kj_ptrs[k][i];
                            // mat_k(j, ki) = cov(f_j, df_i) = -cov(df_j, f_i) = -cov(f_i, df_j)
                            mat_k_ki_ptrs[k][j] = k_ki_j;  // cov(df_j, df_i) = cov(df_i, df_j)
                            // between Dim-k and Dim-k
                            const Dtype &dxk = diff_ij[k];
                            Dtype &k_kj_ki = mat_k_ki_ptrs[k][kj];  // mat_k(kj, ki)
                            // cov(df_j/dx_k, df_i/dx_k)
                            k_kj_ki = l2_inv * k_ij * (1.0 - l2_inv * dxk * dxk);
                            mat_k_kj_ptrs[k][ki] = k_kj_ki;  // cov(df_i/dx_k, df_j/dx_k)
                            for (long l = k + 1, li = ki + n_grad, lj = kj + n_grad; l < dim;
                                 ++l, li += n_grad, lj += n_grad) {
                                // between Dim-k and Dim-l
                                const Dtype &dxl = diff_ij[l];
                                Dtype &k_kj_li = mat_k_ki_ptrs[l][kj];  // mat_k(kj, li)
                                // cov(df_j/dx_k, df_i/dx_l)
                                // mat_k(lj, ki) = cov(df_j/dx_l, df_i/dx_k)
                                k_kj_li = l2_inv * k_ij * (-l2_inv * dxk * dxl);
                                // mat_k(li, kj) = cov(df_i/dx_l, df_j/dx_k)
                                mat_k_ki_ptrs[k][lj] = k_kj_li;
                                // mat_k(ki, lj) = cov(df_i/dx_k, df_j/dx_l)
                                mat_k_kj_ptrs[k][li] = k_kj_li;
                                mat_k_kj_ptrs[l][ki] = k_kj_li;
                            }
                        }
                    }
                } else if (grad_flags[i]) {
                    // cov(f_j, df_i) = cov(df_i, f_j)
                    for (long k = 0, ki = grad_flags[i]; k < dim; ++k, ki += n_grad) {
                        // mat_k(ki, j)
                        Dtype &k_ki_j = mat_k_j_ptr[ki];
                        k_ki_j = -l2_inv * diff_ij[k] * k_ij;  // cov(f_j, df_i)
                        mat_k(j, ki) = k_ki_j;
                    }
                }
            }  // for (long i = j + 1; i < n; ++i)
        }  // for (long j = 0; j < n; ++j)
        return {n_rows, n_cols};
    }

    template<typename Dtype, int Dim>
    std::pair<long, long>
    RadialBiasFunction<Dtype, Dim>::ComputeKtrainWithGradient(
        const Eigen::Ref<const MatrixX> &mat_x,
        const long num_samples,
        Eigen::VectorXl &vec_grad_flags,
        MatrixX &mat_k) {
        MatrixX mat_alpha;
        return ComputeKtrainWithGradient(mat_x, num_samples, vec_grad_flags, mat_k, mat_alpha);
    }

    template<typename Dtype, int Dim>
    std::pair<long, long>
    RadialBiasFunction<Dtype, Dim>::ComputeKtrainWithGradient(
        const Eigen::Ref<const MatrixX> &mat_x,
        const long num_samples,
        Eigen::VectorXl &vec_grad_flags,
        const Eigen::Ref<const VectorX> &vec_var_x,
        const Eigen::Ref<const VectorX> &vec_var_y,
        const Eigen::Ref<const VectorX> &vec_var_grad,
        MatrixX &mat_k,
        MatrixX & /*mat_alpha*/) {

        long dim;
        if constexpr (Dim == Eigen::Dynamic) {
            dim = mat_x.rows();
        } else {
            dim = Dim;
        }

        ERL_DEBUG_ASSERT(mat_x.rows() == dim, "Each column of mat_x should be {}-D vector.", dim);
        long n_grad = 0;
        long *grad_flags = vec_grad_flags.data();
        for (long i = 0; i < num_samples; ++i) {
            if (long &flag = grad_flags[i]; flag > 0) { flag = num_samples + n_grad++; }
        }
        long n_rows = num_samples + n_grad * dim;
        long n_cols = n_rows;
        ERL_DEBUG_ASSERT(
            mat_k.rows() >= n_rows,
            "mat_k.rows() = {}, it should be >= {}.",
            mat_k.rows(),
            n_rows);
        ERL_DEBUG_ASSERT(
            mat_k.cols() >= n_cols,
            "mat_k.cols() = {}, it should be >= {}.",
            mat_k.cols(),
            n_cols);

        const Dtype l2_inv = 1.0f / (Super::m_setting_->scale * Super::m_setting_->scale);
        const Dtype a = 0.5f * l2_inv;
        // buffer to store the difference between x1_i and x2_j
        Eigen::Vector<Dtype, Dim> diff_ij;          // avoid memory allocation on the heap
        Eigen::Vector<Dtype *, Dim> mat_k_kj_ptrs;  // avoid memory allocation on the heap
        Eigen::Vector<Dtype *, Dim> mat_k_ki_ptrs;  // avoid memory allocation on the heap
        if constexpr (Dim == Eigen::Dynamic) {
            diff_ij.resize(dim);
            mat_k_kj_ptrs.resize(dim);
            mat_k_ki_ptrs.resize(dim);
        }
        for (long j = 0; j < num_samples; ++j) {
            Dtype *mat_k_j_ptr = mat_k.col(j).data();
            mat_k_j_ptr[j] = 1.0f + vec_var_x[j] + vec_var_y[j];  // mat_k(j, j)
            if (grad_flags[j]) {
                for (long k = 0, kj = grad_flags[j]; k < dim; ++k, kj += n_grad) {
                    mat_k_kj_ptrs[k] = mat_k.col(kj).data();
                }

                for (long k = 0, kj = grad_flags[j]; k < dim; ++k, kj += n_grad) {
                    // mat_k(kj, kj) = cov(df_j/dx_k, f_j)
                    mat_k_kj_ptrs[k][kj] = l2_inv + vec_var_grad[j];
                    mat_k_kj_ptrs[k][j] = 0.0f;  // mat_k(j, kj) = cov(df_j/dx_k, f_j)
                    mat_k_j_ptr[kj] = 0.0f;      // mat_k(kj, j) = cov(df_j/dx_k, f_j)
                    for (long l = k + 1, lj = kj + n_grad; l < dim; ++l, lj += n_grad) {
                        mat_k_kj_ptrs[l][kj] = 0.0f;  // mat_k(kj, lj) = cov(df_j/dx_k, df_j/dx_l)
                        mat_k_kj_ptrs[k][lj] = 0.0f;  // mat_k(lj, kj) = cov(df_j/dx_l, df_j/dx_k)
                    }
                }
            }

            const Dtype *xj_ptr = mat_x.col(j).data();
            for (long i = j + 1; i < num_samples; ++i) {
                const Dtype *xi_ptr = mat_x.col(i).data();
                Dtype *mat_k_i_ptr = mat_k.col(i).data();
                Dtype r2 = 0;
                for (long k = 0; k < dim; ++k) {
                    Dtype &dx = diff_ij[k];
                    dx = xi_ptr[k] - xj_ptr[k];
                    r2 += dx * dx;
                }
                Dtype &k_ij = mat_k_j_ptr[i];  // mat_k(i, j)
                k_ij = std::exp(-a * r2);      // cov(f_i, f_j)
                mat_k_i_ptr[j] = k_ij;         // mat_k(j, i)

                if (grad_flags[j]) {
                    // cov(df_j, f_i) = cov(f_i, df_j)
                    for (long k = 0, kj = grad_flags[j]; k < dim; ++k, kj += n_grad) {
                        Dtype &k_i_kj = mat_k_kj_ptrs[k][i];  // mat_k(i, kj)
                        k_i_kj = l2_inv * diff_ij[k] * k_ij;  // cov(f_i, df_j/dx_k)
                        mat_k_i_ptr[kj] = k_i_kj;             // mat_k(kj, i) = cov(df_j/dx_k, f_i)
                    }

                    if (grad_flags[i]) {
                        for (long k = 0, ki = grad_flags[i]; k < dim; ++k, ki += n_grad) {
                            mat_k_ki_ptrs[k] = mat_k.col(ki).data();
                        }

                        for (long k = 0, ki = grad_flags[i], kj = grad_flags[j]; k < dim;
                             ++k, ki += n_grad, kj += n_grad) {
                            Dtype &k_ki_j = mat_k_j_ptr[ki];  // mat_k(ki, j)
                            // cov(df_i, f_j) = -cov(f_i, df_j) = cov(f_j, df_i)
                            k_ki_j = -mat_k_kj_ptrs[k][i];
                            // mat_k(j, ki) = cov(f_j, df_i) = -cov(df_j, f_i) = -cov(f_i, df_j)
                            mat_k_ki_ptrs[k][j] = k_ki_j;  // cov(df_j, df_i) = cov(df_i, df_j)
                            // between Dim-k and Dim-k
                            const Dtype &dxk = diff_ij[k];
                            Dtype &k_kj_ki = mat_k_ki_ptrs[k][kj];  // mat_k(kj, ki)
                            // cov(df_j/dx_k, df_i/dx_k)
                            k_kj_ki = l2_inv * k_ij * (1.0f - l2_inv * dxk * dxk);
                            mat_k_kj_ptrs[k][ki] = k_kj_ki;  // cov(df_i/dx_k, df_j/dx_k)
                            for (long l = k + 1, li = ki + n_grad, lj = kj + n_grad; l < dim;
                                 ++l, li += n_grad, lj += n_grad) {
                                // between Dim-k and Dim-l
                                const Dtype &dxl = diff_ij[l];
                                Dtype &k_kj_li = mat_k_ki_ptrs[l][kj];  // mat_k(kj, li)
                                // cov(df_j/dx_k, df_i/dx_l)
                                k_kj_li = l2_inv * k_ij * (-l2_inv * dxk * dxl);
                                // mat_k(lj, ki) = cov(df_j/dx_l, df_i/dx_k)
                                mat_k_ki_ptrs[k][lj] = k_kj_li;
                                // mat_k(li, kj) = cov(df_i/dx_l, df_j/dx_k)
                                mat_k_kj_ptrs[k][li] = k_kj_li;
                                // mat_k(ki, lj) = cov(df_i/dx_k, df_j/dx_l)
                                mat_k_kj_ptrs[l][ki] = k_kj_li;
                            }
                        }
                    }
                } else if (grad_flags[i]) {
                    // cov(f_j, df_i) = cov(df_i, f_j)
                    for (long k = 0, ki = grad_flags[i]; k < dim; ++k, ki += n_grad) {
                        Dtype &k_ki_j = mat_k_j_ptr[ki];       // mat_k(ki, j)
                        k_ki_j = -l2_inv * diff_ij[k] * k_ij;  // cov(f_j, df_i)
                        mat_k(j, ki) = k_ki_j;
                    }
                }
            }  // for (long i = j + 1; i < n; ++i)
        }  // for (long j = 0; j < n; ++j)
        return {n_rows, n_cols};
    }

    template<typename Dtype, int Dim>
    std::pair<long, long>
    RadialBiasFunction<Dtype, Dim>::ComputeKtrainWithGradient(
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
    RadialBiasFunction<Dtype, Dim>::ComputeKtestWithGradient(
        const Eigen::Ref<const MatrixX> &mat_x1,
        const long num_samples1,
        const Eigen::Ref<const Eigen::VectorXl> &vec_grad1_flags,
        const Eigen::Ref<const MatrixX> &mat_x2,
        const long num_samples2,
        const bool predict_gradient,
        MatrixX &mat_k) const {

        long dim;
        if constexpr (Dim == Eigen::Dynamic) {
            dim = mat_x1.rows();
        } else {
            dim = Dim;
        }

        ERL_DEBUG_ASSERT(mat_x1.rows() == dim, "Each column of mat_x1 should be {}-D vector.", dim);
        ERL_DEBUG_ASSERT(mat_x2.rows() == dim, "Each column of mat_x2 should be {}-D vector.", dim);
        const long n_grad = vec_grad1_flags.head(num_samples1).count();
        const long n_rows = num_samples1 + n_grad * dim;
        const long n_cols = predict_gradient ? num_samples2 * (dim + 1) : num_samples2;
        ERL_DEBUG_ASSERT(
            mat_k.rows() >= n_rows,
            "mat_k.rows() = {}, it should be >= {}.",
            mat_k.rows(),
            n_rows);
        ERL_DEBUG_ASSERT(
            mat_k.cols() >= n_cols,
            "mat_k.cols() = {}, it should be >= {}.",
            mat_k.cols(),
            n_cols);

        const Dtype l2_inv = 1.0 / (Super::m_setting_->scale * Super::m_setting_->scale);
        const Dtype a = 0.5 * l2_inv;
        // buffer to store the difference between x1_i and x2_j
        Eigen::Vector<Dtype, Dim> diff_ij;          // avoid memory allocation on the heap
        Eigen::Vector<Dtype *, Dim> mat_k_kj_ptrs;  // avoid memory allocation on the heap
        if constexpr (Dim == Eigen::Dynamic) {
            diff_ij.resize(dim);
            if (predict_gradient) { mat_k_kj_ptrs.resize(dim); }
        }
        for (long j = 0; j < num_samples2; ++j) {
            const Dtype *x2_j_ptr = mat_x2.col(j).data();
            Dtype *mat_k_j_ptr = mat_k.col(j).data();
            if (predict_gradient) {
                for (long k = 0, kj = j + num_samples2; k < dim; ++k, kj += num_samples2) {
                    mat_k_kj_ptrs[k] = mat_k.col(kj).data();
                }
            }

            for (long i = 0, ki_init = num_samples1; i < num_samples1; ++i) {
                const Dtype *x1_i_ptr = mat_x1.col(i).data();
                Dtype r2 = 0;
                for (long k = 0; k < dim; ++k) {
                    Dtype &dx = diff_ij[k];
                    dx = x1_i_ptr[k] - x2_j_ptr[k];
                    r2 += dx * dx;
                }
                Dtype &k_ij = mat_k_j_ptr[i];  // mat_k(i, j)
                k_ij = std::exp(-a * r2);      // cov(f1_i, f2_j)
                Dtype l2_inv_k_ij = 0.0f;
                if (predict_gradient) {
                    l2_inv_k_ij = l2_inv * k_ij;
                    for (long k = 0, kj = j + num_samples2; k < dim;
                         ++k, kj += num_samples2) {  // cov(f1_i, df2_j)
                        mat_k_kj_ptrs[k][i] = l2_inv_k_ij * diff_ij[k];
                    }
                }

                if (!vec_grad1_flags[i]) { continue; }
                if (predict_gradient) {
                    for (long k = 0, ki = ki_init, kj = j + num_samples2; k < dim;
                         ++k, ki += n_grad, kj += num_samples2) {
                        mat_k_j_ptr[ki] = -mat_k_kj_ptrs[k][i];

                        // between Dim-k and Dim-k
                        const Dtype &dxk = diff_ij[k];
                        // mat_k(ki, kj) = cov(df1_i/dx_k, df2_j/dx_k)
                        mat_k_kj_ptrs[k][ki] = l2_inv_k_ij * (1.0f - l2_inv * dxk * dxk);

                        for (long l = k + 1, li = ki + n_grad, lj = kj + num_samples2; l < dim;
                             ++l, li += n_grad, lj += num_samples2) {
                            // between Dim-k and Dim-l
                            const Dtype &dxl = diff_ij[l];
                            Dtype &k_ki_lj = mat_k_kj_ptrs[l][ki];
                            // mat_k(ki, lj) = cov(df1_i/dx_k, df2_j/dx_l)
                            k_ki_lj = l2_inv_k_ij * (-l2_inv * dxk * dxl);
                            // mat_k(li, kj) = cov(df1_i/dx_l, df2_j/dx_k)
                            mat_k_kj_ptrs[k][li] = k_ki_lj;
                        }
                    }
                } else {
                    for (long k = 0, ki = ki_init; k < dim; ++k, ki += n_grad) {
                        mat_k_j_ptr[ki] = -l2_inv_k_ij * diff_ij[k];
                    }
                }
                ++ki_init;
            }
        }
        return {n_rows, n_cols};
    }

    template<typename Dtype, int Dim>
    std::pair<long, long>
    RadialBiasFunction<Dtype, Dim>::ComputeKtestWithGradientSparse(
        const Eigen::Ref<const MatrixX> &mat_x1,
        const long num_samples1,
        const Eigen::Ref<const Eigen::VectorXl> &vec_grad1_flags,
        const Eigen::Ref<const MatrixX> &mat_x2,
        const long num_samples2,
        const bool predict_gradient,
        const Dtype zero_threshold,
        SparseMatrix &mat_k) const {

        long dim;
        if constexpr (Dim == Eigen::Dynamic) {
            dim = mat_x1.rows();
        } else {
            dim = Dim;
        }

        ERL_DEBUG_ASSERT(mat_x1.rows() == dim, "Each column of mat_x1 should be {}-D vector.", dim);
        ERL_DEBUG_ASSERT(mat_x2.rows() == dim, "Each column of mat_x2 should be {}-D vector.", dim);
        const long n_grad = vec_grad1_flags.head(num_samples1).count();
        const long n_rows = num_samples1 + n_grad * dim;
        const long n_cols = predict_gradient ? num_samples2 * (dim + 1) : num_samples2;
        ERL_DEBUG_ASSERT(
            mat_k.rows() >= n_rows,
            "mat_k.rows() = {}, it should be >= {}.",
            mat_k.rows(),
            n_rows);
        ERL_DEBUG_ASSERT(
            mat_k.cols() >= n_cols,
            "mat_k.cols() = {}, it should be >= {}.",
            mat_k.cols(),
            n_cols);

        const Dtype l2_inv = 1.0f / (Super::m_setting_->scale * Super::m_setting_->scale);
        const Dtype a = 0.5f * l2_inv;
        const Dtype threshold = std::log(zero_threshold);

        // buffer to store the difference between x1_i and x2_j
        Eigen::Vector<Dtype, Dim> diff_ij;  // avoid memory allocation on the heap
        mat_k.setZero();
        ERL_DEBUG_ASSERT(
            mat_k.rows() >= n_rows,
            "mat_k.rows() = {}, it should be >= {}.",
            mat_k.rows(),
            n_rows);
        ERL_DEBUG_ASSERT(
            mat_k.cols() >= n_cols,
            "mat_k.cols() = {}, it should be >= {}.",
            mat_k.cols(),
            n_cols);
        for (long j = 0; j < num_samples2; ++j) {
            const Dtype *x2_j_ptr = mat_x2.col(j).data();
            for (long i = 0, ki_init = num_samples1; i < num_samples1; ++i) {
                const Dtype *x1_i_ptr = mat_x1.col(i).data();
                Dtype r2 = 0;
                for (long k = 0; k < dim; ++k) {
                    Dtype &dx = diff_ij[k];
                    dx = x1_i_ptr[k] - x2_j_ptr[k];
                    r2 += dx * dx;
                }
                r2 *= -a;
                if (r2 < threshold) { continue; }  // skip if r2 is too small

                Dtype k_ij;
                k_ij = std::exp(r2);
                mat_k.insert(i, j) = k_ij;  // mat_k(i, j) = cov(f1_i, f2_j)

                Dtype l2_inv_k_ij = 0.0f;
                if (predict_gradient) {
                    l2_inv_k_ij = l2_inv * k_ij;
                    for (long k = 0, kj = j + num_samples2; k < dim; ++k, kj += num_samples2) {
                        // cov(f1_i, df2_j/dx_k)
                        mat_k.insert(i, kj) = l2_inv_k_ij * diff_ij[k];
                    }
                }

                if (!vec_grad1_flags[i]) { continue; }
                if (predict_gradient) {
                    for (long k = 0, ki = ki_init, kj = j + num_samples2; k < dim;
                         ++k, ki += n_grad, kj += num_samples2) {
                        // mat_k(ki, j) = -mat_k(i, kj)
                        // cov(df1_i/dx_k, f2_j) = -cov(f1_i, df2_j/dx_k)
                        mat_k.insert(ki, j) = -mat_k.coeff(i, kj);

                        // between Dim-k and Dim-k
                        const Dtype &dxk = diff_ij[k];
                        // mat_k(ki, kj) = cov(df1_i/dx_k, df2_j/dx_k)
                        mat_k.insert(ki, kj) = l2_inv_k_ij * (1.0f - l2_inv * dxk * dxk);

                        for (long l = k + 1, li = ki + n_grad, lj = kj + num_samples2; l < dim;
                             ++l, li += n_grad, lj += num_samples2) {
                            // between Dim-k and Dim-l
                            const Dtype &dxl = diff_ij[l];
                            const Dtype k_ki_lj = l2_inv_k_ij * (-l2_inv * dxk * dxl);
                            // mat_k(li, lj) = cov(df1_i/dx_k, df2_j/dx_l)
                            mat_k.insert(ki, lj) = k_ki_lj;
                            // mat_k(li, kj) = cov(df1_i/dx_l, df2_j/dx_k)
                            mat_k.insert(li, kj) = k_ki_lj;
                        }
                    }
                } else {
                    for (long k = 0, ki = ki_init; k < dim; ++k, ki += n_grad) {
                        mat_k.insert(ki, j) = -l2_inv_k_ij * diff_ij[k];
                    }
                }
                ++ki_init;
            }
        }
        return {n_rows, n_cols};
    }

    template class RadialBiasFunction<double, 1>;
    template class RadialBiasFunction<double, 2>;
    template class RadialBiasFunction<double, 3>;
    template class RadialBiasFunction<double, Eigen::Dynamic>;

    template class RadialBiasFunction<float, 1>;
    template class RadialBiasFunction<float, 2>;
    template class RadialBiasFunction<float, 3>;
    template class RadialBiasFunction<float, Eigen::Dynamic>;
}  // namespace erl::covariance
