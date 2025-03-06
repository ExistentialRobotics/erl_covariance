#pragma once

namespace erl::covariance {
    template<typename Dtype>
    static Dtype
    InlineRq(const Dtype a, const Dtype b, const Dtype squared_norm) {
        return std::pow(1 + squared_norm * a, -b);
    }

    template<int Dim, typename Dtype>
    std::pair<long, long>
    RationalQuadratic<Dim, Dtype>::ComputeKtrain(const Eigen::Ref<const MatrixX> &mat_x, const long num_samples, MatrixX &k_mat, VectorX & /*vec_alpha*/) const {
        ERL_DEBUG_ASSERT(k_mat.rows() >= num_samples, "k_mat.rows() = {}, it should be >= {}.", k_mat.rows(), num_samples);
        ERL_DEBUG_ASSERT(k_mat.cols() >= num_samples, "k_mat.cols() = {}, it should be >= {}.", k_mat.cols(), num_samples);
        long dim;
        if constexpr (Dim == Eigen::Dynamic) {
            dim = mat_x.rows();
        } else {
            dim = Dim;
        }
        const Dtype alpha = Super::m_setting_->alpha;
        const Dtype scale_mix = Super::m_setting_->scale_mix;
        const Dtype a = 0.5 / (Super::m_setting_->scale * Super::m_setting_->scale * scale_mix);
        const long stride = k_mat.outerStride();
        for (long j = 0; j < num_samples; ++j) {
            Dtype *k_mat_j_ptr = k_mat.col(j).data();   // use raw pointer to improve performance
            const Dtype *xj_ptr = mat_x.col(j).data();  // use raw pointer to improve performance
            k_mat_j_ptr[j] = alpha;                     // k_mat(j, j)
            if (j + 1 >= num_samples) { continue; }
            Dtype *k_ji_ptr = &k_mat(j, j + 1);  // k_mat(j, i)
            for (long i = j + 1; i < num_samples; ++i, k_ji_ptr += stride) {
                const Dtype *xi_ptr = mat_x.col(i).data();
                Dtype r = 0.0;
                for (long k = 0; k < dim; ++k) {
                    const Dtype dx = xi_ptr[k] - xj_ptr[k];
                    r += dx * dx;
                }
                Dtype &k_ij = k_mat_j_ptr[i];
                k_ij = alpha * InlineRq(a, scale_mix, r);  // k_mat(i, j)
                *k_ji_ptr = k_ij;                          // k_mat(j, i) = k_ij;
            }
        }
        return {num_samples, num_samples};
    }

    template<int Dim, typename Dtype>
    std::pair<long, long>
    RationalQuadratic<Dim, Dtype>::ComputeKtrain(
        const Eigen::Ref<const MatrixX> &mat_x,
        const Eigen::Ref<const VectorX> &vec_var_y,
        const long num_samples,
        MatrixX &k_mat,
        VectorX & /*vec_alpha*/) const {
        ERL_DEBUG_ASSERT(k_mat.rows() >= num_samples, "k_mat.rows() = {}, it should be >= {}.", k_mat.rows(), num_samples);
        ERL_DEBUG_ASSERT(k_mat.cols() >= num_samples, "k_mat.cols() = {}, it should be >= {}.", k_mat.cols(), num_samples);
        long dim;
        if constexpr (Dim == Eigen::Dynamic) {
            dim = mat_x.rows();
        } else {
            dim = Dim;
        }
        const Dtype alpha = Super::m_setting_->alpha;
        const Dtype scale_mix = Super::m_setting_->scale_mix;
        const Dtype a = 0.5 / (Super::m_setting_->scale * Super::m_setting_->scale * scale_mix);
        const long stride = k_mat.outerStride();
        for (long j = 0; j < num_samples; ++j) {
            Dtype *k_mat_j_ptr = k_mat.col(j).data();   // use raw pointer to improve performance
            const Dtype *xj_ptr = mat_x.col(j).data();  // use raw pointer to improve performance
            k_mat_j_ptr[j] = alpha + vec_var_y[j];      // k_mat(j, j)
            if (j + 1 >= num_samples) { continue; }
            Dtype *k_ji_ptr = &k_mat(j, j + 1);  // k_mat(j, i)
            for (long i = j + 1; i < num_samples; ++i, k_ji_ptr += stride) {
                const Dtype *xi_ptr = mat_x.col(i).data();
                Dtype r = 0.0;
                for (long k = 0; k < dim; ++k) {
                    const Dtype dx = xi_ptr[k] - xj_ptr[k];
                    r += dx * dx;
                }
                Dtype &k_ij = k_mat_j_ptr[i];  // k_mat(i, j)
                k_ij = alpha * InlineRq(a, scale_mix, r);
                *k_ji_ptr = k_ij;  // k_mat(j, i) = k_ij;
            }
        }
        return {num_samples, num_samples};
    }

    template<int Dim, typename Dtype>
    std::pair<long, long>
    RationalQuadratic<Dim, Dtype>::ComputeKtest(
        const Eigen::Ref<const MatrixX> &mat_x1,
        const long num_samples1,
        const Eigen::Ref<const MatrixX> &mat_x2,
        const long num_samples2,
        MatrixX &k_mat) const {
        ERL_DEBUG_ASSERT(mat_x1.rows() == mat_x2.rows(), "Sample vectors stored in x1 and x2 should have the same dimension.");
        ERL_DEBUG_ASSERT(k_mat.rows() >= num_samples1, "k_mat.rows() = {}, it should be >= {}.", k_mat.rows(), num_samples1);
        ERL_DEBUG_ASSERT(k_mat.cols() >= num_samples2, "k_mat.cols() = {}, it should be >= {}.", k_mat.cols(), num_samples2);
        long dim;
        if constexpr (Dim == Eigen::Dynamic) {
            dim = mat_x1.rows();
        } else {
            dim = Dim;
        }
        const Dtype alpha = Super::m_setting_->alpha;
        const Dtype scale_mix = Super::m_setting_->scale_mix;
        const Dtype a = 0.5 / (Super::m_setting_->scale * Super::m_setting_->scale * scale_mix);
        for (long j = 0; j < num_samples2; ++j) {
            const Dtype *x2_ptr = mat_x2.col(j).data();
            Dtype *col_j_ptr = k_mat.col(j).data();
            for (long i = 0; i < num_samples1; ++i) {
                const Dtype *x1_ptr = mat_x1.col(i).data();
                Dtype r = 0.0;
                for (long k = 0; k < dim; ++k) {
                    const Dtype dx = x1_ptr[k] - x2_ptr[k];
                    r += dx * dx;
                }
                col_j_ptr[i] = alpha * InlineRq(a, scale_mix, r);
            }
        }
        return {num_samples1, num_samples2};
    }

    template<int Dim, typename Dtype>
    std::pair<long, long>
    RationalQuadratic<Dim, Dtype>::ComputeKtrainWithGradient(
        const Eigen::Ref<const MatrixX> &mat_x,
        const long num_samples,
        Eigen::VectorXl &vec_grad_flags,
        MatrixX &k_mat,
        VectorX & /*vec_alpha*/) const {

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
        ERL_DEBUG_ASSERT(k_mat.rows() >= n_rows, "k_mat.rows() = {}, it should be >= {}.", k_mat.rows(), n_rows);
        ERL_DEBUG_ASSERT(k_mat.cols() >= n_cols, "k_mat.cols() = {}, it should be >= {}.", k_mat.cols(), n_cols);

        const Dtype alpha = Super::m_setting_->alpha;
        const Dtype scale_mix = Super::m_setting_->scale_mix;
        const Dtype l2_inv = 1. / (Super::m_setting_->scale * Super::m_setting_->scale);
        const Dtype a = 0.5 * l2_inv / scale_mix;
        // buffer to store the difference between x1_i and x2_j
        Eigen::Vector<Dtype, Dim> diff_ij;          // avoid memory allocation on the heap
        Eigen::Vector<Dtype *, Dim> k_mat_kj_ptrs;  // avoid memory allocation on the heap
        Eigen::Vector<Dtype *, Dim> k_mat_ki_ptrs;  // avoid memory allocation on the heap
        if constexpr (Dim == Eigen::Dynamic) {
            diff_ij.resize(dim);
            k_mat_kj_ptrs.resize(dim);
            k_mat_ki_ptrs.resize(dim);
        }
        for (long j = 0; j < num_samples; ++j) {
            Dtype *k_mat_j_ptr = k_mat.col(j).data();
            k_mat_j_ptr[j] = alpha;  // k_mat(j, j)
            if (grad_flags[j]) {
                for (long k = 0, kj = grad_flags[j]; k < dim; ++k, kj += n_grad) { k_mat_kj_ptrs[k] = k_mat.col(kj).data(); }

                for (long k = 0, kj = grad_flags[j]; k < dim; ++k, kj += n_grad) {
                    k_mat_kj_ptrs[k][kj] = l2_inv;  // k_mat(kj, kj) = cov(df_j/dx_k, f_j)
                    k_mat_kj_ptrs[k][j] = 0.0;      // k_mat(j, kj) = cov(df_j/dx_k, f_j)
                    k_mat_j_ptr[kj] = 0.0;          // k_mat(kj, j) = cov(df_j/dx_k, f_j)
                    for (long l = k + 1, lj = kj + n_grad; l < dim; ++l, lj += n_grad) {
                        k_mat_kj_ptrs[l][kj] = 0.0;  // k_mat(kj, lj) = cov(df_j/dx_k, df_j/dx_l)
                        k_mat_kj_ptrs[k][lj] = 0.0;  // k_mat(lj, kj) = cov(df_j/dx_l, df_j/dx_k)
                    }
                }
            }

            const Dtype *xj_ptr = mat_x.col(j).data();
            for (long i = j + 1; i < num_samples; ++i) {
                const Dtype *xi_ptr = mat_x.col(i).data();
                Dtype *k_mat_i_ptr = k_mat.col(i).data();
                Dtype r = 0;
                for (long k = 0; k < dim; ++k) {
                    Dtype &dx = diff_ij[k];
                    dx = xi_ptr[k] - xj_ptr[k];
                    r += dx * dx;
                }
                Dtype &k_ij = k_mat_j_ptr[i];              // k_mat(i, j)
                k_ij = alpha * InlineRq(a, scale_mix, r);  // cov(f_i, f_j)
                k_mat_i_ptr[j] = k_ij;

                if (const Dtype beta = 1. / (1. + a * r), gamma = beta * beta * l2_inv * (1. + scale_mix) / scale_mix; grad_flags[j]) {
                    // cov(df_j, f_i) = cov(f_i, df_j)
                    for (long k = 0, kj = grad_flags[j]; k < dim; ++k, kj += n_grad) {
                        Dtype &k_i_kj = k_mat_kj_ptrs[k][i];         // k_mat(i, kj)
                        k_i_kj = beta * l2_inv * diff_ij[k] * k_ij;  // cov(f_i, df_j/dx_k)
                        k_mat_i_ptr[kj] = k_i_kj;                    // k_mat(kj, i) = cov(df_j/dx_k, f_i)
                    }

                    if (grad_flags[i]) {
                        for (long k = 0, ki = grad_flags[i]; k < dim; ++k, ki += n_grad) { k_mat_ki_ptrs[k] = k_mat.col(ki).data(); }

                        for (long k = 0, ki = grad_flags[i], kj = grad_flags[j]; k < dim; ++k, ki += n_grad, kj += n_grad) {
                            Dtype &k_ki_j = k_mat_j_ptr[ki];  // k_mat(ki, j), use reference to improve performance
                            k_ki_j = -k_mat_kj_ptrs[k][i];    // cov(df_i, f_j) = -cov(f_i, df_j) = cov(f_j, df_i)
                            k_mat_ki_ptrs[k][j] = k_ki_j;     // k_mat(j, ki) = cov(f_j, df_i) = -cov(df_j, f_i) = -cov(f_i, df_j)

                            // cov(df_j, df_i) = cov(df_i, df_j)
                            // between Dim-k and Dim-k
                            const Dtype &dxk = diff_ij[k];                         // use reference to improve performance
                            Dtype &k_kj_ki = k_mat_ki_ptrs[k][kj];                 // k_mat(kj, ki)
                            k_kj_ki = l2_inv * k_ij * (beta - gamma * dxk * dxk);  // cov(df_j/dx_k, df_i/dx_k)
                            k_mat_kj_ptrs[k][ki] = k_kj_ki;                        // cov(df_i/dx_k, df_j/dx_k)
                            for (long l = k + 1, li = ki + n_grad, lj = kj + n_grad; l < dim; ++l, li += n_grad, lj += n_grad) {
                                // between Dim-k and Dim-l
                                const Dtype &dxl = diff_ij[l];
                                Dtype &k_kj_li = k_mat_ki_ptrs[l][kj];           // k_mat(kj, li)
                                k_kj_li = l2_inv * k_ij * (-gamma * dxk * dxl);  // cov(df_j/dx_k, df_i/dx_l)
                                k_mat_ki_ptrs[k][lj] = k_kj_li;                  // k_mat(lj, ki) = cov(df_j/dx_l, df_i/dx_k)
                                k_mat_kj_ptrs[k][li] = k_kj_li;                  // k_mat(li, kj) = cov(df_i/dx_l, df_j/dx_k)
                                k_mat_kj_ptrs[l][ki] = k_kj_li;                  // k_mat(ki, lj) = cov(df_i/dx_k, df_j/dx_l)
                            }
                        }
                    }
                } else if (grad_flags[i]) {
                    // cov(f_j, df_i) = cov(df_i, f_j)
                    for (long k = 0, ki = grad_flags[i]; k < dim; ++k, ki += n_grad) {
                        Dtype &k_ki_j = k_mat_j_ptr[ki];              // k_mat(ki, j), use reference to improve performance
                        k_ki_j = -beta * l2_inv * diff_ij[k] * k_ij;  // cov(f_j, df_i)
                        k_mat(j, ki) = k_ki_j;
                    }
                }
            }  // for (long i = j + 1; i < n; ++i)
        }  // for (long j = 0; j < n; ++j)
        return {n_rows, n_cols};
    }

    template<int Dim, typename Dtype>
    std::pair<long, long>
    RationalQuadratic<Dim, Dtype>::ComputeKtrainWithGradient(
        const Eigen::Ref<const MatrixX> &mat_x,
        const long num_samples,
        Eigen::VectorXl &vec_grad_flags,
        const Eigen::Ref<const VectorX> &vec_var_x,
        const Eigen::Ref<const VectorX> &vec_var_y,
        const Eigen::Ref<const VectorX> &vec_var_grad,
        MatrixX &k_mat,
        VectorX & /*vec_alpha*/) const {

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
        ERL_DEBUG_ASSERT(k_mat.rows() >= n_rows, "k_mat.rows() = {}, it should be >= {}.", k_mat.rows(), n_rows);
        ERL_DEBUG_ASSERT(k_mat.cols() >= n_cols, "k_mat.cols() = {}, it should be >= {}.", k_mat.cols(), n_cols);

        const Dtype alpha = Super::m_setting_->alpha;
        const Dtype scale_mix = Super::m_setting_->scale_mix;
        const Dtype l2_inv = 1. / (Super::m_setting_->scale * Super::m_setting_->scale);
        const Dtype a = 0.5 * l2_inv / scale_mix;
        // buffer to store the difference between x1_i and x2_j
        Eigen::Vector<Dtype, Dim> diff_ij;          // avoid memory allocation on the heap
        Eigen::Vector<Dtype *, Dim> k_mat_kj_ptrs;  // avoid memory allocation on the heap
        Eigen::Vector<Dtype *, Dim> k_mat_ki_ptrs;  // avoid memory allocation on the heap
        if constexpr (Dim == Eigen::Dynamic) {
            diff_ij.resize(dim);
            k_mat_kj_ptrs.resize(dim);
            k_mat_ki_ptrs.resize(dim);
        }
        for (long j = 0; j < num_samples; ++j) {
            Dtype *k_mat_j_ptr = k_mat.col(j).data();
            k_mat_j_ptr[j] = alpha + vec_var_x[j] + vec_var_y[j];  // k_mat(j, j)
            if (grad_flags[j]) {
                for (long k = 0, kj = grad_flags[j]; k < dim; ++k, kj += n_grad) { k_mat_kj_ptrs[k] = k_mat.col(kj).data(); }

                for (long k = 0, kj = grad_flags[j]; k < dim; ++k, kj += n_grad) {
                    k_mat_kj_ptrs[k][kj] = l2_inv + vec_var_grad[j];  // k_mat(kj, kj) = cov(df_j/dx_k, f_j)
                    k_mat_kj_ptrs[k][j] = 0.;                         // k_mat(j, kj) = cov(df_j/dx_k, f_j)
                    k_mat_j_ptr[kj] = 0.;                             // k_mat(kj, j) = cov(df_j/dx_k, f_j)
                    for (long l = k + 1, lj = kj + n_grad; l < dim; ++l, lj += n_grad) {
                        k_mat_kj_ptrs[l][kj] = 0.;  // k_mat(kj, lj) = cov(df_j/dx_k, df_j/dx_l)
                        k_mat_kj_ptrs[k][lj] = 0.;  // k_mat(lj, kj) = cov(df_j/dx_l, df_j/dx_k)
                    }
                }
            }

            const Dtype *xj_ptr = mat_x.col(j).data();
            for (long i = j + 1; i < num_samples; ++i) {
                const Dtype *xi_ptr = mat_x.col(i).data();
                Dtype *k_mat_i_ptr = k_mat.col(i).data();
                Dtype r2 = 0;
                for (long k = 0; k < dim; ++k) {
                    Dtype &dx = diff_ij[k];
                    dx = xi_ptr[k] - xj_ptr[k];
                    r2 += dx * dx;
                }
                Dtype &k_ij = k_mat_j_ptr[i];               // k_mat(i, j)
                k_ij = alpha * InlineRq(a, scale_mix, r2);  // cov(f_i, f_j)
                k_mat_i_ptr[j] = k_ij;                      // k_mat(j, i)

                if (const Dtype beta = 1. / (1. + a * r2), gamma = beta * beta * l2_inv * (1. + scale_mix) / scale_mix; grad_flags[j]) {
                    // cov(df_j, f_i) = cov(f_i, df_j)
                    for (long k = 0, kj = grad_flags[j]; k < dim; ++k, kj += n_grad) {
                        Dtype &k_i_kj = k_mat_kj_ptrs[k][i];         // k_mat(i, kj)
                        k_i_kj = beta * l2_inv * diff_ij[k] * k_ij;  // cov(f_i, df_j/dx_k)
                        k_mat_i_ptr[kj] = k_i_kj;                    // k_mat(kj, i) = cov(df_j/dx_k, f_i)
                    }

                    if (grad_flags[i]) {
                        for (long k = 0, ki = grad_flags[i]; k < dim; ++k, ki += n_grad) { k_mat_ki_ptrs[k] = k_mat.col(ki).data(); }

                        for (long k = 0, ki = grad_flags[i], kj = grad_flags[j]; k < dim; ++k, ki += n_grad, kj += n_grad) {
                            Dtype &k_ki_j = k_mat_j_ptr[ki];  // k_mat(ki, j), use reference to improve performance
                            k_ki_j = -k_mat_kj_ptrs[k][i];    // cov(df_i, f_j) = -cov(f_i, df_j) = cov(f_j, df_i)
                            k_mat_ki_ptrs[k][j] = k_ki_j;     // k_mat(j, ki) = cov(f_j, df_i) = -cov(df_j, f_i) = -cov(f_i, df_j)

                            // cov(df_j, df_i) = cov(df_i, df_j)
                            // between Dim-k and Dim-k
                            const Dtype &dxk = diff_ij[k];                         // use reference to improve performance
                            Dtype &k_kj_ki = k_mat_ki_ptrs[k][kj];                 // k_mat(kj, ki)
                            k_kj_ki = l2_inv * k_ij * (beta - gamma * dxk * dxk);  // cov(df_j/dx_k, df_i/dx_k)
                            k_mat_kj_ptrs[k][ki] = k_kj_ki;                        // cov(df_i/dx_k, df_j/dx_k)
                            for (long l = k + 1, li = ki + n_grad, lj = kj + n_grad; l < dim; ++l, li += n_grad, lj += n_grad) {
                                // between Dim-k and Dim-l
                                const Dtype &dxl = diff_ij[l];
                                Dtype &k_kj_li = k_mat_ki_ptrs[l][kj];           // k_mat(kj, li)
                                k_kj_li = l2_inv * k_ij * (-gamma * dxk * dxl);  // cov(df_j/dx_k, df_i/dx_l)
                                k_mat_ki_ptrs[k][lj] = k_kj_li;                  // k_mat(lj, ki) = cov(df_j/dx_l, df_i/dx_k)
                                k_mat_kj_ptrs[k][li] = k_kj_li;                  // k_mat(li, kj) = cov(df_i/dx_l, df_j/dx_k)
                                k_mat_kj_ptrs[l][ki] = k_kj_li;                  // k_mat(ki, lj) = cov(df_i/dx_k, df_j/dx_l)
                            }
                        }
                    }
                } else if (grad_flags[i]) {
                    // cov(f_j, df_i) = cov(df_i, f_j)
                    for (long k = 0, ki = grad_flags[i]; k < dim; ++k, ki += n_grad) {
                        Dtype &k_ki_j = k_mat_j_ptr[ki];              // k_mat(ki, j), use reference to improve performance
                        k_ki_j = -beta * l2_inv * k_ij * diff_ij[k];  // cov(f_j, df_i)
                        k_mat(j, ki) = k_ki_j;
                    }
                }
            }  // for (long i = j + 1; i < n; ++i)
        }  // for (long j = 0; j < n; ++j)
        return {n_rows, n_cols};
    }

    template<int Dim, typename Dtype>
    std::pair<long, long>
    RationalQuadratic<Dim, Dtype>::ComputeKtestWithGradient(
        const Eigen::Ref<const MatrixX> &mat_x1,
        const long num_samples1,
        const Eigen::Ref<const Eigen::VectorXl> &vec_grad1_flags,
        const Eigen::Ref<const MatrixX> &mat_x2,
        const long num_samples2,
        const bool predict_gradient,
        MatrixX &k_mat) const {

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
        ERL_DEBUG_ASSERT(k_mat.rows() >= n_rows, "k_mat.rows() = {}, it should be >= {}.", k_mat.rows(), n_rows);
        ERL_DEBUG_ASSERT(k_mat.cols() >= n_cols, "k_mat.cols() = {}, it should be >= {}.", k_mat.cols(), n_cols);

        const Dtype alpha = Super::m_setting_->alpha;
        const Dtype scale_mix = Super::m_setting_->scale_mix;
        const Dtype l2_inv = 1. / (Super::m_setting_->scale * Super::m_setting_->scale);
        const Dtype a = 0.5 * l2_inv / scale_mix;
        // buffer to store the difference between x1_i and x2_j
        Eigen::Vector<Dtype, Dim> diff_ij;          // avoid memory allocation on the heap
        Eigen::Vector<Dtype *, Dim> k_mat_kj_ptrs;  // avoid memory allocation on the heap
        if constexpr (Dim == Eigen::Dynamic) {
            diff_ij.resize(dim);
            if (predict_gradient) { k_mat_kj_ptrs.resize(dim); }
        }
        for (long j = 0; j < num_samples2; ++j) {
            const Dtype *x2_j_ptr = mat_x2.col(j).data();
            Dtype *k_mat_j_ptr = k_mat.col(j).data();
            if (predict_gradient) {
                for (long k = 0, kj = j + num_samples2; k < dim; ++k, kj += num_samples2) { k_mat_kj_ptrs[k] = k_mat.col(kj).data(); }
            }

            for (long i = 0, ki_init = num_samples1; i < num_samples1; ++i) {
                const Dtype *x1_i_ptr = mat_x1.col(i).data();
                Dtype r2 = 0;
                for (long k = 0; k < dim; ++k) {
                    Dtype &dx = diff_ij[k];
                    dx = x1_i_ptr[k] - x2_j_ptr[k];
                    r2 += dx * dx;
                }
                Dtype &k_ij = k_mat_j_ptr[i];               // k_mat(i, j)
                k_ij = alpha * InlineRq(a, scale_mix, r2);  // cov(f1_i, f2_j)
                const Dtype beta = 1. / (1. + a * r2);
                if (predict_gradient) {
                    for (long k = 0, kj = j + num_samples2; k < dim; ++k, kj += num_samples2) {  // cov(f1_i, df2_j)
                        k_mat_kj_ptrs[k][i] = beta * l2_inv * k_ij * diff_ij[k];
                    }
                }

                if (!vec_grad1_flags[i]) { continue; }
                const Dtype gamma = beta * beta * l2_inv * (1. + scale_mix) / scale_mix;
                if (predict_gradient) {
                    for (long k = 0, ki = ki_init, kj = j + num_samples2; k < dim; ++k, ki += n_grad, kj += num_samples2) {
                        k_mat_j_ptr[ki] = -k_mat_kj_ptrs[k][i];

                        // between Dim-k and Dim-k
                        const Dtype &dxk = diff_ij[k];
                        // k_mat(ki, kj) = cov(df1_i/dx_k, df2_j/dx_k)
                        k_mat_kj_ptrs[k][ki] = l2_inv * k_ij * (beta - gamma * dxk * dxk);

                        for (long l = k + 1, li = ki + n_grad, lj = kj + num_samples2; l < dim; ++l, li += n_grad, lj += num_samples2) {
                            // between Dim-k and Dim-l
                            const Dtype &dxl = diff_ij[l];
                            Dtype &k_ki_lj = k_mat_kj_ptrs[l][li];
                            k_ki_lj = l2_inv * k_ij * (-gamma * dxk * dxl);  // cov(df1_i, df2_j)
                            k_mat_kj_ptrs[k][li] = k_ki_lj;
                        }
                    }
                } else {
                    for (long k = 0, ki = ki_init; k < dim; ++k, ki += n_grad) { k_mat_j_ptr[ki] = -beta * l2_inv * k_ij * diff_ij[k]; }
                }
                ++ki_init;
            }
        }
        return {n_rows, n_cols};
    }
}  // namespace erl::covariance
