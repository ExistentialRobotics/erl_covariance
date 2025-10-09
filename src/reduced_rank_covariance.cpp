#include "erl_covariance/reduced_rank_covariance.hpp"

#include "erl_common/serialization.hpp"
#include "erl_common/storage_order.hpp"

namespace erl::covariance {
    template<typename Dtype>
    void
    ReducedRankCovariance<Dtype>::Setting::BuildSpectralDensities(
        const std::function<VectorX(const VectorX &)> &kernel_spectral_density_func) {
        // double-check locking. see https://en.wikipedia.org/wiki/Double-checked_locking
        // double-check locking may not work correctly before C++11 due to reordering of
        // instructions. but it is safe here.
        if (m_is_built_) { return; }  // already built
        {
            std::lock_guard<std::mutex> lock(m_mutex_);  // lock for building spectral densities
            if (m_is_built_) { return; }                 // already built by another thread

            const long x_dim = num_basis.size();
            const long total_size = num_basis.prod();

            ERL_ASSERTM(
                x_dim == boundaries.size(),
                "num_basis size ({}) does not match boundaries size ({})",
                x_dim,
                boundaries.size());
            ERL_ASSERTM((num_basis.array() > 0).all(), "num_basis should be > 0.");
            ERL_ASSERTM((boundaries.array() > 0).all(), "boundaries should be > 0.");

            m_frequencies_.resize(x_dim, total_size);
            const Eigen::VectorXl strides = common::ComputeFStrides<long>(num_basis, 1);
            // e.g. 2D:
            // x: [0, 1, 2, 3, 0, 1, 2, 3, ...]
            // y: [0, 0, 0, 0, 1, 1, 1, 1, ...]
            for (long i = 0; i < x_dim; ++i) {
                const long stride = strides[i];
                const long dim_size = num_basis[i];
                const long n_copies = total_size / dim_size;
                const Dtype f = M_PI / (2.0 * boundaries[i]);
                MatrixX frequencies =
                    VectorX::LinSpaced(dim_size, f, static_cast<Dtype>(dim_size) * f)
                        .transpose()
                        .replicate(stride, n_copies / stride);
#if EIGEN_VERSION_AT_LEAST(3, 4, 0)
                m_frequencies_.row(i) << frequencies.reshaped(total_size, 1).transpose();
#else
                for (long c = 0; c < total_size; ++c) {
                    m_frequencies_(i, c) = frequencies.data()[c];
                }
#endif
            }

            VectorX freq_squared_norm = m_frequencies_.colwise().squaredNorm();
            if (max_num_basis > 0 && max_num_basis < freq_squared_norm.size()) {
                std::vector<long> indices(freq_squared_norm.size());
                std::iota(indices.begin(), indices.end(), 0);
                std::sort(indices.begin(), indices.end(), [&freq_squared_norm](long i, long j) {
                    return freq_squared_norm[i] < freq_squared_norm[j];
                });
                indices.resize(max_num_basis);
#if EIGEN_VERSION_AT_LEAST(3, 4, 0)
                VectorX freq_squared_norm_new = freq_squared_norm(indices);
                MatrixX frequencies_new = m_frequencies_(Eigen::indexing::all, indices);
#else
                VectorX freq_squared_norm_new(indices.size());
                MatrixX frequencies_new(m_frequencies_.rows(), static_cast<long>(indices.size()));
                for (long i = 0; i < static_cast<long>(indices.size()); ++i) {
                    freq_squared_norm_new[i] = freq_squared_norm[indices[i]];
                    frequencies_new.col(i) = m_frequencies_.col(indices[i]);
                }
#endif
                m_frequencies_.swap(frequencies_new);
                freq_squared_norm.swap(freq_squared_norm_new);
            }
            m_spectral_densities_ = kernel_spectral_density_func(freq_squared_norm);
            m_inv_spectral_densities_ = m_spectral_densities_.cwiseInverse();

            m_is_built_ = true;
        }
    }

    template<typename Dtype>
    void
    ReducedRankCovariance<Dtype>::Setting::ResetSpectralDensities() {
        m_is_built_ = false;
    }

    template<typename Dtype>
    const typename ReducedRankCovariance<Dtype>::MatrixX &
    ReducedRankCovariance<Dtype>::Setting::GetFrequencies() const {
        ERL_DEBUG_ASSERT(m_is_built_, "Spectral densities are not built yet");
        return m_frequencies_;
    }

    template<typename Dtype>
    const typename ReducedRankCovariance<Dtype>::VectorX &
    ReducedRankCovariance<Dtype>::Setting::GetSpectralDensities() const {
        ERL_DEBUG_ASSERT(m_is_built_, "Spectral densities are not built yet");
        return m_spectral_densities_;
    }

    template<typename Dtype>
    const typename ReducedRankCovariance<Dtype>::VectorX &
    ReducedRankCovariance<Dtype>::Setting::GetInvSpectralDensities() const {
        ERL_DEBUG_ASSERT(m_is_built_, "Spectral densities are not built yet");
        return m_inv_spectral_densities_;
    }

    template<typename Dtype>
    YAML::Node
    ReducedRankCovariance<Dtype>::Setting::YamlConvertImpl::encode(const Setting &setting) {
        YAML::Node node = Super::Setting::YamlConvertImpl::encode(setting);
        ERL_YAML_SAVE_ATTR(node, setting, max_num_basis);
        ERL_YAML_SAVE_ATTR(node, setting, num_basis);
        ERL_YAML_SAVE_ATTR(node, setting, boundaries);
        ERL_YAML_SAVE_ATTR(node, setting, accumulated);
        return node;
    }

    template<typename Dtype>
    bool
    ReducedRankCovariance<Dtype>::Setting::YamlConvertImpl::decode(
        const YAML::Node &node,
        Setting &setting) {
        if (!Super::Setting::YamlConvertImpl::decode(node, setting)) { return false; }
        ERL_YAML_LOAD_ATTR(node, setting, max_num_basis);
        ERL_YAML_LOAD_ATTR(node, setting, num_basis);
        ERL_YAML_LOAD_ATTR(node, setting, boundaries);
        ERL_YAML_LOAD_ATTR(node, setting, accumulated);
        return true;
    }

    template<typename Dtype>
    ReducedRankCovariance<Dtype>::ReducedRankCovariance(std::shared_ptr<Setting> setting)
        : Super(setting),
          m_setting_(std::move(setting)) {
        ERL_WARN_COND(
            m_setting_->boundaries.size() != m_setting_->x_dim,
            "Boundaries size ({}) does not match x_dim ({})",
            m_setting_->boundaries.size(),
            m_setting_->x_dim);
    }

    template<typename Dtype>
    std::pair<long, long>
    ReducedRankCovariance<Dtype>::GetMinimumKtrainSize(
        const long /*i*/,
        const long /*i1*/,
        const long /*i2*/) const {
        long e = m_setting_->num_basis.prod();
        return {e, e};
    }

    template<typename Dtype>
    std::pair<long, long>
    ReducedRankCovariance<Dtype>::GetMinimumKtestSize(
        const long /*i*/,
        const long /*i1*/,
        const long num_gradient_dimensions,
        const long num_test_queries,
        const bool predict_gradient) const {
        long e = m_setting_->num_basis.prod();
        return {
            e,
            predict_gradient ? num_test_queries * (1 + num_gradient_dimensions) : num_test_queries};
    }

    template<typename Dtype>
    std::pair<long, long>
    ReducedRankCovariance<Dtype>::ComputeKtrain(
        const Eigen::Ref<const MatrixX> &mat_x,
        long num_samples,
        MatrixX &mat_k,
        MatrixX &mat_alpha) {
        long dims = m_setting_->x_dim;
        if (dims <= 0) { dims = mat_x.rows(); }  // if x_dim is not set, use the rows of mat_x
        const long e = m_setting_->GetFrequencies().cols();  // number of frequencies

        ERL_DEBUG_ASSERT(
            mat_k.rows() >= e,
            "mat_k.rows() = {}, it should be >= {}.",
            mat_k.rows(),
            e);
        ERL_DEBUG_ASSERT(
            mat_k.cols() >= e,
            "mat_k.cols() = {}, it should be >= {}.",
            mat_k.cols(),
            e);
        ERL_DEBUG_ASSERT(
            mat_alpha.rows() >= e,
            "mat_alpha.rows() = {}, it should be >= {}.",
            mat_alpha.rows(),
            e);

        const MatrixX &phi = ComputeEigenFunctions(mat_x, dims, num_samples);  // (N, E)
        auto mat_k_block = mat_k.topLeftCorner(e, e);                          // (E, E)
        mat_k_block << phi.transpose() * phi;  // (E, N) * (N, E) = (E, E)

        if (m_alpha_.size() == 0) { m_alpha_ = MatrixX::Zero(e, mat_alpha.cols()); }  // (E, D)
        auto phi_alpha = mat_alpha.topRows(e);                                        // (E, D)
        phi_alpha << phi.transpose() *
                         MatrixX(mat_alpha.topRows(num_samples));  // (E, N) * (N, D) = (E, D)

        if (m_setting_->accumulated) {
            auto acc_mat_k = const_cast<MatrixX &>(m_mat_k_);
            if (acc_mat_k.size() == 0) { acc_mat_k = MatrixX::Constant(e, e, 0.0f); }
            acc_mat_k += mat_k_block;
            mat_k_block << acc_mat_k;

            m_alpha_ += phi_alpha;
            phi_alpha << m_alpha_;
        }
        return {e, e};
    }

    template<typename Dtype>
    std::pair<long, long>
    ReducedRankCovariance<Dtype>::ComputeKtrain(
        const Eigen::Ref<const MatrixX> &mat_x,
        const Eigen::Ref<const VectorX> &vec_var_y,
        long num_samples,
        MatrixX &mat_k,
        MatrixX &mat_alpha) {

        long dims = m_setting_->x_dim;
        if (dims <= 0) { dims = mat_x.rows(); }  // if x_dim is not set, use the rows of mat_x
        const long e = m_setting_->GetFrequencies().cols();

        ERL_DEBUG_ASSERT(
            mat_k.rows() >= e,
            "mat_k.rows() = {}, it should be >= {}.",
            mat_k.rows(),
            e);
        ERL_DEBUG_ASSERT(
            mat_k.cols() >= e,
            "mat_k.cols() = {}, it should be >= {}.",
            mat_k.cols(),
            e);
        ERL_DEBUG_ASSERT(
            mat_alpha.rows() >= e,
            "mat_alpha.rows() = {}, it should be >= {}.",
            mat_alpha.rows(),
            e);

        const MatrixX phi = ComputeEigenFunctions(mat_x, dims, num_samples);    // (num_samples, E)
        const VectorX inv_sigmas = vec_var_y.head(num_samples).cwiseInverse();  // (num_samples, )
        const MatrixX mat_y = mat_alpha.topRows(num_samples);                   // (num_samples, D)
        const VectorX inv_spectral_densities = m_setting_->GetInvSpectralDensities();
        const bool accumulated = m_setting_->accumulated;

        if (m_alpha_.size() == 0) { m_alpha_ = MatrixX::Zero(e, mat_alpha.cols()); }  // (E, D)
        VectorX inv_sigmas_phi_i(num_samples);

        // phi = [phi_1, phi_2, ..., phi_e]
        // inv_sigmas = [1/var_y_1, 1/var_y_2, ..., 1/var_y_N]
        // inv_sigmas_phi = [inv_sigmas .* phi_1, inv_sigmas .* phi_2, ..., inv_sigmas .* phi_N]
        for (long col = 0; col < e; ++col) {
            auto phi_col = phi.col(col);

            for (long d = 0; d < mat_y.cols(); ++d) {
                Dtype &alpha = mat_alpha(col, d);
                alpha = inv_sigmas.cwiseProduct(phi_col).dot(mat_y.col(d));
                if (accumulated) {
                    Dtype &acc_alpha = m_alpha_(col, d);
                    acc_alpha += alpha;
                    alpha = acc_alpha;
                }
            }

            Dtype *mat_k_col = mat_k.col(col).data();
            Dtype *acc_mat_k_col = accumulated ? m_mat_k_.col(col).data() : nullptr;

            Dtype &mat_k_cc = mat_k_col[col];
            mat_k_cc = inv_sigmas_phi_i.dot(phi_col);
            if (acc_mat_k_col != nullptr) {
                Dtype &acc_mat_k_cc = acc_mat_k_col[col];
                acc_mat_k_cc += mat_k_cc;
                mat_k_cc = acc_mat_k_cc;
            }
            mat_k_cc += inv_spectral_densities[col];

            for (long row = col + 1; row < e; ++row) {
                Dtype &mat_k_rc = mat_k_col[row];
                mat_k_rc = inv_sigmas_phi_i.dot(phi.col(row));
                if (acc_mat_k_col != nullptr) {
                    Dtype &acc_mat_k_rc = acc_mat_k_col[row];
                    acc_mat_k_rc += mat_k_rc;
                    mat_k_rc = acc_mat_k_rc;
                }
                mat_k(col, row) = mat_k_rc;
            }
        }

        return {e, e};
    }

    template<typename Dtype>
    std::pair<long, long>
    ReducedRankCovariance<Dtype>::ComputeKtrainWithGradient(
        const Eigen::Ref<const MatrixX> &mat_x,
        const long num_samples,
        Eigen::VectorXl &vec_grad_flags,
        MatrixX &mat_k,
        MatrixX &mat_alpha) {

        long dims = m_setting_->x_dim;
        if (dims <= 0) { dims = mat_x.rows(); }  // if x_dim is not set, use the rows of mat_x
        const long e = m_setting_->GetFrequencies().cols();

        ERL_DEBUG_ASSERT(
            mat_k.rows() >= e,
            "mat_k.rows() = {}, it should be >= {}.",
            mat_k.rows(),
            e);
        ERL_DEBUG_ASSERT(
            mat_k.cols() >= e,
            "mat_k.cols() = {}, it should be >= {}.",
            mat_k.cols(),
            e);
        ERL_DEBUG_ASSERT(
            mat_alpha.rows() >= e,
            "mat_alpha.rows() = {}, it should be >= {}.",
            mat_alpha.rows(),
            e);

        // (m, e)
        MatrixX phi = ComputeEigenFunctionsWithGradient(mat_x, dims, num_samples, vec_grad_flags);
        const MatrixX mat_y = mat_alpha.topRows(phi.rows());  // (m, D)
        const VectorX inv_spectral_densities = m_setting_->GetInvSpectralDensities();
        const bool accumulated = m_setting_->accumulated;

        // (e, D)
        if (accumulated && m_alpha_.size() == 0) { m_alpha_ = MatrixX::Zero(e, mat_alpha.cols()); }
        // K = phi^T * phi, phi = [phi_1, phi_2, ..., phi_e]
        for (long col = 0; col < e; ++col) {
            auto phi_col = phi.col(col);

            for (long d = 0; d < mat_y.cols(); ++d) {
                Dtype &alpha = mat_alpha(col, d);
                alpha = phi_col.dot(mat_y.col(d));
                if (accumulated) {
                    Dtype &acc_alpha = m_alpha_(col, d);
                    acc_alpha += alpha;
                    alpha = acc_alpha;
                }
            }

            Dtype *mat_k_col = mat_k.col(col).data();
            Dtype *acc_mat_k_col = accumulated ? m_mat_k_.col(col).data() : nullptr;

            Dtype &mat_k_cc = mat_k_col[col];
            mat_k_cc = phi_col.squaredNorm();
            if (acc_mat_k_col != nullptr) {
                Dtype &acc_mat_k_cc = acc_mat_k_col[col];
                acc_mat_k_cc += mat_k_cc;
                mat_k_cc = acc_mat_k_cc;
            }
            mat_k_cc += inv_spectral_densities[col];

            for (long row = col + 1; row < e; ++row) {
                Dtype &mat_k_rc = mat_k_col[row];
                mat_k_rc = phi_col.dot(phi.col(row));
                if (acc_mat_k_col != nullptr) {
                    Dtype &acc_mat_k_rc = acc_mat_k_col[row];
                    acc_mat_k_rc += mat_k_rc;
                    mat_k_rc = acc_mat_k_rc;
                }
                mat_k(col, row) = mat_k_rc;
            }
        }
        return {e, e};
    }

    template<typename Dtype>
    std::pair<long, long>
    ReducedRankCovariance<Dtype>::ComputeKtrainWithGradient(
        const Eigen::Ref<const MatrixX> &mat_x,
        const long num_samples,
        Eigen::VectorXl &vec_grad_flags,
        const Eigen::Ref<const VectorX> &vec_var_x,
        const Eigen::Ref<const VectorX> &vec_var_y,
        const Eigen::Ref<const VectorX> &vec_var_grad,
        MatrixX &mat_k,
        MatrixX &mat_alpha) {

        long dims = m_setting_->x_dim;
        if (dims <= 0) { dims = mat_x.rows(); }  // if x_dim is not set, use the rows of mat_x
        const long e = m_setting_->GetFrequencies().cols();

        ERL_DEBUG_ASSERT(
            mat_k.rows() >= e,
            "mat_k.rows() = {}, it should be >= {}.",
            mat_k.rows(),
            e);
        ERL_DEBUG_ASSERT(
            mat_k.cols() >= e,
            "mat_k.cols() = {}, it should be >= {}.",
            mat_k.cols(),
            e);
        ERL_DEBUG_ASSERT(
            mat_alpha.rows() >= e,
            "mat_alpha.rows() = {}, it should be >= {}.",
            mat_alpha.rows(),
            e);

        // (m, e)
        MatrixX phi = ComputeEigenFunctionsWithGradient(mat_x, dims, num_samples, vec_grad_flags);
        const long m = phi.rows();
        const long n_grad = (m - num_samples) / dims;  // m = num_samples + n_grad * dims
        const MatrixX mat_y = mat_alpha.topRows(m);
        const VectorX inv_spectral_densities = m_setting_->GetInvSpectralDensities();
        const bool accumulated = m_setting_->accumulated;

        // (e, D)
        if (accumulated && m_alpha_.size() == 0) { m_alpha_ = MatrixX::Zero(e, mat_alpha.cols()); }

        VectorX inv_sigmas(m);
        VectorX inv_sigmas_phi_i(m);

        for (long i = 0; i < num_samples; ++i) {
            inv_sigmas[i] = 1.0 / (vec_var_x[i] + vec_var_y[i]);
            if (long j = vec_grad_flags[i]; j > 0) {
                for (; j < m; j += n_grad) { inv_sigmas[j] = 1.0 / vec_var_grad[i]; }
            }
        }

        // phi = [phi_1, phi_2, ..., phi_e]
        // inv_sigmas = [1/var_y_1, 1/var_y_2, ..., 1/var_y_N]
        // inv_sigmas_phi = [inv_sigmas .* phi_1, inv_sigmas .* phi_2, ..., inv_sigmas .* phi_N]
        for (long col = 0; col < e; ++col) {
            auto phi_col = phi.col(col);

            for (long d = 0; d < mat_y.cols(); ++d) {
                Dtype &alpha = mat_alpha(col, d);
                alpha = inv_sigmas.cwiseProduct(phi_col).dot(mat_y.col(d));
                if (accumulated) {
                    Dtype &acc_alpha = m_alpha_(col, d);
                    acc_alpha += alpha;
                    alpha = acc_alpha;
                }
            }

            Dtype *mat_k_col = mat_k.col(col).data();
            Dtype *acc_mat_k_col = accumulated ? m_mat_k_.col(col).data() : nullptr;

            Dtype &mat_k_cc = mat_k_col[col];
            mat_k_cc = inv_sigmas_phi_i.dot(phi_col);
            if (acc_mat_k_col != nullptr) {
                Dtype &acc_mat_k_cc = acc_mat_k_col[col];
                acc_mat_k_cc += mat_k_cc;
                mat_k_cc = acc_mat_k_cc;
            }
            mat_k_cc += inv_spectral_densities[col];

            for (long row = col + 1; row < e; ++row) {
                Dtype &mat_k_rc = mat_k_col[row];
                mat_k_rc = inv_sigmas_phi_i.dot(phi.col(row));
                if (acc_mat_k_col != nullptr) {
                    Dtype &acc_mat_k_rc = acc_mat_k_col[row];
                    acc_mat_k_rc += mat_k_rc;
                    mat_k_rc = acc_mat_k_rc;
                }
                mat_k(col, row) = mat_k_rc;
            }
        }

        return {e, e};
    }

    template<typename Dtype>
    std::pair<long, long>
    ReducedRankCovariance<Dtype>::ComputeKtest(
        const Eigen::Ref<const MatrixX> &mat_x1,
        long /*num_samples1*/,
        const Eigen::Ref<const MatrixX> &mat_x2,
        long num_samples2,
        MatrixX &mat_k) const {

        long dims = m_setting_->x_dim;
        if (dims <= 0) { dims = mat_x1.rows(); }  // if x_dim is not set, use the rows of mat_x1
        const long e = m_setting_->GetFrequencies().cols();
        ERL_DEBUG_ASSERT(
            mat_k.rows() >= e,
            "mat_k.rows() = {}, it should be >= {}.",
            mat_k.rows(),
            e);
        ERL_DEBUG_ASSERT(
            mat_k.cols() >= num_samples2,
            "mat_k.cols() = {}, it should be >= {}.",
            mat_k.cols(),
            num_samples2);
        mat_k.topLeftCorner(e, num_samples2) =
            ComputeEigenFunctions(mat_x2, dims, num_samples2).transpose();
        return {e, num_samples2};
    }

    template<typename Dtype>
    std::pair<long, long>
    ReducedRankCovariance<Dtype>::ComputeKtestWithGradient(
        const Eigen::Ref<const MatrixX> &mat_x1,
        long /*num_samples1*/,
        const Eigen::Ref<const Eigen::VectorXl> & /*vec_grad1_flags*/,
        const Eigen::Ref<const MatrixX> &mat_x2,
        const long num_samples2,
        const bool predict_gradient,
        MatrixX &mat_k) const {

        long dims = m_setting_->x_dim;
        if (dims <= 0) { dims = mat_x1.rows(); }  // if x_dim is not set, use the rows of mat_x1
        const long e = m_setting_->GetFrequencies().cols();
        const long m = predict_gradient ? num_samples2 * (dims + 1) : num_samples2;
        ERL_DEBUG_ASSERT(
            mat_k.rows() >= e,
            "mat_k.rows() = {}, it should be >= {}.",
            mat_k.rows(),
            e);
        ERL_DEBUG_ASSERT(
            mat_k.cols() >= m,
            "mat_k.cols() = {}, it should be >= {}.",
            mat_k.cols(),
            m);
        Eigen::VectorXl grad_flags = Eigen::VectorXl::Ones(num_samples2);
        if (predict_gradient) {
            mat_k.topLeftCorner(e, m) =
                ComputeEigenFunctionsWithGradient(mat_x2, dims, num_samples2, grad_flags)
                    .transpose();
        } else {
            mat_k.topLeftCorner(e, m) =
                ComputeEigenFunctions(mat_x2, dims, num_samples2).transpose();
        }
        return {e, m};
    }

    template<typename Dtype>
    void
    ReducedRankCovariance<Dtype>::BuildSpectralDensities() {
        m_setting_->BuildSpectralDensities([this](const VectorX &freq_squared_norm) -> VectorX {
            return ComputeSpectralDensities(freq_squared_norm);
        });
        const long e = m_setting_->num_basis.prod();
        if (m_setting_->accumulated) {
            if (m_mat_k_.size() == 0) { m_mat_k_ = MatrixX::Zero(e, e); }
        }
    }

    template<typename Dtype>
    typename ReducedRankCovariance<Dtype>::MatrixX
    ReducedRankCovariance<Dtype>::ComputeEigenFunctions(
        const Eigen::Ref<const MatrixX> &mat_x,
        const long dims,
        const long num_samples) const {

        const MatrixX &frequencies = m_setting_->GetFrequencies();
        const Dtype *boundaries = m_setting_->boundaries.data();

        ERL_DEBUG_ASSERT(
            frequencies.rows() >= dims,
            "Number of frequencies ({}) is less than the number of dimensions ({})",
            frequencies.rows(),
            dims);

        const long e = frequencies.cols();
        MatrixX eigen_functions(num_samples, e);
        const Dtype alpha = 1.0 / std::sqrt(m_setting_->boundaries.head(dims).prod());
        VectorX coord_origin = m_coord_origin_;
        if (coord_origin.size() == 0) { coord_origin = VectorX::Zero(dims); }

        for (long j = 0; j < e; ++j) {  // number of basis functions
            Dtype *ef = eigen_functions.col(j).data();
            const Dtype *f = frequencies.col(j).data();
            for (long i = 0; i < num_samples; ++i) {  // number of samples
                ef[i] = alpha;
                const Dtype *x = mat_x.col(i).data();
                for (long d = 0; d < dims; ++d) {
                    ef[i] *= std::sin(f[d] * (x[d] - coord_origin[d] + boundaries[d]));
                }
            }
        }
        return eigen_functions;
    }

    template<typename Dtype>
    typename ReducedRankCovariance<Dtype>::MatrixX
    ReducedRankCovariance<Dtype>::ComputeEigenFunctionsWithGradient(
        const Eigen::Ref<const MatrixX> &mat_x,
        const long dims,
        const long num_samples,
        Eigen::VectorXl &vec_grad_flags) const {

        const MatrixX &frequencies = m_setting_->GetFrequencies();
        const Dtype *boundaries = m_setting_->boundaries.data();
        long *grad_flags = vec_grad_flags.data();

        ERL_DEBUG_ASSERT(
            frequencies.rows() >= dims,
            "Number of frequencies ({}) is less than the number of dimensions ({})",
            frequencies.rows(),
            dims);

        const long e = frequencies.cols();
        long n_grad = 0;
        for (long i = 0; i < num_samples; ++i) {
            if (long &flag = grad_flags[i]; flag > 0) { flag = num_samples + n_grad++; }
        }

        const Dtype alpha = 1.0 / std::sqrt(m_setting_->boundaries.head(dims).prod());
        MatrixX eigen_functions(num_samples + n_grad * dims, e);
        VectorX coord_origin = m_coord_origin_;
        if (coord_origin.size() == 0) { coord_origin = VectorX::Zero(dims); }

        for (long j = 0; j < e; ++j) {  // number of basis functions
            Dtype *ef = eigen_functions.col(j).data();
            const Dtype *f = frequencies.col(j).data();
            VectorX y(dims);

            for (long i = 0; i < num_samples; ++i) {  // number of samples
                ef[i] = alpha;
                const Dtype *x = mat_x.col(i).data();
                for (long d = 0; d < dims; ++d) {
                    y[d] = f[d] * (x[d] - coord_origin[d] + boundaries[d]);
                    ef[i] *= std::sin(y[d]);
                }

                if (long k = grad_flags[i]; k > 0) {
                    for (long d = 0; d < dims; ++d, k += n_grad) {
                        ef[k] = f[d] / std::tan(y[d]) * ef[i];
                    }
                }
            }
        }
        return eigen_functions;
    }

    template<typename Dtype>
    typename ReducedRankCovariance<Dtype>::VectorX
    ReducedRankCovariance<Dtype>::GetCoordOrigin() const {
        return m_coord_origin_;
    }

    template<typename Dtype>
    void
    ReducedRankCovariance<Dtype>::SetCoordOrigin(const VectorX &coord_origin) {
        m_coord_origin_ = coord_origin;
    }

    template<typename Dtype>
    const typename ReducedRankCovariance<Dtype>::MatrixX &
    ReducedRankCovariance<Dtype>::GetKtrain() const {
        return m_mat_k_;
    }

    template<typename Dtype>
    typename ReducedRankCovariance<Dtype>::MatrixX &
    ReducedRankCovariance<Dtype>::GetKtrainBuffer() {
        return m_mat_k_;
    }

    template<typename Dtype>
    const typename ReducedRankCovariance<Dtype>::MatrixX &
    ReducedRankCovariance<Dtype>::GetAlpha() const {
        return m_alpha_;
    }

    template<typename Dtype>
    typename ReducedRankCovariance<Dtype>::MatrixX &
    ReducedRankCovariance<Dtype>::GetAlphaBuffer() {
        return m_alpha_;
    }

    template<typename Dtype>
    bool
    ReducedRankCovariance<Dtype>::operator==(const ReducedRankCovariance &other) const {
        if (m_setting_ == nullptr && other.m_setting_ != nullptr) { return false; }
        if (m_setting_ != nullptr &&
            (other.m_setting_ == nullptr || *m_setting_ != *other.m_setting_)) {
            return false;
        }
        using namespace common;
        if (!SafeEigenMatrixEqual(m_coord_origin_, other.m_coord_origin_)) { return false; }
        if (!SafeEigenMatrixEqual(m_mat_k_, other.m_mat_k_)) { return false; }
        if (!SafeEigenMatrixEqual(m_alpha_, other.m_alpha_)) { return false; }
        return true;
    }

    template<typename Dtype>
    bool
    ReducedRankCovariance<Dtype>::operator!=(const ReducedRankCovariance &other) const {
        return !(*this == other);
    }

    template<typename Dtype>
    bool
    ReducedRankCovariance<Dtype>::Write(std::ostream &s) const {
        static const std::vector<std::pair<
            const char *,
            std::function<bool(const ReducedRankCovariance *, std::ostream &)>>>
            token_function_pairs = {
                {
                    "setting",
                    [](const ReducedRankCovariance *self, std::ostream &stream) {
                        return self->m_setting_->Write(stream) && stream.good();
                    },
                },
                {
                    "coord_origin",
                    [](const ReducedRankCovariance *self, std::ostream &stream) {
                        return common::SaveEigenMatrixToBinaryStream(
                                   stream,
                                   self->m_coord_origin_) &&
                               stream.good();
                    },
                },
                {
                    "mat_k",
                    [](const ReducedRankCovariance *self, std::ostream &stream) {
                        return common::SaveEigenMatrixToBinaryStream(stream, self->m_mat_k_) &&
                               stream.good();
                    },
                },
                {
                    "alpha",
                    [](const ReducedRankCovariance *self, std::ostream &stream) {
                        return common::SaveEigenMatrixToBinaryStream(stream, self->m_alpha_) &&
                               stream.good();
                    },
                },
            };
        return common::WriteTokens(s, this, token_function_pairs);
    }

    template<typename Dtype>
    bool
    ReducedRankCovariance<Dtype>::Read(std::istream &s) {
        static const std::vector<
            std::pair<const char *, std::function<bool(ReducedRankCovariance *, std::istream &)>>>
            token_function_pairs = {
                {
                    "setting",
                    [](ReducedRankCovariance *self, std::istream &stream) {
                        return self->m_setting_->Read(stream) && stream.good();
                    },
                },
                {
                    "coord_origin",
                    [](ReducedRankCovariance *self, std::istream &stream) {
                        return common::LoadEigenMatrixFromBinaryStream(
                                   stream,
                                   self->m_coord_origin_) &&
                               stream.good();
                    },
                },
                {
                    "mat_k",
                    [](ReducedRankCovariance *self, std::istream &stream) {
                        return common::LoadEigenMatrixFromBinaryStream(stream, self->m_mat_k_) &&
                               stream.good();
                    },
                },
                {
                    "alpha",
                    [](ReducedRankCovariance *self, std::istream &stream) {
                        return common::LoadEigenMatrixFromBinaryStream(stream, self->m_alpha_) &&
                               stream.good();
                    },
                },
            };
        return common::ReadTokens(s, this, token_function_pairs);
    }

    template class ReducedRankCovariance<double>;
    template class ReducedRankCovariance<float>;
}  // namespace erl::covariance
