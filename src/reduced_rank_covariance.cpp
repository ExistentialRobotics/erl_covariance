#include "erl_covariance/reduced_rank_covariance.hpp"

namespace erl::covariance {

    void
    ReducedRankCovariance::Setting::BuildSpectralDensities(const std::function<Eigen::VectorXd(const Eigen::VectorXd &)> &kernel_spectral_density_func) {
        if (m_is_built_) { return; }  // already built

        {
            std::lock_guard<std::mutex> lock(m_mutex_);  // lock for building spectral densities
            if (m_is_built_) { return; }                 // already built by another thread

            ERL_ASSERTM(num_basis.size() == boundaries.size(), "num_basis size ({}) does not match boundaries size ({})", num_basis.size(), boundaries.size());
            ERL_WARN_COND(num_basis.size() != x_dim, "num_basis size ({}) does not match x_dim ({})", num_basis.size(), x_dim);

            const long x_dim = num_basis.size();
            const long total_size = num_basis.prod();
            m_frequencies_.resize(x_dim, total_size);
            const Eigen::VectorXl strides = common::ComputeFStrides<long>(num_basis, 1);
            // e.g. 2D:
            // x: [0, 1, 2, 3, 0, 1, 2, 3, ...]
            // y: [0, 0, 0, 0, 1, 1, 1, 1, ...]
            for (long i = 0; i < x_dim; ++i) {
                const long stride = strides[i];
                const long dim_size = num_basis[i];
                const long n_copies = total_size / dim_size;
                const double f = M_PI / (2.0 * boundaries[i]);
                Eigen::MatrixXd frequencies = Eigen::VectorXd::LinSpaced(dim_size, f, static_cast<double>(dim_size) * f)  //
                                                  .transpose()
                                                  .replicate(stride, n_copies / stride);
                m_frequencies_.row(i) << frequencies.reshaped(total_size, 1).transpose();
            }

            const Eigen::VectorXd freq_squared_norm = m_frequencies_.colwise().squaredNorm();
            m_spectral_densities_ = kernel_spectral_density_func(freq_squared_norm);
            m_inv_spectral_densities_ = m_spectral_densities_.cwiseInverse();

            m_is_built_ = true;
        }
    }

    std::pair<long, long>
    ReducedRankCovariance::ComputeKtrain(const Eigen::Ref<const Eigen::MatrixXd> &mat_x, long num_samples, Eigen::MatrixXd &mat_k, Eigen::VectorXd &vec_alpha)
        const {
        long dims = m_setting_->x_dim;
        if (dims <= 0) { dims = mat_x.rows(); }  // if x_dim is not set, use the number of rows of mat_x
        const long e = m_setting_->GetFrequencies().cols();

        ERL_DEBUG_ASSERT(mat_k.rows() >= e, "mat_k.rows() = {}, it should be >= {}.", mat_k.rows(), e);
        ERL_DEBUG_ASSERT(mat_k.cols() >= e, "mat_k.cols() = {}, it should be >= {}.", mat_k.cols(), e);

        const Eigen::MatrixXd &phi = ComputeEigenFunctions(mat_x, dims, num_samples);  // (N, E)
        auto mat_k_block = mat_k.topLeftCorner(e, e);
        mat_k_block << phi.transpose() * phi;
        auto vec_alpha_block = vec_alpha.head(e);
        vec_alpha_block << phi.transpose() * Eigen::VectorXd(vec_alpha.head(num_samples));

        if (m_setting_->accumulated) {
            auto acc_mat_k = const_cast<Eigen::MatrixXd &>(m_mat_k_);
            if (acc_mat_k.size() == 0) { acc_mat_k = Eigen::MatrixXd ::Constant(e, e, 0.0); }
            acc_mat_k += mat_k_block;
            mat_k_block << acc_mat_k;

            auto acc_vec_alpha = const_cast<Eigen::VectorXd &>(m_vec_alpha_);
            if (acc_vec_alpha.size() == 0) { acc_vec_alpha = Eigen::VectorXd::Constant(e, 0.0); }
            acc_vec_alpha += vec_alpha_block;
            vec_alpha_block << acc_vec_alpha;
        }
        return {e, e};
    }

    std::pair<long, long>
    ReducedRankCovariance::ComputeKtrain(
        const Eigen::Ref<const Eigen::MatrixXd> &mat_x,
        const Eigen::Ref<const Eigen::VectorXd> &vec_var_y,
        long num_samples,
        Eigen::MatrixXd &mat_k,
        Eigen::VectorXd &vec_alpha) const {

        long dims = m_setting_->x_dim;
        if (dims <= 0) { dims = mat_x.rows(); }  // if x_dim is not set, use the number of rows of mat_x
        const long e = m_setting_->GetFrequencies().cols();

        ERL_DEBUG_ASSERT(mat_k.rows() >= e, "mat_k.rows() = {}, it should be >= {}.", mat_k.rows(), e);
        ERL_DEBUG_ASSERT(mat_k.cols() >= e, "mat_k.cols() = {}, it should be >= {}.", mat_k.cols(), e);

        Eigen::MatrixXd phi = ComputeEigenFunctions(mat_x, dims, num_samples);  // (N, E)
        Eigen::VectorXd vec_inv_sigmas = vec_var_y.head(num_samples).cwiseInverse();
        Eigen::MatrixXd inv_sigmas = vec_inv_sigmas.asDiagonal();
        Eigen::MatrixXd phi_t_inv_sigmas = phi.transpose() * inv_sigmas;

        auto mat_k_block = mat_k.topLeftCorner(e, e);
        mat_k_block << phi_t_inv_sigmas * phi;
        auto vec_alpha_block = vec_alpha.head(e);
        vec_alpha_block << phi_t_inv_sigmas * Eigen::VectorXd(vec_alpha.head(num_samples));

        if (m_setting_->accumulated) {
            auto acc_mat_k = const_cast<Eigen::MatrixXd &>(m_mat_k_);
            if (acc_mat_k.size() == 0) { acc_mat_k = Eigen::MatrixXd ::Constant(e, e, 0.0); }
            acc_mat_k += mat_k_block;
            mat_k_block << acc_mat_k;

            auto acc_vec_alpha = const_cast<Eigen::VectorXd &>(m_vec_alpha_);
            if (acc_vec_alpha.size() == 0) { acc_vec_alpha = Eigen::VectorXd::Constant(e, 0.0); }
            acc_vec_alpha += vec_alpha_block;
            vec_alpha_block << acc_vec_alpha;
        }
        mat_k_block.diagonal() += m_setting_->GetInvSpectralDensities();
        return {e, e};
    }

    std::pair<long, long>
    ReducedRankCovariance::ComputeKtrainWithGradient(
        const Eigen::Ref<const Eigen::MatrixXd> &mat_x,
        long num_samples,
        Eigen::VectorXl &vec_grad_flags,
        Eigen::MatrixXd &mat_k,
        Eigen::VectorXd &vec_alpha) const {

        long dims = m_setting_->x_dim;
        if (dims <= 0) { dims = mat_x.rows(); }  // if x_dim is not set, use the number of rows of mat_x
        const long e = m_setting_->GetFrequencies().cols();

        ERL_DEBUG_ASSERT(mat_k.rows() >= e, "mat_k.rows() = {}, it should be >= {}.", mat_k.rows(), e);
        ERL_DEBUG_ASSERT(mat_k.cols() >= e, "mat_k.cols() = {}, it should be >= {}.", mat_k.cols(), e);

        Eigen::MatrixXd phi = ComputeEigenFunctionsWithGradient(mat_x, dims, num_samples, vec_grad_flags);  // (N, E)
        auto mat_k_block = mat_k.topLeftCorner(e, e);
        mat_k_block << phi.transpose() * phi;
        auto vec_alpha_block = vec_alpha.head(e);
        vec_alpha_block << phi.transpose() * Eigen::VectorXd(vec_alpha.head(phi.rows()));

        if (m_setting_->accumulated) {
            auto acc_mat_k = const_cast<Eigen::MatrixXd &>(m_mat_k_);
            if (acc_mat_k.size() == 0) { acc_mat_k = Eigen::MatrixXd ::Constant(e, e, 0.0); }
            acc_mat_k += mat_k_block;
            mat_k_block << acc_mat_k;

            auto acc_vec_alpha = const_cast<Eigen::VectorXd &>(m_vec_alpha_);
            if (acc_vec_alpha.size() == 0) { acc_vec_alpha = Eigen::VectorXd::Constant(e, 0.0); }
            acc_vec_alpha += vec_alpha_block;
            vec_alpha_block << acc_vec_alpha;
        }
        return {e, e};
    }

    std::pair<long, long>
    ReducedRankCovariance::ComputeKtrainWithGradient(
        const Eigen::Ref<const Eigen::MatrixXd> &mat_x,
        long num_samples,
        Eigen::VectorXl &vec_grad_flags,
        const Eigen::Ref<const Eigen::VectorXd> &vec_var_x,
        const Eigen::Ref<const Eigen::VectorXd> &vec_var_y,
        const Eigen::Ref<const Eigen::VectorXd> &vec_var_grad,
        Eigen::MatrixXd &mat_k,
        Eigen::VectorXd &vec_alpha) const {

        long dims = m_setting_->x_dim;
        if (dims <= 0) { dims = mat_x.rows(); }  // if x_dim is not set, use the number of rows of mat_x
        const long e = m_setting_->GetFrequencies().cols();

        ERL_DEBUG_ASSERT(mat_k.rows() >= e, "mat_k.rows() = {}, it should be >= {}.", mat_k.rows(), e);
        ERL_DEBUG_ASSERT(mat_k.cols() >= e, "mat_k.cols() = {}, it should be >= {}.", mat_k.cols(), e);

        Eigen::MatrixXd phi = ComputeEigenFunctionsWithGradient(mat_x, dims, num_samples, vec_grad_flags);  // (N, E)
        Eigen::VectorXd sigmas(phi.rows());
        for (long i = 0; i < num_samples; ++i) {
            sigmas[i] = vec_var_x[i] + vec_var_y[i];
            if (long &flag = vec_grad_flags[i]; flag > 0) { sigmas[flag] = vec_var_grad[i]; }
        }
        Eigen::MatrixXd phi_t_inv_sigmas = phi.transpose();
        phi_t_inv_sigmas.diagonal() *= sigmas.cwiseInverse();

        auto mat_k_block = mat_k.topLeftCorner(e, e);
        mat_k_block << phi_t_inv_sigmas * phi;
        auto vec_alpha_block = vec_alpha.head(e);
        vec_alpha_block << phi_t_inv_sigmas * Eigen::VectorXd(vec_alpha.head(num_samples));

        if (m_setting_->accumulated) {
            auto acc_mat_k = const_cast<Eigen::MatrixXd &>(m_mat_k_);
            if (acc_mat_k.size() == 0) { acc_mat_k = Eigen::MatrixXd ::Constant(e, e, 0.0); }
            acc_mat_k += mat_k_block;
            mat_k_block << acc_mat_k;

            auto acc_vec_alpha = const_cast<Eigen::VectorXd &>(m_vec_alpha_);
            if (acc_vec_alpha.size() == 0) { acc_vec_alpha = Eigen::VectorXd::Constant(e, 0.0); }
            acc_vec_alpha += vec_alpha_block;
            vec_alpha_block << acc_vec_alpha;
        }
        mat_k_block.diagonal() += m_setting_->GetInvSpectralDensities();
        return {e, e};
    }

    std::pair<long, long>
    ReducedRankCovariance::ComputeKtest(
        const Eigen::Ref<const Eigen::MatrixXd> &mat_x1,
        long /*num_samples1*/,
        const Eigen::Ref<const Eigen::MatrixXd> &mat_x2,
        long num_samples2,
        Eigen::MatrixXd &mat_k) const {

        long dims = m_setting_->x_dim;
        if (dims <= 0) { dims = mat_x1.rows(); }  // if x_dim is not set, use the number of rows of mat_x1
        const long e = m_setting_->GetFrequencies().cols();
        ERL_DEBUG_ASSERT(mat_k.rows() >= e, "mat_k.rows() = {}, it should be >= {}.", mat_k.rows(), e);
        ERL_DEBUG_ASSERT(mat_k.cols() >= num_samples2, "mat_k.cols() = {}, it should be >= {}.", mat_k.cols(), num_samples2);
        mat_k.topLeftCorner(e, num_samples2) = ComputeEigenFunctions(mat_x2, dims, num_samples2).transpose();
        return {e, num_samples2};
    }

    std::pair<long, long>
    ReducedRankCovariance::ComputeKtestWithGradient(
        const Eigen::Ref<const Eigen::MatrixXd> &mat_x1,
        long /*num_samples1*/,
        const Eigen::Ref<const Eigen::VectorXl> & /*vec_grad1_flags*/,
        const Eigen::Ref<const Eigen::MatrixXd> &mat_x2,
        long num_samples2,
        Eigen::MatrixXd &mat_k) const {

        long dims = m_setting_->x_dim;
        if (dims <= 0) { dims = mat_x1.rows(); }  // if x_dim is not set, use the number of rows of mat_x1
        const long e = m_setting_->GetFrequencies().cols();
        const long m = num_samples2 * (dims + 1);
        ERL_DEBUG_ASSERT(mat_k.rows() >= e, "mat_k.rows() = {}, it should be >= {}.", mat_k.rows(), e);
        ERL_DEBUG_ASSERT(mat_k.cols() >= m, "mat_k.cols() = {}, it should be >= {}.", mat_k.cols(), m);
        Eigen::VectorXl grad_flags = Eigen::VectorXl::Ones(num_samples2);
        mat_k.topLeftCorner(e, m) = ComputeEigenFunctionsWithGradient(mat_x2, dims, num_samples2, grad_flags).transpose();
        return {e, m};
    }

    Eigen::MatrixXd
    ReducedRankCovariance::ComputeEigenFunctions(const Eigen::Ref<const Eigen::MatrixXd> &mat_x, const long dims, const long num_samples) const {

        const Eigen::MatrixXd &frequencies = m_setting_->GetFrequencies();
        const double *boundaries = m_setting_->boundaries.data();

        ERL_DEBUG_ASSERT(frequencies.rows() >= dims, "Number of frequencies ({}) is less than the number of dimensions ({})", frequencies.rows(), dims);

        const long e = frequencies.cols();
        Eigen::MatrixXd eigen_functions(num_samples, e);
        const double alpha = 1.0 / std::sqrt(m_setting_->boundaries.head(dims).prod());
        Eigen::VectorXd coord_origin = m_coord_origin_;
        if (coord_origin.size() == 0) { coord_origin = Eigen::VectorXd::Zero(dims); }

        for (long j = 0; j < e; ++j) {  // number of basis functions
            double *ef = eigen_functions.col(j).data();
            const double *f = frequencies.col(j).data();
            for (long i = 0; i < num_samples; ++i) {  // number of samples
                ef[i] = alpha;
                const double *x = mat_x.col(i).data();
                for (long d = 0; d < dims; ++d) { ef[i] *= std::sin(f[d] * (x[d] - coord_origin[d] + boundaries[d])); }
            }
        }
        return eigen_functions;
    }

    Eigen::MatrixXd
    ReducedRankCovariance::ComputeEigenFunctionsWithGradient(
        const Eigen::Ref<const Eigen::MatrixXd> &mat_x,
        const long dims,
        const long num_samples,
        Eigen::VectorXl &vec_grad_flags) const {

        const Eigen::MatrixXd &frequencies = m_setting_->GetFrequencies();
        const double *boundaries = m_setting_->boundaries.data();
        long *grad_flags = vec_grad_flags.data();

        ERL_DEBUG_ASSERT(frequencies.rows() >= dims, "Number of frequencies ({}) is less than the number of dimensions ({})", frequencies.rows(), dims);

        const long e = frequencies.cols();
        long n_grad = 0;
        for (long i = 0; i < num_samples; ++i) {
            if (long &flag = grad_flags[i]; flag > 0) { flag = num_samples + n_grad++; }
        }

        const double alpha = 1.0 / std::sqrt(m_setting_->boundaries.head(dims).prod());
        Eigen::MatrixXd eigen_functions(num_samples + n_grad * dims, e);
        Eigen::VectorXd coord_origin = m_coord_origin_;
        if (coord_origin.size() == 0) { coord_origin = Eigen::VectorXd::Zero(dims); }

        for (long j = 0; j < e; ++j) {  // number of basis functions
            double *ef = eigen_functions.col(j).data();
            const double *f = frequencies.col(j).data();
            Eigen::VectorXd y(dims);

            for (long i = 0; i < num_samples; ++i) {  // number of samples
                ef[i] = alpha;
                const double *x = mat_x.col(i).data();
                for (long d = 0; d < dims; ++d) {
                    y[d] = f[d] * (x[d] - coord_origin[d] + boundaries[d]);
                    ef[i] *= std::sin(y[d]);
                }

                if (long k = grad_flags[i]; k > 0) {
                    for (long d = 0; d < dims; ++d, k += n_grad) { ef[k] = f[d] / std::tan(y[d]) * ef[i]; }
                }
            }
        }
        return eigen_functions;
    }

}  // namespace erl::covariance
