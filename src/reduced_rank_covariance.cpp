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

            Eigen::VectorXd freq_squared_norm = m_frequencies_.colwise().squaredNorm();
            if (max_num_basis > 0 && max_num_basis < freq_squared_norm.size()) {
                std::vector<long> indices(freq_squared_norm.size());
                std::iota(indices.begin(), indices.end(), 0);
                std::sort(indices.begin(), indices.end(), [&freq_squared_norm](long i, long j) { return freq_squared_norm[i] < freq_squared_norm[j]; });
                indices.resize(max_num_basis);
                Eigen::VectorXd freq_squared_norm_new = freq_squared_norm(indices);
                Eigen::MatrixXd freqeuencies_new = m_frequencies_(Eigen::indexing::all, indices);
                m_frequencies_.swap(freqeuencies_new);
                freq_squared_norm.swap(freq_squared_norm_new);
            }
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

        const Eigen::MatrixXd phi = ComputeEigenFunctions(mat_x, dims, num_samples);  // (num_samples, e)
        const Eigen::VectorXd inv_sigmas = vec_var_y.head(num_samples).cwiseInverse();
        const Eigen::VectorXd y = vec_alpha.head(num_samples);
        const Eigen::VectorXd inv_spectral_densities = m_setting_->GetInvSpectralDensities();
        const bool accumulated = m_setting_->accumulated;

        Eigen::VectorXd inv_sigmas_phi_i(num_samples);
        auto acc_mat_k = const_cast<Eigen::MatrixXd &>(m_mat_k_);
        auto acc_vec_alpha = const_cast<Eigen::VectorXd &>(m_vec_alpha_);

        // phi = [phi_1, phi_2, ..., phi_e]
        // inv_sigmas = [1/var_y_1, 1/var_y_2, ..., 1/var_y_N]
        // inv_sigmas_phi = [inv_sigmas .* phi_1, inv_sigmas .* phi_2, ..., inv_sigmas .* phi_N]
        double *acc_alpha = accumulated ? acc_vec_alpha.data() : nullptr;
        for (long i = 0; i < e; ++i) {
            const double *phi_i = phi.col(i).data();
            double &alpha_i = vec_alpha[i];
            alpha_i = 0;
            for (long j = 0; j < num_samples; ++j) {
                double &inv_sigmas_phi_ij = inv_sigmas_phi_i[j];
                inv_sigmas_phi_ij = inv_sigmas[j] * phi_i[j];
                alpha_i += inv_sigmas_phi_ij * y[j];
            }
            if (acc_alpha != nullptr) {
                double &acc_alpha_i = acc_alpha[i];
                acc_alpha_i += alpha_i;
                alpha_i = acc_alpha_i;
            }

            double *mat_k_i = mat_k.col(i).data();
            double *acc_mat_k_i = accumulated ? acc_mat_k.col(i).data() : nullptr;
            double &mat_k_ii = mat_k_i[i];
            mat_k_ii = inv_sigmas_phi_i.dot(phi.col(i));
            if (acc_mat_k_i != nullptr) {
                double &acc_mat_k_ii = acc_mat_k_i[i];
                acc_mat_k_ii += mat_k_ii;
                mat_k_ii = acc_mat_k_ii;
            }
            mat_k_ii += inv_spectral_densities[i];

            for (long j = i + 1; j < e; ++j) {
                double &mat_k_ij = mat_k_i[j];
                mat_k_ij = inv_sigmas_phi_i.dot(phi.col(j));
                if (acc_mat_k_i != nullptr) {
                    double &acc_mat_k_ij = acc_mat_k_i[j];
                    acc_mat_k_ij += mat_k_ij;
                    mat_k_ij = acc_mat_k_ij;
                }
                mat_k(j, i) = mat_k_ij;
            }
        }

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

        const Eigen::MatrixXd phi = ComputeEigenFunctionsWithGradient(mat_x, dims, num_samples, vec_grad_flags);  // (m, e)
        const long m = phi.rows();
        const Eigen::VectorXd y = vec_alpha.head(m);
        const Eigen::VectorXd inv_spectral_densities = m_setting_->GetInvSpectralDensities();
        const bool accumulated = m_setting_->accumulated;

        auto acc_mat_k = const_cast<Eigen::MatrixXd &>(m_mat_k_);
        auto acc_vec_alpha = const_cast<Eigen::VectorXd &>(m_vec_alpha_);

        // phi = [phi_1, phi_2, ..., phi_e]
        // K = phi^T * phi
        double *acc_alpha = accumulated ? acc_vec_alpha.data() : nullptr;
        for (long i = 0; i < e; ++i) {
            const double *phi_i = phi.col(i).data();
            double &alpha_i = vec_alpha[i];
            alpha_i = 0;
            for (long j = 0; j < m; ++j) { alpha_i += phi_i[j] * y[j]; }
            if (acc_alpha != nullptr) {
                double &acc_alpha_i = acc_alpha[i];
                acc_alpha_i += alpha_i;
                alpha_i = acc_alpha_i;
            }

            double *mat_k_i = mat_k.col(i).data();
            double *acc_mat_k_i = accumulated ? acc_mat_k.col(i).data() : nullptr;
            double &mat_k_ii = mat_k_i[i];
            mat_k_ii = phi.col(i).squaredNorm();
            if (acc_mat_k_i != nullptr) {
                double &acc_mat_k_ii = acc_mat_k_i[i];
                acc_mat_k_ii += mat_k_ii;
                mat_k_ii = acc_mat_k_ii;
            }
            mat_k_ii += inv_spectral_densities[i];

            for (long j = i + 1; j < e; ++j) {
                double &mat_k_ij = mat_k_i[j];
                mat_k_ij = phi.col(i).dot(phi.col(j));
                if (acc_mat_k_i != nullptr) {
                    double &acc_mat_k_ij = acc_mat_k_i[j];
                    acc_mat_k_ij += mat_k_ij;
                    mat_k_ij = acc_mat_k_ij;
                }
                mat_k(j, i) = mat_k_ij;
            }
        }

        // long dims = m_setting_->x_dim;
        // if (dims <= 0) { dims = mat_x.rows(); }  // if x_dim is not set, use the number of rows of mat_x
        // const long e = m_setting_->GetFrequencies().cols();
        //
        // ERL_DEBUG_ASSERT(mat_k.rows() >= e, "mat_k.rows() = {}, it should be >= {}.", mat_k.rows(), e);
        // ERL_DEBUG_ASSERT(mat_k.cols() >= e, "mat_k.cols() = {}, it should be >= {}.", mat_k.cols(), e);
        //
        // Eigen::MatrixXd phi = ComputeEigenFunctionsWithGradient(mat_x, dims, num_samples, vec_grad_flags);  // (N, E)
        // auto mat_k_block = mat_k.topLeftCorner(e, e);
        // mat_k_block << phi.transpose() * phi;
        // auto vec_alpha_block = vec_alpha.head(e);
        // vec_alpha_block << phi.transpose() * Eigen::VectorXd(vec_alpha.head(phi.rows()));
        //
        // if (m_setting_->accumulated) {
        //     auto acc_mat_k = const_cast<Eigen::MatrixXd &>(m_mat_k_);
        //     if (acc_mat_k.size() == 0) { acc_mat_k = Eigen::MatrixXd ::Constant(e, e, 0.0); }
        //     acc_mat_k += mat_k_block;
        //     mat_k_block << acc_mat_k;
        //
        //     auto acc_vec_alpha = const_cast<Eigen::VectorXd &>(m_vec_alpha_);
        //     if (acc_vec_alpha.size() == 0) { acc_vec_alpha = Eigen::VectorXd::Constant(e, 0.0); }
        //     acc_vec_alpha += vec_alpha_block;
        //     vec_alpha_block << acc_vec_alpha;
        // }
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

        const Eigen::MatrixXd phi = ComputeEigenFunctionsWithGradient(mat_x, dims, num_samples, vec_grad_flags);  // (m, e)
        const long m = phi.rows();
        Eigen::VectorXd inv_sigmas(m);
        const Eigen::VectorXd y = vec_alpha.head(m);
        const Eigen::VectorXd inv_spectral_densities = m_setting_->GetInvSpectralDensities();
        const bool accumulated = m_setting_->accumulated;

        Eigen::VectorXd inv_sigmas_phi_i(m);
        auto acc_mat_k = const_cast<Eigen::MatrixXd &>(m_mat_k_);
        auto acc_vec_alpha = const_cast<Eigen::VectorXd &>(m_vec_alpha_);

        for (long i = 0; i < num_samples; ++i) {
            inv_sigmas[i] = 1.0 / (vec_var_x[i] + vec_var_y[i]);
            if (long &flag = vec_grad_flags[i]; flag > 0) { inv_sigmas[flag] = 1.0 / vec_var_grad[i]; }
        }

        // phi = [phi_1, phi_2, ..., phi_e]
        // inv_sigmas = [1/var_y_1, 1/var_y_2, ..., 1/var_y_N]
        // inv_sigmas_phi = [inv_sigmas .* phi_1, inv_sigmas .* phi_2, ..., inv_sigmas .* phi_N]
        double *acc_alpha = accumulated ? acc_vec_alpha.data() : nullptr;
        for (long i = 0; i < e; ++i) {
            const double *phi_i = phi.col(i).data();
            double &alpha_i = vec_alpha[i];
            alpha_i = 0;
            for (long j = 0; j < m; ++j) {
                double &inv_sigmas_phi_ij = inv_sigmas_phi_i[j];
                inv_sigmas_phi_ij = inv_sigmas[j] * phi_i[j];
                alpha_i += inv_sigmas_phi_ij * y[j];
            }
            if (acc_alpha != nullptr) {
                double &acc_alpha_i = acc_alpha[i];
                acc_alpha_i += alpha_i;
                alpha_i = acc_alpha_i;
            }

            double *mat_k_i = mat_k.col(i).data();
            double *acc_mat_k_i = accumulated ? acc_mat_k.col(i).data() : nullptr;
            double &mat_k_ii = mat_k_i[i];
            mat_k_ii = inv_sigmas_phi_i.dot(phi.col(i));
            if (acc_mat_k_i != nullptr) {
                double &acc_mat_k_ii = acc_mat_k_i[i];
                acc_mat_k_ii += mat_k_ii;
                mat_k_ii = acc_mat_k_ii;
            }
            mat_k_ii += inv_spectral_densities[i];

            for (long j = i + 1; j < e; ++j) {
                double &mat_k_ij = mat_k_i[j];
                mat_k_ij = inv_sigmas_phi_i.dot(phi.col(j));
                if (acc_mat_k_i != nullptr) {
                    double &acc_mat_k_ij = acc_mat_k_i[j];
                    acc_mat_k_ij += mat_k_ij;
                    mat_k_ij = acc_mat_k_ij;
                }
                mat_k(j, i) = mat_k_ij;
            }
        }

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
        bool predict_gradient,
        Eigen::MatrixXd &mat_k) const {

        long dims = m_setting_->x_dim;
        if (dims <= 0) { dims = mat_x1.rows(); }  // if x_dim is not set, use the number of rows of mat_x1
        const long e = m_setting_->GetFrequencies().cols();
        const long m = predict_gradient ? num_samples2 * (dims + 1) : num_samples2;
        ERL_DEBUG_ASSERT(mat_k.rows() >= e, "mat_k.rows() = {}, it should be >= {}.", mat_k.rows(), e);
        ERL_DEBUG_ASSERT(mat_k.cols() >= m, "mat_k.cols() = {}, it should be >= {}.", mat_k.cols(), m);
        Eigen::VectorXl grad_flags = Eigen::VectorXl::Ones(num_samples2);
        if (predict_gradient) {
            mat_k.topLeftCorner(e, m) = ComputeEigenFunctionsWithGradient(mat_x2, dims, num_samples2, grad_flags).transpose();
        } else {
            mat_k.topLeftCorner(e, m) = ComputeEigenFunctions(mat_x2, dims, num_samples2).transpose();
        }
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

    bool
    ReducedRankCovariance::operator==(const erl::covariance::ReducedRankCovariance &other) const {
        if (m_setting_ == nullptr && other.m_setting_ != nullptr) { return false; }
        if (m_setting_ != nullptr && (other.m_setting_ == nullptr || *m_setting_ != *other.m_setting_)) { return false; }
        if (m_coord_origin_.size() != other.m_coord_origin_.size() ||
            std::memcmp(m_coord_origin_.data(), other.m_coord_origin_.data(), m_coord_origin_.size() * sizeof(double)) != 0) {
            return false;
        }
        if (m_mat_k_.size() != other.m_mat_k_.size() || std::memcmp(m_mat_k_.data(), other.m_mat_k_.data(), m_mat_k_.size() * sizeof(double)) != 0) {
            return false;
        }
        if (m_vec_alpha_.size() != other.m_vec_alpha_.size() ||
            std::memcmp(m_vec_alpha_.data(), other.m_vec_alpha_.data(), m_vec_alpha_.size() * sizeof(double)) != 0) {
            return false;
        }
        return true;
    }

    bool
    ReducedRankCovariance::Write(const std::string &filename) const {
        ERL_INFO("Writing ReducedRankCovariance to file: {}", filename);
        std::filesystem::create_directories(std::filesystem::path(filename).parent_path());
        std::ofstream file(filename, std::ios_base::out | std::ios_base::binary);
        if (!file.is_open()) {
            ERL_WARN("Failed to open file: {}", filename);
            return false;
        }
        const bool success = Write(file);
        file.close();
        return success;
    }

    static const std::string kFileHeader = "# erl::covariance::ReducedRankCovariance";

    bool
    ReducedRankCovariance::Write(std::ostream &s) const {
        s << kFileHeader << std::endl  //
          << "# (feel free to add / change comments, but leave the first line as it is!)" << std::endl
          << "setting" << std::endl;
        // write setting
        if (!m_setting_->Write(s)) {
            ERL_WARN("Failed to write setting.");
            return false;
        }
        // write data
        s << "coord_origin" << std::endl;
        if (!common::SaveEigenMatrixToBinaryStream(s, m_coord_origin_)) {
            ERL_WARN("Failed to write coord_origin.");
            return false;
        }
        s << "mat_k" << std::endl;
        if (!common::SaveEigenMatrixToBinaryStream(s, m_mat_k_)) {
            ERL_WARN("Failed to write mat_k.");
            return false;
        }
        s << "vec_alpha" << std::endl;
        if (!common::SaveEigenMatrixToBinaryStream(s, m_vec_alpha_)) {
            ERL_WARN("Failed to write vec_alpha.");
            return false;
        }
        s << "end_of_ReducedRankCovariance" << std::endl;
        return s.good();
    }

    bool
    ReducedRankCovariance::Read(const std::string &filename) {
        ERL_INFO("Reading ReducedRankCovariance from file: {}", std::filesystem::absolute(filename));
        std::ifstream file(filename.c_str(), std::ios_base::in | std::ios_base::binary);
        if (!file.is_open()) {
            ERL_WARN("Failed to open file: {}", filename.c_str());
            return false;
        }
        const bool success = Read(file);
        file.close();
        return success;
    }

    bool
    ReducedRankCovariance::Read(std::istream &s) {
        if (!s.good()) {
            ERL_WARN("Input stream is not ready for reading");
            return false;
        }

        // check if the first line is valid
        std::string line;
        std::getline(s, line);
        if (line.compare(0, kFileHeader.length(), kFileHeader) != 0) {  // check if the first line is valid
            ERL_WARN("Header does not start with \"{}\"", kFileHeader.c_str());
            return false;
        }

        auto skip_line = [&s]() {
            char c;
            do { c = static_cast<char>(s.get()); } while (s.good() && c != '\n');
        };

        static const char *tokens[] = {
            "setting",
            "coord_origin",
            "mat_k",
            "vec_alpha",
            "end_of_ReducedRankCovariance",
        };

        // read data
        std::string token;
        int token_idx = 0;
        while (s.good()) {
            s >> token;
            if (token.compare(0, 1, "#") == 0) {
                skip_line();  // comment line, skip forward until end of line
                continue;
            }
            // non-comment line
            if (token != tokens[token_idx]) {
                ERL_WARN("Expected token {}, got {}.", tokens[token_idx], token);  // check token
                return false;
            }
            // reading state machine
            switch (token_idx) {
                case 0: {         // setting
                    skip_line();  // skip the line to read the binary data section
                    if (!m_setting_->Read(s)) {
                        ERL_WARN("Failed to read setting.");
                        return false;
                    }
                    break;
                }
                case 1: {  // coord_origin
                    skip_line();
                    if (!common::LoadEigenMatrixFromBinaryStream(s, m_coord_origin_)) {
                        ERL_WARN("Failed to read coord_origin.");
                        return false;
                    }
                    break;
                }
                case 2: {  // mat_k
                    skip_line();
                    if (!common::LoadEigenMatrixFromBinaryStream(s, m_mat_k_)) {
                        ERL_WARN("Failed to read mat_k.");
                        return false;
                    }
                    break;
                }
                case 3: {  // vec_alpha
                    skip_line();
                    if (!common::LoadEigenMatrixFromBinaryStream(s, m_vec_alpha_)) {
                        ERL_WARN("Failed to read vec_alpha.");
                        return false;
                    }
                    break;
                }
                case 4: {  // end_of_ReducedRankCovariance
                    skip_line();
                    return true;
                }
                default: {  // should not reach here
                    ERL_FATAL("Internal error, should not reach here.");
                }
            }
            ++token_idx;
        }
        ERL_WARN("Failed to read Covariance. Truncated file?");
        return false;  // should not reach here
    }

}  // namespace erl::covariance
