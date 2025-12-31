#include "erl_covariance/covariance.hpp"

#include "erl_common/exception.hpp"
#include "erl_common/serialization.hpp"

namespace erl::covariance {
    template<typename Dtype>
    std::size_t
    Covariance<Dtype>::GetMemoryUsage() const {
        std::size_t memory_usage = sizeof(*this);
        if (m_setting_ != nullptr) { memory_usage += sizeof(Setting); }
        return memory_usage;
    }

    template<typename Dtype>
    std::shared_ptr<Covariance<Dtype>>
    Covariance<Dtype>::CreateCovariance(
        const std::string &covariance_type,
        std::shared_ptr<Setting> setting) {
        return Factory::GetInstance().Create(covariance_type, std::move(setting));
    }

    template<typename Dtype>
    std::shared_ptr<typename Covariance<Dtype>::Setting>
    Covariance<Dtype>::GetSetting() const {
        return m_setting_;
    }

    template<typename Dtype>
    std::pair<long, long>
    Covariance<Dtype>::GetMinimumKtrainSize(
        const long num_samples,
        const long num_samples_with_gradient,
        const long num_gradient_dimensions) const {
        const long n = num_samples + num_samples_with_gradient * num_gradient_dimensions;
        return {n, n};
    }

    template<typename Dtype>
    std::pair<long, long>
    Covariance<Dtype>::GetMinimumKtestSize(
        const long num_train_samples,
        const long num_train_samples_with_gradient,
        const long num_gradient_dimensions,
        const long num_test_queries,
        const bool predict_gradient) const {
        return {
            num_train_samples + num_train_samples_with_gradient * num_gradient_dimensions,
            predict_gradient ? num_test_queries * (1 + num_gradient_dimensions) : num_test_queries};
    }

    template<typename Dtype>
    std::pair<long, long>
    Covariance<Dtype>::ComputeKtrain(
        const Eigen::Ref<const MatrixX> &mat_x,
        const long num_samples,
        MatrixX &mat_k) {
        MatrixX mat_alpha;
        return ComputeKtrain(mat_x, num_samples, mat_k, mat_alpha);
    }

    template<typename Dtype>
    std::pair<long, long>
    Covariance<Dtype>::ComputeKtrain(
        const Eigen::Ref<const MatrixX> & /*mat_x*/,
        long /*num_samples*/,
        Dtype /*exp_bias*/,
        MatrixX & /*mat_k*/,
        MatrixX & /*mat_alpha*/) {
        throw NotImplemented(__PRETTY_FUNCTION__);
    }

    template<typename Dtype>
    std::pair<long, long>
    Covariance<Dtype>::ComputeKtrain(
        const Eigen::Ref<const MatrixX> &mat_x,
        const long num_samples,
        const Dtype exp_bias,
        MatrixX &mat_k) {
        MatrixX mat_alpha;
        return ComputeKtrain(mat_x, num_samples, exp_bias, mat_k, mat_alpha);
    }

    template<typename Dtype>
    std::pair<long, long>
    Covariance<Dtype>::ComputeKtrain(
        const Eigen::Ref<const MatrixX> &mat_x,
        const Eigen::Ref<const VectorX> &vec_var_y,
        const long num_samples,
        MatrixX &mat_k) {
        MatrixX mat_alpha;
        return ComputeKtrain(mat_x, vec_var_y, num_samples, mat_k, mat_alpha);
    }

    template<typename Dtype>
    std::pair<long, long>
    Covariance<Dtype>::ComputeKtrain(
        const Eigen::Ref<const MatrixX> & /*mat_x*/,
        const Eigen::Ref<const VectorX> & /*vec_var_y*/,
        long /*num_samples*/,
        Dtype /*exp_bias*/,
        MatrixX & /*mat_k*/,
        MatrixX & /*mat_alpha*/) {
        throw NotImplemented(__PRETTY_FUNCTION__);
    }

    template<typename Dtype>
    std::pair<long, long>
    Covariance<Dtype>::ComputeKtestSparse(
        const Eigen::Ref<const MatrixX> &mat_x1,
        const long num_samples,
        const Eigen::Ref<const MatrixX> &mat_x2,
        const long num_queries,
        const Dtype zero_threshold,
        SparseMatrix &mat_k) const {
        // default implementation, not efficient
        const auto [rows, cols] = GetMinimumKtestSize(num_samples, 0, 0, num_queries, false);
        MatrixX mat_k_dense(rows, cols);
        (void) ComputeKtest(mat_x1, num_samples, mat_x2, num_queries, mat_k_dense);
        mat_k = mat_k_dense.sparseView(zero_threshold);
        mat_k.makeCompressed();
        return {rows, cols};
    }

    template<typename Dtype>
    std::pair<long, long>
    Covariance<Dtype>::ComputeKtest(
        const Eigen::Ref<const MatrixX> & /*mat_x1*/,
        long /*num_samples*/,
        const Eigen::Ref<const MatrixX> & /*mat_x2*/,
        long /*num_queries*/,
        Dtype /*exp_bias*/,
        MatrixX & /*mat_k*/) const {
        throw NotImplemented(__PRETTY_FUNCTION__);
    }

    template<typename Dtype>
    std::pair<long, long>
    Covariance<Dtype>::ComputeKtestSparse(
        const Eigen::Ref<const MatrixX> &mat_x1,
        const long num_samples,
        const Eigen::Ref<const MatrixX> &mat_x2,
        const long num_queries,
        const Dtype exp_bias,
        const Dtype zero_threshold,
        SparseMatrix &mat_k) const {
        // default implementation, not efficient
        const auto [rows, cols] = GetMinimumKtestSize(num_samples, 0, 0, num_queries, false);
        MatrixX mat_k_dense(rows, cols);
        (void) ComputeKtest(mat_x1, num_samples, mat_x2, num_queries, exp_bias, mat_k_dense);
        mat_k = mat_k_dense.sparseView(zero_threshold);
        mat_k.makeCompressed();
        return {rows, cols};
    }

    template<typename Dtype>
    std::pair<long, long>
    Covariance<Dtype>::ComputeKtrainWithGradient(
        const Eigen::Ref<const MatrixX> &mat_x,
        long num_samples,
        Eigen::VectorXl &vec_grad_flags,
        MatrixX &mat_k) {
        MatrixX mat_alpha;
        return ComputeKtrainWithGradient(mat_x, num_samples, vec_grad_flags, mat_k, mat_alpha);
    }

    template<typename Dtype>
    std::pair<long, long>
    Covariance<Dtype>::ComputeKtrainWithGradient(
        const Eigen::Ref<const MatrixX> & /*mat_x*/,
        long /*num_samples*/,
        Dtype /*exp_bias*/,
        Eigen::VectorXl & /*vec_grad_flags*/,
        MatrixX & /*mat_k*/,
        MatrixX & /*mat_alpha*/) {
        throw NotImplemented(__PRETTY_FUNCTION__);
    }

    template<typename Dtype>
    std::pair<long, long>
    Covariance<Dtype>::ComputeKtrainWithGradient(
        const Eigen::Ref<const MatrixX> &mat_x,
        long num_samples,
        Dtype exp_bias,
        Eigen::VectorXl &vec_grad_flags,
        MatrixX &mat_k) {
        MatrixX mat_alpha;
        return ComputeKtrainWithGradient(
            mat_x,
            num_samples,
            exp_bias,
            vec_grad_flags,
            mat_k,
            mat_alpha);
    }

    template<typename Dtype>
    std::pair<long, long>
    Covariance<Dtype>::ComputeKtrainWithGradient(
        const Eigen::Ref<const MatrixX> &mat_x,
        long num_samples,
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

    template<typename Dtype>
    std::pair<long, long>
    Covariance<Dtype>::ComputeKtrainWithGradient(
        const Eigen::Ref<const MatrixX> & /*mat_x*/,
        long /*num_samples*/,
        Dtype /*exp_bias*/,
        Eigen::VectorXl & /*vec_grad_flags*/,
        const Eigen::Ref<const VectorX> & /*vec_var_x*/,
        const Eigen::Ref<const VectorX> & /*vec_var_y*/,
        const Eigen::Ref<const VectorX> & /*vec_var_grad*/,
        MatrixX & /*mat_k*/,
        MatrixX & /*mat_alpha*/) {
        throw NotImplemented(__PRETTY_FUNCTION__);
    }

    template<typename Dtype>
    std::pair<long, long>
    Covariance<Dtype>::ComputeKtrainWithGradient(
        const Eigen::Ref<const MatrixX> &mat_x,
        long num_samples,
        Dtype exp_bias,
        Eigen::VectorXl &vec_grad_flags,
        const Eigen::Ref<const VectorX> &vec_var_x,
        const Eigen::Ref<const VectorX> &vec_var_y,
        const Eigen::Ref<const VectorX> &vec_var_grad,
        MatrixX &mat_k) {
        MatrixX mat_alpha;
        return ComputeKtrainWithGradient(
            mat_x,
            num_samples,
            exp_bias,
            vec_grad_flags,
            vec_var_x,
            vec_var_y,
            vec_var_grad,
            mat_k,
            mat_alpha);
    }

    template<typename Dtype>
    std::pair<long, long>
    Covariance<Dtype>::ComputeKtestWithGradientSparse(
        const Eigen::Ref<const MatrixX> &mat_x1,
        const long num_samples,
        const Eigen::Ref<const Eigen::VectorXl> &vec_grad1_flags,
        const Eigen::Ref<const MatrixX> &mat_x2,
        const long num_queries,
        const bool predict_gradient,
        const Dtype zero_threshold,
        SparseMatrix &mat_k) const {

        const long num_train_samples_with_gradient = vec_grad1_flags.head(num_samples).count();
        const auto [rows, cols] = GetMinimumKtestSize(
            num_samples,
            num_train_samples_with_gradient,
            mat_x1.rows(),
            num_queries,
            predict_gradient);
        MatrixX mat_k_dense(rows, cols);
        (void) ComputeKtestWithGradient(
            mat_x1,
            num_samples,
            vec_grad1_flags,
            mat_x2,
            num_queries,
            predict_gradient,
            mat_k_dense);
        mat_k = mat_k_dense.sparseView(zero_threshold);
        mat_k.makeCompressed();
        return {rows, cols};
    }

    template<typename Dtype>
    std::pair<long, long>
    Covariance<Dtype>::ComputeKtestWithGradient(
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

    template<typename Dtype>
    std::pair<long, long>
    Covariance<Dtype>::ComputeKtestWithGradientSparse(
        const Eigen::Ref<const MatrixX> &mat_x1,
        const long num_samples,
        const Eigen::Ref<const Eigen::VectorXl> &vec_grad1_flags,
        const Eigen::Ref<const MatrixX> &mat_x2,
        const long num_queries,
        const bool predict_gradient,
        const Dtype exp_bias,
        const Dtype zero_threshold,
        SparseMatrix &mat_k) const {

        const long num_train_samples_with_gradient = vec_grad1_flags.head(num_samples).count();
        const auto [rows, cols] = GetMinimumKtestSize(
            num_samples,
            num_train_samples_with_gradient,
            mat_x1.rows(),
            num_queries,
            predict_gradient);
        MatrixX mat_k_dense(rows, cols);
        (void) ComputeKtestWithGradient(
            mat_x1,
            num_samples,
            vec_grad1_flags,
            mat_x2,
            num_queries,
            predict_gradient,
            exp_bias,
            mat_k_dense);
        mat_k = mat_k_dense.sparseView(zero_threshold);
        mat_k.makeCompressed();
        return {rows, cols};
    }

    template<typename Dtype>
    bool
    Covariance<Dtype>::operator==(const Covariance &other) const {
        if (m_setting_ == nullptr && other.m_setting_ != nullptr) { return false; }
        if (m_setting_ != nullptr &&
            (other.m_setting_ == nullptr || *m_setting_ != *other.m_setting_)) {
            return false;
        }
        return true;
    }

    template<typename Dtype>
    bool
    Covariance<Dtype>::operator!=(const Covariance &other) const {
        return !(*this == other);
    }

    template<typename Dtype>
    bool
    Covariance<Dtype>::Write(std::ostream &s) const {
        using namespace common::serialization;
        static const TokenWriteFunctionPairs<Covariance> token_function_pairs = {
            {
                "setting",
                [](const Covariance *cov, std::ostream &stream) -> bool {
                    return cov->m_setting_->Write(stream) && stream.good();
                },
            },
        };
        return WriteTokens(s, this, token_function_pairs);
    }

    template<typename Dtype>
    bool
    Covariance<Dtype>::Read(std::istream &s) {
        using namespace common::serialization;
        static const TokenReadFunctionPairs<Covariance> token_function_pairs = {
            {
                "setting",
                [](Covariance *cov, std::istream &stream) -> bool {
                    return cov->m_setting_->Read(stream) && stream.good();
                },
            },
        };
        return ReadTokens(s, this, token_function_pairs);
    }

    template<typename Dtype>
    Covariance<Dtype>::Covariance(std::shared_ptr<Setting> setting)
        : m_setting_(std::move(setting)) {}

    template class Covariance<double>;
    template class Covariance<float>;
}  // namespace erl::covariance
