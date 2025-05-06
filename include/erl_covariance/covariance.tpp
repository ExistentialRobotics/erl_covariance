#pragma once

#include "erl_common/serialization.hpp"

namespace erl::covariance {
    template<typename Dtype>
    YAML::Node
    Covariance<Dtype>::Setting::YamlConvertImpl::encode(const Setting &setting) {
        YAML::Node node(YAML::NodeType::Map);
        ERL_YAML_SAVE_ATTR(node, setting, x_dim);
        ERL_YAML_SAVE_ATTR(node, setting, scale);
        ERL_YAML_SAVE_ATTR(node, setting, scale_mix);
        ERL_YAML_SAVE_ATTR(node, setting, weights);
        return node;
    }

    template<typename Dtype>
    bool
    Covariance<Dtype>::Setting::YamlConvertImpl::decode(const YAML::Node &node, Setting &setting) {
        if (!node.IsMap()) { return false; }
        ERL_YAML_LOAD_ATTR_TYPE(node, setting, x_dim, long);
        ERL_YAML_LOAD_ATTR_TYPE(node, setting, scale, Dtype);
        ERL_YAML_LOAD_ATTR_TYPE(node, setting, scale_mix, Dtype);
        ERL_YAML_LOAD_ATTR_TYPE(node, setting, weights, VectorX);
        return true;
    }

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
    template<typename Derived>
    bool
    Covariance<Dtype>::Register(std::string covariance_type) {
        return Factory::GetInstance().template Register<Derived>(
            covariance_type,
            [](std::shared_ptr<Setting> setting) {
                auto covariance_setting =
                    std::dynamic_pointer_cast<typename Derived::Setting>(setting);
                if (setting == nullptr) {
                    covariance_setting = std::make_shared<typename Derived::Setting>();
                }
                ERL_ASSERTM(
                    covariance_setting != nullptr,
                    "Failed to cast setting for derived Covariance of type {}.",
                    typeid(Derived).name());
                return std::make_shared<Derived>(covariance_setting);
            });
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
        long n = num_samples + num_samples_with_gradient * num_gradient_dimensions;
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
    Covariance<Dtype>::ComputeKtestSparse(
        const Eigen::Ref<const MatrixX> &mat_x1,
        const long num_samples1,
        const Eigen::Ref<const MatrixX> &mat_x2,
        const long num_samples2,
        const Dtype zero_threshold,
        SparseMatrix &mat_k) const {
        // default implementation, not efficient
        const auto [rows, cols] = GetMinimumKtestSize(num_samples1, 0, 0, num_samples2, false);
        MatrixX mat_k_dense(rows, cols);
        (void) ComputeKtest(mat_x1, num_samples1, mat_x2, num_samples2, mat_k_dense);
        mat_k = mat_k_dense.sparseView(zero_threshold);
        return {rows, cols};
    }

    template<typename Dtype>
    std::pair<long, long>
    Covariance<Dtype>::ComputeKtestWithGradientSparse(
        const Eigen::Ref<const MatrixX> &mat_x1,
        const long num_samples1,
        const Eigen::Ref<const Eigen::VectorXl> &vec_grad1_flags,
        const Eigen::Ref<const MatrixX> &mat_x2,
        const long num_samples2,
        const bool predict_gradient,
        const Dtype zero_threshold,
        SparseMatrix &mat_k) const {

        const long num_train_samples_with_gradient = vec_grad1_flags.head(num_samples1).count();
        const auto [rows, cols] = GetMinimumKtestSize(
            num_samples1,
            num_train_samples_with_gradient,
            mat_x1.rows(),
            num_samples2,
            predict_gradient);
        MatrixX mat_k_dense(rows, cols);
        (void) ComputeKtestWithGradient(
            mat_x1,
            num_samples1,
            vec_grad1_flags,
            mat_x2,
            num_samples2,
            predict_gradient,
            mat_k_dense);
        mat_k = mat_k_dense.sparseView(zero_threshold);
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
        static const std::vector<
            std::pair<const char *, std::function<bool(const Covariance *, std::ostream &)>>>
            token_function_pairs = {
                {
                    "setting",
                    [](const Covariance *cov, std::ostream &stream) -> bool {
                        return cov->m_setting_->Write(stream) && stream.good();
                    },
                },
            };
        return common::WriteTokens(s, this, token_function_pairs);
    }

    template<typename Dtype>
    bool
    Covariance<Dtype>::Read(std::istream &s) {
        static const std::vector<
            std::pair<const char *, std::function<bool(Covariance *, std::istream &)>>>
            token_function_pairs = {
                {
                    "setting",
                    [](Covariance *cov, std::istream &stream) -> bool {
                        return cov->m_setting_->Read(stream) && stream.good();
                    },
                },
            };
        return common::ReadTokens(s, this, token_function_pairs);
    }

    template<typename Dtype>
    Covariance<Dtype>::Covariance(std::shared_ptr<Setting> setting)
        : m_setting_(std::move(setting)) {}
}  // namespace erl::covariance
