#pragma once

#include "covariance.hpp"

#include "erl_common/eigen.hpp"
#include "erl_common/storage_order.hpp"
#include "erl_common/yaml.hpp"

#include <functional>
#include <memory>

namespace erl::covariance {

    class ReducedRankCovariance : public Covariance {

    public:
        struct Setting : public common::Yamlable<Setting, Covariance::Setting> {
            long max_num_basis = -1;     // maximum number of basis functions per dimension, -1 means no limit
            Eigen::VectorXl num_basis;   // number of basis functions per dimension
            Eigen::VectorXd boundaries;  // boundaries for the basis functions per dimension
            bool accumulated = true;     // whether to accumulate the kernel matrix or not

        private:
            // the following members should be computed once and shared among all instances that refer to the same setting
            std::mutex m_mutex_;                        // mutex for building spectral densities
            bool volatile m_is_built_ = false;          // whether the spectral densities are built or not
            Eigen::MatrixXd m_frequencies_;             // frequencies for the basis functions, each column is a frequency vector, (fx, fy, fz, ...)
            Eigen::VectorXd m_spectral_densities_;      // spectral densities for the basis functions, (s1, s2, s3, ...)
            Eigen::VectorXd m_inv_spectral_densities_;  // inverse spectral densities for the basis functions, (1/s1, 1/s2, 1/s3, ...)

        public:
            void
            BuildSpectralDensities(const std::function<Eigen::VectorXd(const Eigen::VectorXd & /*freq_squared_norm*/)> &kernel_spectral_density_func);

            void
            ResetSpectralDensities() {
                m_is_built_ = false;
            }

            [[nodiscard]] const Eigen::MatrixXd &
            GetFrequencies() const {
                ERL_DEBUG_ASSERT(m_is_built_, "Spectral densities are not built yet");
                return m_frequencies_;
            }

            [[nodiscard]] const Eigen::VectorXd &
            GetSpectralDensities() const {
                ERL_DEBUG_ASSERT(m_is_built_, "Spectral densities are not built yet");
                return m_spectral_densities_;
            }

            [[nodiscard]] const Eigen::VectorXd &
            GetInvSpectralDensities() const {
                ERL_DEBUG_ASSERT(m_is_built_, "Spectral densities are not built yet");
                return m_inv_spectral_densities_;
            }
        };

        inline static const volatile bool kSettingRegistered = common::YamlableBase::Register<Setting>();

    protected:
        std::shared_ptr<Setting> m_setting_ = nullptr;
        Eigen::VectorXd m_coord_origin_;  // origin of the coordinate system for the basis functions
        Eigen::MatrixXd m_mat_k_;         // accumulated kernel matrix approximation
        Eigen::VectorXd m_vec_alpha_;     // accumulated alpha vector approximation

    public:
        explicit ReducedRankCovariance(std::shared_ptr<Setting> setting)
            : Covariance(setting),
              m_setting_(std::move(setting)) {
            ERL_WARN_COND(
                m_setting_->boundaries.size() != m_setting_->x_dim,
                "Boundaries size ({}) does not match x_dim ({})",
                m_setting_->boundaries.size(),
                m_setting_->x_dim);
        }

        ReducedRankCovariance(const ReducedRankCovariance &other) = default;
        ReducedRankCovariance(ReducedRankCovariance &&other) = default;
        ReducedRankCovariance &
        operator=(const ReducedRankCovariance &other) = default;
        ReducedRankCovariance &
        operator=(ReducedRankCovariance &&other) = default;

        [[nodiscard]] std::pair<long, long>
        GetMinimumKtrainSize(const long /*num_samples*/, const long /*num_samples_with_gradient*/, const long /*num_gradient_dimensions*/) const override {
            long e = m_setting_->num_basis.prod();
            return {e, e};
        }

        [[nodiscard]] std::pair<long, long>
        GetMinimumKtestSize(
            const long /*num_train_samples*/,
            const long /*num_train_samples_with_gradient*/,
            const long num_gradient_dimensions,
            const long num_test_queries,
            const bool predict_gradient) const override {
            long e = m_setting_->num_basis.prod();
            return {e, predict_gradient ? num_test_queries * (1 + num_gradient_dimensions) : num_test_queries};
        }

        std::pair<long, long>
        ComputeKtrain(const Eigen::Ref<const Eigen::MatrixXd> &mat_x, long num_samples, Eigen::MatrixXd &mat_k, Eigen::VectorXd &vec_alpha) const override;

        std::pair<long, long>
        ComputeKtrain(
            const Eigen::Ref<const Eigen::MatrixXd> &mat_x,
            const Eigen::Ref<const Eigen::VectorXd> &vec_var_y,
            long num_samples,
            Eigen::MatrixXd &mat_k,
            Eigen::VectorXd &vec_alpha) const override;

        std::pair<long, long>
        ComputeKtrainWithGradient(
            const Eigen::Ref<const Eigen::MatrixXd> &mat_x,
            long num_samples,
            Eigen::VectorXl &vec_grad_flags,
            Eigen::MatrixXd &mat_k,
            Eigen::VectorXd &vec_alpha) const override;

        std::pair<long, long>
        ComputeKtrainWithGradient(
            const Eigen::Ref<const Eigen::MatrixXd> &mat_x,
            long num_samples,
            Eigen::VectorXl &vec_grad_flags,
            const Eigen::Ref<const Eigen::VectorXd> &vec_var_x,
            const Eigen::Ref<const Eigen::VectorXd> &vec_var_y,
            const Eigen::Ref<const Eigen::VectorXd> &vec_var_grad,
            Eigen::MatrixXd &mat_k,
            Eigen::VectorXd &vec_alpha) const override;

        std::pair<long, long>
        ComputeKtest(
            const Eigen::Ref<const Eigen::MatrixXd> &mat_x1,
            long num_samples1,
            const Eigen::Ref<const Eigen::MatrixXd> &mat_x2,
            long num_samples2,
            Eigen::MatrixXd &mat_k) const override;

        std::pair<long, long>
        ComputeKtestWithGradient(
            const Eigen::Ref<const Eigen::MatrixXd> &mat_x1,
            long num_samples1,
            const Eigen::Ref<const Eigen::VectorXl> &vec_grad1_flags,
            const Eigen::Ref<const Eigen::MatrixXd> &mat_x2,
            long num_samples2,
            bool predict_gradient,
            Eigen::MatrixXd &mat_k) const override;

        void
        BuildSpectralDensities() {
            m_setting_->BuildSpectralDensities(
                [this](const Eigen::VectorXd &freq_squared_norm) -> Eigen::VectorXd { return ComputeSpectralDensities(freq_squared_norm); });
            const long e = m_setting_->num_basis.prod();
            if (m_setting_->accumulated) {
                if (m_mat_k_.size() == 0) { m_mat_k_ = Eigen::MatrixXd::Zero(e, e); }
                if (m_vec_alpha_.size() == 0) { m_vec_alpha_ = Eigen::VectorXd::Zero(e); }
            }
        }

        [[nodiscard]] virtual Eigen::VectorXd
        ComputeSpectralDensities(const Eigen::VectorXd &freq_squared_norm) const = 0;

        [[nodiscard]] Eigen::MatrixXd
        ComputeEigenFunctions(const Eigen::Ref<const Eigen::MatrixXd> &mat_x, long dims, long num_samples) const;

        [[nodiscard]] Eigen::MatrixXd
        ComputeEigenFunctionsWithGradient(const Eigen::Ref<const Eigen::MatrixXd> &mat_x, long dims, long num_samples, Eigen::VectorXl &vec_grad_flags) const;

        [[nodiscard]] Eigen::VectorXd
        GetCoordOrigin() const {
            return m_coord_origin_;
        }

        void
        SetCoordOrigin(const Eigen::VectorXd &coord_origin) {
            m_coord_origin_ = coord_origin;
        }

        [[nodiscard]] const Eigen::MatrixXd &
        GetKtrain() const {
            return m_mat_k_;
        }

        Eigen::MatrixXd &
        GetKtrainBuffer() {
            return m_mat_k_;
        }

        [[nodiscard]] const Eigen::VectorXd &
        GetAlpha() const {
            return m_vec_alpha_;
        }

        Eigen::VectorXd &
        GetAlphaBuffer() {
            return m_vec_alpha_;
        }

        [[nodiscard]] bool
        operator==(const ReducedRankCovariance &other) const;

        [[nodiscard]] bool
        operator!=(const ReducedRankCovariance &other) const {
            return !(*this == other);
        }

        [[nodiscard]] bool
        Write(const std::string &filename) const override;

        [[nodiscard]] bool
        Write(std::ostream &s) const override;

        [[nodiscard]] bool
        Read(const std::string &filename) override;

        [[nodiscard]] bool
        Read(std::istream &s) override;
    };

}  // namespace erl::covariance

template<>
struct YAML::convert<erl::covariance::ReducedRankCovariance::Setting> {
    static Node
    encode(const erl::covariance::ReducedRankCovariance::Setting &setting) {
        Node node = convert<erl::covariance::Covariance::Setting>::encode(setting);
        node["max_num_basis"] = setting.max_num_basis;
        node["num_basis"] = setting.num_basis;
        node["boundaries"] = setting.boundaries;
        node["accumulated"] = setting.accumulated;
        return node;
    }

    static bool
    decode(const Node &node, erl::covariance::ReducedRankCovariance::Setting &setting) {
        if (!convert<erl::covariance::Covariance::Setting>::decode(node, setting)) { return false; }
        setting.max_num_basis = node["max_num_basis"].as<long>();
        setting.num_basis = node["num_basis"].as<Eigen::VectorXl>();
        setting.boundaries = node["boundaries"].as<Eigen::VectorXd>();
        setting.accumulated = node["accumulated"].as<bool>();
        return true;
    }
};
