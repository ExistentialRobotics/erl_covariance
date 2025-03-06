#pragma once

#include "covariance.hpp"

#include "erl_common/eigen.hpp"
#include "erl_common/yaml.hpp"

#include <functional>
#include <memory>

namespace erl::covariance {

    template<typename Dtype>
    class ReducedRankCovariance : public Covariance<Dtype> {
        inline static const std::string kFileHeader = "# erl::covariance::ReducedRankCovariance";

    public:
        using Super = Covariance<Dtype>;
        using MatrixX = Eigen::MatrixX<Dtype>;
        using VectorX = Eigen::VectorX<Dtype>;

        struct Setting : Super::Setting {
            long max_num_basis = -1;    // maximum number of basis functions per dimension, -1 means no limit
            Eigen::VectorXl num_basis;  // number of basis functions per dimension
            VectorX boundaries;         // boundaries for the basis functions per dimension
            bool accumulated = true;    // whether to accumulate the kernel matrix or not

        private:
            // the following members should be computed once and shared among all instances that refer to the same setting
            std::mutex m_mutex_;                // mutex for building spectral densities
            bool volatile m_is_built_ = false;  // whether the spectral densities are built or not
            MatrixX m_frequencies_;             // frequencies for the basis functions, each column is a frequency vector, (fx, fy, fz, ...)
            VectorX m_spectral_densities_;      // spectral densities for the basis functions, (s1, s2, s3, ...)
            VectorX m_inv_spectral_densities_;  // inverse spectral densities for the basis functions, (1/s1, 1/s2, 1/s3, ...)

        public:
            void
            BuildSpectralDensities(const std::function<VectorX(const VectorX & /*freq_squared_norm*/)> &kernel_spectral_density_func);

            void
            ResetSpectralDensities() {
                m_is_built_ = false;
            }

            [[nodiscard]] const MatrixX &
            GetFrequencies() const {
                ERL_DEBUG_ASSERT(m_is_built_, "Spectral densities are not built yet");
                return m_frequencies_;
            }

            [[nodiscard]] const VectorX &
            GetSpectralDensities() const {
                ERL_DEBUG_ASSERT(m_is_built_, "Spectral densities are not built yet");
                return m_spectral_densities_;
            }

            [[nodiscard]] const VectorX &
            GetInvSpectralDensities() const {
                ERL_DEBUG_ASSERT(m_is_built_, "Spectral densities are not built yet");
                return m_inv_spectral_densities_;
            }

            struct YamlConvertImpl {
                static YAML::Node
                encode(const Setting &setting);

                static bool
                decode(const YAML::Node &node, Setting &setting);
            };
        };

        // inline static const volatile bool kSettingRegistered = common::YamlableBase::Register<Setting>();

    protected:
        std::shared_ptr<Setting> m_setting_ = nullptr;
        VectorX m_coord_origin_;  // origin of the coordinate system for the basis functions
        MatrixX m_mat_k_;         // accumulated kernel matrix approximation
        VectorX m_vec_alpha_;     // accumulated alpha vector approximation

    public:
        explicit ReducedRankCovariance(std::shared_ptr<Setting> setting)
            : Super(setting),
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
        ComputeKtrain(const Eigen::Ref<const MatrixX> &mat_x, long num_samples, MatrixX &mat_k, VectorX &vec_alpha) const override;

        std::pair<long, long>
        ComputeKtrain(const Eigen::Ref<const MatrixX> &mat_x, const Eigen::Ref<const VectorX> &vec_var_y, long num_samples, MatrixX &mat_k, VectorX &vec_alpha)
            const override;

        std::pair<long, long>
        ComputeKtrainWithGradient(const Eigen::Ref<const MatrixX> &mat_x, long num_samples, Eigen::VectorXl &vec_grad_flags, MatrixX &mat_k, VectorX &vec_alpha)
            const override;

        std::pair<long, long>
        ComputeKtrainWithGradient(
            const Eigen::Ref<const MatrixX> &mat_x,
            long num_samples,
            Eigen::VectorXl &vec_grad_flags,
            const Eigen::Ref<const VectorX> &vec_var_x,
            const Eigen::Ref<const VectorX> &vec_var_y,
            const Eigen::Ref<const VectorX> &vec_var_grad,
            MatrixX &mat_k,
            VectorX &vec_alpha) const override;

        std::pair<long, long>
        ComputeKtest(const Eigen::Ref<const MatrixX> &mat_x1, long num_samples1, const Eigen::Ref<const MatrixX> &mat_x2, long num_samples2, MatrixX &mat_k)
            const override;

        std::pair<long, long>
        ComputeKtestWithGradient(
            const Eigen::Ref<const MatrixX> &mat_x1,
            long num_samples1,
            const Eigen::Ref<const Eigen::VectorXl> &vec_grad1_flags,
            const Eigen::Ref<const MatrixX> &mat_x2,
            long num_samples2,
            bool predict_gradient,
            MatrixX &mat_k) const override;

        void
        BuildSpectralDensities() {
            m_setting_->BuildSpectralDensities([this](const VectorX &freq_squared_norm) -> VectorX { return ComputeSpectralDensities(freq_squared_norm); });
            const long e = m_setting_->num_basis.prod();
            if (m_setting_->accumulated) {
                if (m_mat_k_.size() == 0) { m_mat_k_ = MatrixX::Zero(e, e); }
                if (m_vec_alpha_.size() == 0) { m_vec_alpha_ = VectorX::Zero(e); }
            }
        }

        [[nodiscard]] virtual VectorX
        ComputeSpectralDensities(const VectorX &freq_squared_norm) const = 0;

        [[nodiscard]] MatrixX
        ComputeEigenFunctions(const Eigen::Ref<const MatrixX> &mat_x, long dims, long num_samples) const;

        [[nodiscard]] MatrixX
        ComputeEigenFunctionsWithGradient(const Eigen::Ref<const MatrixX> &mat_x, long dims, long num_samples, Eigen::VectorXl &vec_grad_flags) const;

        [[nodiscard]] VectorX
        GetCoordOrigin() const {
            return m_coord_origin_;
        }

        void
        SetCoordOrigin(const VectorX &coord_origin) {
            m_coord_origin_ = coord_origin;
        }

        [[nodiscard]] const MatrixX &
        GetKtrain() const {
            return m_mat_k_;
        }

        MatrixX &
        GetKtrainBuffer() {
            return m_mat_k_;
        }

        [[nodiscard]] const VectorX &
        GetAlpha() const {
            return m_vec_alpha_;
        }

        VectorX &
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

#include "reduced_rank_covariance.tpp"

template<>
struct YAML::convert<erl::covariance::ReducedRankCovariance<double>::Setting> : erl::covariance::ReducedRankCovariance<double>::Setting::YamlConvertImpl {};

template<>
struct YAML::convert<erl::covariance::ReducedRankCovariance<float>::Setting> : erl::covariance::ReducedRankCovariance<float>::Setting::YamlConvertImpl {};
