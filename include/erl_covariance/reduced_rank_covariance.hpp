#pragma once

#include "covariance.hpp"

#include "erl_common/eigen.hpp"
#include "erl_common/yaml.hpp"

#include <functional>
#include <memory>

namespace erl::covariance {

    template<typename Dtype>
    class ReducedRankCovariance : public Covariance<Dtype> {

    public:
        using Super = Covariance<Dtype>;
        using MatrixX = Eigen::MatrixX<Dtype>;
        using VectorX = Eigen::VectorX<Dtype>;

        struct Setting : public common::Yamlable<Setting, typename Super::Setting> {
            // maximum number of basis functions per dimension, -1 means no limit
            long max_num_basis = -1;
            // number of basis functions per dimension
            Eigen::VectorXl num_basis;
            VectorX boundaries;       // boundaries for the basis functions per dimension
            bool accumulated = true;  // whether to accumulate the kernel matrix or not

        private:
            // the following members should be computed once and shared among all instances that
            // refer to the same setting

            std::mutex m_mutex_;                // mutex for building spectral densities
            bool volatile m_is_built_ = false;  // whether the spectral densities are built or not

            // frequencies for the basis functions, each column is a frequency vector,
            // (fx, fy, fz, ...)
            MatrixX m_frequencies_;
            // spectral densities for the basis functions, (s1, s2, s3, ...)
            VectorX m_spectral_densities_;
            // inverse spectral densities for the basis functions, (1/s1, 1/s2, 1/s3, ...)
            VectorX m_inv_spectral_densities_;

        public:
            void
            BuildSpectralDensities(
                const std::function<VectorX(const VectorX & /*freq_squared_norm*/)>
                    &kernel_spectral_density_func);

            void
            ResetSpectralDensities();

            [[nodiscard]] const MatrixX &
            GetFrequencies() const;

            [[nodiscard]] const VectorX &
            GetSpectralDensities() const;

            [[nodiscard]] const VectorX &
            GetInvSpectralDensities() const;

            struct YamlConvertImpl {
                static YAML::Node
                encode(const Setting &setting);

                static bool
                decode(const YAML::Node &node, Setting &setting);
            };
        };

    protected:
        std::shared_ptr<Setting> m_setting_ = nullptr;
        VectorX m_coord_origin_;  // origin of the coordinate system for the basis functions
        MatrixX m_mat_k_;         // accumulated kernel matrix approximation
        MatrixX m_alpha_;         // accumulated alpha vector(matrix) approximation

    public:
        explicit ReducedRankCovariance(std::shared_ptr<Setting> setting);

        ReducedRankCovariance(const ReducedRankCovariance &other) = default;
        ReducedRankCovariance(ReducedRankCovariance &&other) = default;
        ReducedRankCovariance &
        operator=(const ReducedRankCovariance &other) = default;
        ReducedRankCovariance &
        operator=(ReducedRankCovariance &&other) = default;

        [[nodiscard]] std::pair<long, long>
        GetMinimumKtrainSize(
            long /*num_samples*/,
            long /*num_samples_with_gradient*/,
            long /*num_gradient_dimensions*/) const override;

        [[nodiscard]] std::pair<long, long>
        GetMinimumKtestSize(
            long /*num_train_samples*/,
            long /*num_train_samples_with_gradient*/,
            long num_gradient_dimensions,
            long num_test_queries,
            bool predict_gradient) const override;

        std::pair<long, long>
        ComputeKtrain(
            const Eigen::Ref<const MatrixX> &mat_x,
            long num_samples,
            MatrixX &mat_k,
            MatrixX &mat_alpha) override;

        std::pair<long, long>
        ComputeKtrain(
            const Eigen::Ref<const MatrixX> &mat_x,
            const Eigen::Ref<const VectorX> &vec_var_y,
            long num_samples,
            MatrixX &mat_k,
            MatrixX &mat_alpha) override;

        std::pair<long, long>
        ComputeKtrainWithGradient(
            const Eigen::Ref<const MatrixX> &mat_x,
            long num_samples,
            Eigen::VectorXl &vec_grad_flags,
            MatrixX &mat_k,
            MatrixX &mat_alpha) override;

        std::pair<long, long>
        ComputeKtrainWithGradient(
            const Eigen::Ref<const MatrixX> &mat_x,
            long num_samples,
            Eigen::VectorXl &vec_grad_flags,
            const Eigen::Ref<const VectorX> &vec_var_x,
            const Eigen::Ref<const VectorX> &vec_var_y,
            const Eigen::Ref<const VectorX> &vec_var_grad,
            MatrixX &mat_k,
            MatrixX &mat_alpha) override;

        std::pair<long, long>
        ComputeKtest(
            const Eigen::Ref<const MatrixX> &mat_x1,
            long num_samples1,
            const Eigen::Ref<const MatrixX> &mat_x2,
            long num_samples2,
            MatrixX &mat_k) const override;

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
        BuildSpectralDensities();

        [[nodiscard]] virtual VectorX
        ComputeSpectralDensities(const VectorX &freq_squared_norm) const = 0;

        [[nodiscard]] MatrixX
        ComputeEigenFunctions(const Eigen::Ref<const MatrixX> &mat_x, long dims, long num_samples)
            const;

        [[nodiscard]] MatrixX
        ComputeEigenFunctionsWithGradient(
            const Eigen::Ref<const MatrixX> &mat_x,
            long dims,
            long num_samples,
            Eigen::VectorXl &vec_grad_flags) const;

        [[nodiscard]] VectorX
        GetCoordOrigin() const;

        void
        SetCoordOrigin(const VectorX &coord_origin);

        [[nodiscard]] const MatrixX &
        GetKtrain() const;

        MatrixX &
        GetKtrainBuffer();

        [[nodiscard]] const MatrixX &
        GetAlpha() const;

        MatrixX &
        GetAlphaBuffer();

        [[nodiscard]] bool
        operator==(const ReducedRankCovariance &other) const;

        [[nodiscard]] bool
        operator!=(const ReducedRankCovariance &other) const;

        [[nodiscard]] bool
        Write(std::ostream &s) const override;

        [[nodiscard]] bool
        Read(std::istream &s) override;
    };

}  // namespace erl::covariance

#include "reduced_rank_covariance.tpp"

template<>
struct YAML::convert<erl::covariance::ReducedRankCovariance<double>::Setting>
    : erl::covariance::ReducedRankCovariance<double>::Setting::YamlConvertImpl {};

template<>
struct YAML::convert<erl::covariance::ReducedRankCovariance<float>::Setting>
    : erl::covariance::ReducedRankCovariance<float>::Setting::YamlConvertImpl {};
