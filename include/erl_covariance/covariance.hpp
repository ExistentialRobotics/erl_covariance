#pragma once

#include "init.hpp"

#include "erl_common/eigen.hpp"
#include "erl_common/factory_pattern.hpp"
#include "erl_common/yaml.hpp"

#include <memory>

namespace erl::covariance {

    // ref1: https://peterroelants.github.io/posts/gaussian-process-kernels/
    // ref2: https://www.cs.toronto.edu/~duvenaud/cookbook/

    template<typename Dtype>
    class Covariance {
    public:
        using MatrixX = Eigen::MatrixX<Dtype>;
        using VectorX = Eigen::VectorX<Dtype>;

        // structure for holding the parameters
        struct Setting : common::Yamlable<Setting> {
            long x_dim = 2;        // dimension of input space
            Dtype alpha = 1.;      // overall covariance magnitude
            Dtype scale = 1.;      // scale length
            Dtype scale_mix = 1.;  // used by RationalQuadratic, decreasing this value allows more local variations, inf --> Gaussian kernel
            VectorX weights;       // used by some custom kernels

            struct YamlConvertImpl {
                static YAML::Node
                encode(const Setting &setting);

                static bool
                decode(const YAML::Node &node, Setting &setting);
            };
        };

        using Factory = common::FactoryPattern<Covariance, false, false, std::shared_ptr<Setting>>;

    protected:
        std::shared_ptr<Setting> m_setting_ = nullptr;

    private:
        inline static const std::string kFileHeader = "# erl::covariance::Covariance<Dtype>";

    public:
        virtual ~Covariance() = default;

        Covariance(const Covariance &) = default;
        Covariance(Covariance &&) = default;
        Covariance &
        operator=(const Covariance &) = default;
        Covariance &
        operator=(Covariance &&) = default;

        [[nodiscard]] std::size_t
        GetMemoryUsage() const;

        //-- factory pattern
        /**
         * returns actual class name as string for identification
         * @return The type of the tree.
         */
        [[nodiscard]] virtual std::string
        GetCovarianceType() const = 0;

        /**
         * Create a new covariance of the given type.
         * @param covariance_type
         * @param setting
         * @return
         */
        static std::shared_ptr<Covariance>
        CreateCovariance(const std::string &covariance_type, std::shared_ptr<Setting> setting);

        template<typename Derived>
        static bool
        Register(std::string covariance_type = "");

        [[nodiscard]] std::shared_ptr<Setting>
        GetSetting() const;

        [[nodiscard]] virtual std::pair<long, long>
        GetMinimumKtrainSize(long num_samples, long num_samples_with_gradient, long num_gradient_dimensions) const;

        [[nodiscard]] virtual std::pair<long, long>
        GetMinimumKtestSize(
            long num_train_samples,
            long num_train_samples_with_gradient,
            long num_gradient_dimensions,
            long num_test_queries,
            bool predict_gradient) const;

        [[nodiscard]] virtual std::pair<long, long>
        ComputeKtrain(const Eigen::Ref<const MatrixX> &mat_x, long num_samples, MatrixX &mat_k, VectorX &vec_alpha) const = 0;

        [[nodiscard]] virtual std::pair<long, long>
        ComputeKtrain(const Eigen::Ref<const MatrixX> &mat_x, const Eigen::Ref<const VectorX> &vec_var_y, long num_samples, MatrixX &mat_k, VectorX &vec_alpha)
            const = 0;

        [[nodiscard]] virtual std::pair<long, long>
        ComputeKtest(const Eigen::Ref<const MatrixX> &mat_x1, long num_samples1, const Eigen::Ref<const MatrixX> &mat_x2, long num_samples2, MatrixX &mat_k)
            const = 0;

        [[nodiscard]] virtual std::pair<long, long>
        ComputeKtrainWithGradient(const Eigen::Ref<const MatrixX> &mat_x, long num_samples, Eigen::VectorXl &vec_grad_flags, MatrixX &mat_k, VectorX &vec_alpha)
            const = 0;

        [[nodiscard]] virtual std::pair<long, long>
        ComputeKtrainWithGradient(
            const Eigen::Ref<const MatrixX> &mat_x,
            long num_samples,
            Eigen::VectorXl &vec_grad_flags,
            const Eigen::Ref<const VectorX> &vec_var_x,
            const Eigen::Ref<const VectorX> &vec_var_y,
            const Eigen::Ref<const VectorX> &vec_var_grad,
            MatrixX &mat_k,
            VectorX &vec_alpha) const = 0;

        /**
         * @brief compute kernel matrix between train samples and test queries with gradient.
         * @param mat_x1
         * @param num_samples1
         * @param vec_grad1_flags
         * @param mat_x2
         * @param num_samples2
         * @param predict_gradient whether to predict gradient
         * @param mat_k output kernel matrix
         * @return
         */
        [[nodiscard]] virtual std::pair<long, long>
        ComputeKtestWithGradient(
            const Eigen::Ref<const MatrixX> &mat_x1,
            long num_samples1,
            const Eigen::Ref<const Eigen::VectorXl> &vec_grad1_flags,
            const Eigen::Ref<const MatrixX> &mat_x2,
            long num_samples2,
            bool predict_gradient,
            MatrixX &mat_k) const = 0;

        [[nodiscard]] bool
        operator==(const Covariance &other) const;

        [[nodiscard]] bool
        operator!=(const Covariance &other) const;

        [[nodiscard]] virtual bool
        Write(const std::string &filename) const;

        [[nodiscard]] virtual bool
        Write(std::ostream &s) const;

        [[nodiscard]] virtual bool
        Read(const std::string &filename);

        [[nodiscard]] virtual bool
        Read(std::istream &s);

    protected:
        explicit Covariance(std::shared_ptr<Setting> setting);
    };

}  // namespace erl::covariance

#include "covariance.tpp"

template<>
struct YAML::convert<erl::covariance::Covariance<double>::Setting> : erl::covariance::Covariance<double>::Setting::YamlConvertImpl {};

template<>
struct YAML::convert<erl::covariance::Covariance<float>::Setting> : erl::covariance::Covariance<float>::Setting::YamlConvertImpl {};
