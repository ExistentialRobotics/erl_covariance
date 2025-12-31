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
        using SparseMatrix = Eigen::SparseMatrix<Dtype>;
        using VectorX = Eigen::VectorX<Dtype>;

        // structure for holding the parameters
        struct Setting : public common::Yamlable<Setting> {
            long x_dim = 2;      // dimension of input space
            Dtype scale = 1.0f;  // scale length

            // used by RationalQuadratic, decreasing this value allows more local variations,
            // inf --> Gaussian kernel
            Dtype scale_mix = 1.0f;

            ERL_REFLECT_SCHEMA(
                Setting,
                ERL_REFLECT_MEMBER(Setting, x_dim),
                ERL_REFLECT_MEMBER(Setting, scale),
                ERL_REFLECT_MEMBER(Setting, scale_mix));
        };

        using Factory = common::FactoryPattern<Covariance, false, false, std::shared_ptr<Setting>>;

    protected:
        std::shared_ptr<Setting> m_setting_ = nullptr;

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
         * returns the actual class name as string for identification
         * @return The type of the tree.
         */
        [[nodiscard]] virtual std::string
        GetCovarianceType() const = 0;

        /**
         * returns the human-readable name of the covariance function
         * @return The name of the covariance function.
         */
        [[nodiscard]] virtual std::string
        GetCovarianceName() const = 0;

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
        Register(std::string covariance_type = "") {
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

        [[nodiscard]] std::shared_ptr<Setting>
        GetSetting() const;

        [[nodiscard]] virtual std::pair<long, long>
        GetMinimumKtrainSize(
            long num_samples,
            long num_samples_with_gradient,
            long num_gradient_dimensions) const;

        [[nodiscard]] virtual std::pair<long, long>
        GetMinimumKtestSize(
            long num_train_samples,
            long num_train_samples_with_gradient,
            long num_gradient_dimensions,
            long num_test_queries,
            bool predict_gradient) const;

        /**
         * The scale factor s when computing the Hessian matrix with x=x'. When x=x', the Hessian
         * matrix is s * I, where I is the identity matrix. This function is useful when computing
         * the variance of gradient predictions.
         * @return The scale factor s.
         */
        [[nodiscard]] virtual Dtype
        GetHessianScaleFactor() const = 0;

        /**
         * @brief Compute the training kernel matrix. For ordinary covariance functions, calling
         * another overload without mat_alpha is sufficient.
         * @param mat_x The input data matrix, size: (x_dim x num_samples).
         * @param num_samples Number of samples.
         * @param mat_k Output kernel matrix.
         * @param mat_alpha Input-output alpha matrix, size: (num_samples x y_dim) or (E, y_dim) if
         * using any dimension reduction. Input should be mat_y, where each column is a function
         * output dimension. Output will be overwritten if necessary.
         * @return Shape of mat_k: (num_samples, num_samples) or (E, E) if using any dimension
         * reduction.
         */
        [[nodiscard]] virtual std::pair<long, long>
        ComputeKtrain(
            const Eigen::Ref<const MatrixX> &mat_x,
            long num_samples,
            MatrixX &mat_k,
            MatrixX &mat_alpha) = 0;

        /**
         * @brief Compute the training kernel matrix.
         * @param mat_x the input data matrix, size: (x_dim x num_samples).
         * @param num_samples Number of samples.
         * @param mat_k Output kernel matrix.
         * @return Shape of mat_k: (num_samples, num_samples) or (E, E) if using any dimension
         * reduction.
         */
        [[nodiscard]] std::pair<long, long>
        ComputeKtrain(const Eigen::Ref<const MatrixX> &mat_x, long num_samples, MatrixX &mat_k);

        /**
         * @brief Compute the training kernel matrix with a bias term added to the pairwise scaled
         * distances: exp(d') = exp(d + exp_bias) = exp(d) * exp(exp_bias).
         * This is useful when the kernel scale is very small, causing the kernel values to be
         * extremely close to zero. By adding a bias term to the distances before exponentiation,
         * we can increase the kernel values to a more reasonable range. This bias term should be
         * then considered when interpreting the results. For ordinary covariance functions,
         * calling another overload without mat_alpha is sufficient.
         * @param mat_x The input data matrix, size: (x_dim x num_samples).
         * @param num_samples Number of samples.
         * @param exp_bias The bias term added to the pairwise scaled distances before exp.
         * @param mat_k Output kernel matrix.
         * @param mat_alpha Input-output alpha matrix, size: (num_samples x y_dim) or (E, y_dim) if
         * using any dimension reduction. Input should be mat_y, where each column is a function
         * output dimension. Output will be overwritten if necessary.
         * @return Shape of mat_k: (num_samples, num_samples) or (E, E) if using any dimension
         * reduction.
         */
        [[nodiscard]] virtual std::pair<long, long>
        ComputeKtrain(
            const Eigen::Ref<const MatrixX> &mat_x,
            long num_samples,
            Dtype exp_bias,
            MatrixX &mat_k,
            MatrixX &mat_alpha);

        /**
         * @brief Compute the training kernel matrix with a bias term added to the pairwise scaled
         * distances: exp(d') = exp(d + exp_bias) = exp(d) * exp(exp_bias).
         * This is useful when the kernel scale is very small, causing the kernel values to be
         * extremely close to zero. By adding a bias term to the distances before exponentiation,
         * we can increase the kernel values to a more reasonable range. This bias term should be
         * then considered when interpreting the results.
         * @param mat_x The input data matrix, size: (x_dim x num_samples).
         * @param num_samples Number of samples.
         * @param exp_bias The bias term added to the pairwise scaled distances before exp.
         * @param mat_k Output kernel matrix.
         * @return Shape of mat_k: (num_samples, num_samples) or (E, E) if using any dimension
         * reduction.
         */
        [[nodiscard]] std::pair<long, long>
        ComputeKtrain(
            const Eigen::Ref<const MatrixX> &mat_x,
            long num_samples,
            Dtype exp_bias,
            MatrixX &mat_k);

        /**
         * @brief Compute the training kernel matrix with variable noise levels. For ordinary
         * covariance functions, calling another overload without mat_alpha is sufficient.
         * @param mat_x The input data matrix, size: (x_dim x num_samples).
         * @param vec_var_y The vector of noise variances for each sample, size: num_samples.
         * @param num_samples Number of samples.
         * @param mat_k Output kernel matrix, size: (num_samples x num_samples) or (E, E) if using
         * any dimension reduction.
         * @param mat_alpha Output alpha matrix, size: (num_samples x y_dim).
         * @param mat_k Output kernel matrix.
         * @param mat_alpha Input-output alpha matrix, size: (num_samples x y_dim) or (E, y_dim) if
         * using any dimension reduction. Input should be mat_y, where each column is a function
         * output dimension. Output will be overwritten if necessary.
         * @return Shape of mat_k: (num_samples, num_samples) or (E, E) if using any dimension
         * reduction. Shape of mat_alpha: if using any dimension reduction, (E, y_dim), else
         * mat_alpha is untouched.
         */
        [[nodiscard]] virtual std::pair<long, long>
        ComputeKtrain(
            const Eigen::Ref<const MatrixX> &mat_x,
            const Eigen::Ref<const VectorX> &vec_var_y,
            long num_samples,
            MatrixX &mat_k,
            MatrixX &mat_alpha) = 0;

        /**
         * @brief Compute the training kernel matrix with variable noise levels.
         * @param mat_x The input data matrix, size: x_dim x num_samples.
         * @param vec_var_y The vector of noise variances for each sample, size: num_samples.
         * @param num_samples Number of samples.
         * @param mat_k Output kernel matrix.
         * @return Shape of mat_k: (num_samples, num_samples) or (E, E) if using any dimension
         * reduction.
         */
        [[nodiscard]] std::pair<long, long>
        ComputeKtrain(
            const Eigen::Ref<const MatrixX> &mat_x,
            const Eigen::Ref<const VectorX> &vec_var_y,
            long num_samples,
            MatrixX &mat_k);

        /**
         * @brief Compute the training kernel matrix with variable noise levels and a bias term
         * added to the pairwise scaled distances: exp(d') = exp(d + exp_bias) = exp(d) *
         * exp(exp_bias). This is useful when the kernel scale is very small, causing the kernel
         * values to be extremely close to zero. By adding a bias term to the distances before
         * exponentiation, we can increase the kernel values to a more reasonable range. This bias
         * term should be then considered when interpreting the results. For ordinary covariance
         * functions, calling another overload without mat_alpha is sufficient.
         * @param mat_x The input data matrix, size: x_dim x num_samples.
         * @param vec_var_y The vector of noise variances for each sample, size: num_samples.
         * @param num_samples Number of samples.
         * @param exp_bias The bias term added to the pairwise scaled distances before exp.
         * @param mat_k Output kernel matrix.
         * @param mat_alpha Input-output alpha matrix, size: (num_samples x y_dim) or (E, y_dim) if
         * using any dimension reduction. Input should be mat_y, where each column is a function
         * output dimension. Output will be overwritten if necessary.
         * @return Shape of mat_k: (num_samples, num_samples) or (E, E) if using any dimension
         * reduction. Shape of mat_alpha: if using any dimension reduction, (E, y_dim), else
         * mat_alpha is untouched.
         */
        [[nodiscard]] virtual std::pair<long, long>
        ComputeKtrain(
            const Eigen::Ref<const MatrixX> &mat_x,
            const Eigen::Ref<const VectorX> &vec_var_y,
            long num_samples,
            Dtype exp_bias,
            MatrixX &mat_k,
            MatrixX &mat_alpha);

        /**
         * @brief Compute the test kernel matrix between training samples and test queries.
         * @param mat_x1 The input data matrix for training samples, size: (x_dim x num_samples).
         * @param num_samples Number of training samples.
         * @param mat_x2 The input data matrix for test queries, size: (x_dim x num_queries).
         * @param num_queries Number of test queries.
         * @param mat_k Output kernel matrix.
         * @return Shape of mat_k: (num_samples, num_queries) or (E, num_queries) if using any
         * dimension reduction.
         */
        [[nodiscard]] virtual std::pair<long, long>
        ComputeKtest(
            const Eigen::Ref<const MatrixX> &mat_x1,
            long num_samples,
            const Eigen::Ref<const MatrixX> &mat_x2,
            long num_queries,
            MatrixX &mat_k) const = 0;

        /**
         * @brief Compute the sparse test kernel matrix.
         * @param mat_x1 The input data matrix for training samples, size: (x_dim x num_samples).
         * @param num_samples Number of training samples.
         * @param mat_x2 The input data matrix for test queries, size: (x_dim x num_queries).
         * @param num_queries Number of test queries.
         * @param zero_threshold Threshold below which kernel values are considered zero.
         * @param mat_k Output kernel matrix.
         * @return Shape of mat_k: (num_samples, num_queries) or (E, num_queries) if using any
         * dimension reduction.
         */
        [[nodiscard]] virtual std::pair<long, long>
        ComputeKtestSparse(
            const Eigen::Ref<const MatrixX> &mat_x1,
            long num_samples,
            const Eigen::Ref<const MatrixX> &mat_x2,
            long num_queries,
            Dtype zero_threshold,
            SparseMatrix &mat_k) const;

        /**
         * @brief Compute the test kernel matrix with a bias term added to the pairwise scaled
         * distances: exp(d') = exp(d + exp_bias) = exp(d) * exp(exp_bias). This is useful when the
         * kernel scale is very small, causing the kernel values to be extremely close to zero. By
         * adding a bias term to the distances before exponentiation, we can increase the kernel
         * values to a more reasonable range. This bias term should be then considered when
         * interpreting the results.
         * @param mat_x1 The input data matrix for training samples, size: (x_dim x num_samples).
         * @param num_samples Number of training samples.
         * @param mat_x2 The input data matrix for test queries, size: (x_dim x num_queries).
         * @param num_queries Number of test queries.
         * @param exp_bias The bias term added to the pairwise scaled distances.
         * @param mat_k Output kernel matrix.
         * @return Shape of mat_k: (num_samples, num_queries) or (E, num_queries) if using any
         * dimension reduction.
         */
        [[nodiscard]] virtual std::pair<long, long>
        ComputeKtest(
            const Eigen::Ref<const MatrixX> &mat_x1,
            long num_samples,
            const Eigen::Ref<const MatrixX> &mat_x2,
            long num_queries,
            Dtype exp_bias,
            MatrixX &mat_k) const;

        /**
         * @brief Compute the sparse test kernel matrix with a bias term added to the pairwise
         * scaled distances: exp(d') = exp(d + exp_bias) = exp(d) * exp(exp_bias). This is useful
         * when the kernel scale is very small, causing the kernel values to be extremely close to
         * zero. By adding a bias term to the distances before exponentiation, we can increase the
         * kernel values to a more reasonable range. This bias term should be then considered when
         * interpreting the results.
         * @param mat_x1 The input data matrix for training samples, size: (x_dim x num_samples).
         * @param num_samples Number of training samples.
         * @param mat_x2 The input data matrix for test queries, size: (x_dim x num_queries).
         * @param num_queries Number of test queries.
         * @param exp_bias The bias term added to the pairwise scaled distances.
         * @param zero_threshold Threshold below which kernel values are considered zero.
         * @param mat_k Output kernel matrix.
         * @return Shape of mat_k: (num_samples, num_queries) or (E, num_queries) if using any
         * dimension reduction.
         */
        [[nodiscard]] virtual std::pair<long, long>
        ComputeKtestSparse(
            const Eigen::Ref<const MatrixX> &mat_x1,
            long num_samples,
            const Eigen::Ref<const MatrixX> &mat_x2,
            long num_queries,
            Dtype exp_bias,
            Dtype zero_threshold,
            SparseMatrix &mat_k) const;

        /**
         * @brief compute kernel matrix between train samples with gradient.
         * @param mat_x The input data matrix, size: (x_dim x num_samples).
         * @param num_samples Number of samples.
         * @param vec_grad_flags The vector indicating which training samples have gradient
         * observations, size: num_samples. If an element is <=0, no gradient observation for that
         * sample; if >0, there is gradient observed for that sample. Positive elements will be
         * modified to the start index of df/dx in the kernel matrix.
         * @param mat_k Output kernel matrix.
         * @param mat_alpha Input-output alpha matrix, size: (num_samples x y_dim) or (E, y_dim) if
         * using any dimension reduction. Input should be mat_y, where each column is a function
         * output dimension. Output will be overwritten if necessary.
         * @return Shape of mat_k: (N, N) or (E, E) if using any dimension reduction. N =
         * num_samples + x_dim * num_samples_with_grad.
         */
        [[nodiscard]] virtual std::pair<long, long>
        ComputeKtrainWithGradient(
            const Eigen::Ref<const MatrixX> &mat_x,
            long num_samples,
            Eigen::VectorXl &vec_grad_flags,
            MatrixX &mat_k,
            MatrixX &mat_alpha) = 0;

        /**
         * @brief compute kernel matrix between train samples with gradient.
         * @param mat_x The input data matrix, size: (x_dim x num_samples).
         * @param num_samples Number of samples.
         * @param vec_grad_flags The vector indicating which training samples have gradient
         * observations, size: num_samples. If an element is <=0, no gradient observation for that
         * sample; if >0, there is gradient observed for that sample. Positive elements will be
         * modified to the start index of df/dx in the kernel matrix.
         * @param mat_k Output kernel matrix.
         * @return Shape of mat_k: (N, N) or (E, E) if using any dimension reduction. N =
         * num_samples + x_dim * num_samples_with_grad.
         */
        [[nodiscard]] std::pair<long, long>
        ComputeKtrainWithGradient(
            const Eigen::Ref<const MatrixX> &mat_x,
            long num_samples,
            Eigen::VectorXl &vec_grad_flags,
            MatrixX &mat_k);

        /**
         * @brief Compute kernel matrix between train samples with gradient. For ordinary covariance
         * functions, calling another overload without mat_alpha is sufficient. A bias term added to
         * the pairwise scaled distances: exp(d') = exp(d + exp_bias) = exp(d) * exp(exp_bias). This
         * is useful when the kernel scale is very small, causing the kernel values to be extremely
         * close to zero. By adding a bias term to the distances before exponentiation, we can
         * increase the kernel values to a more reasonable range. This bias term should be then
         * considered when interpreting the results.
         * @param mat_x The input data matrix, size: (x_dim x num_samples).
         * @param num_samples Number of samples.
         * @param exp_bias The bias term added to the pairwise scaled distances before exp.
         * @param vec_grad_flags The vector indicating which training samples have gradient
         * observations, size: num_samples. If an element is <=0, no gradient observation for that
         * sample; if >0, there is gradient observed for that sample. Positive elements will be
         * modified to the start index of df/dx in the kernel matrix.
         * @param mat_k Output kernel matrix.
         * @param mat_alpha Input-output alpha matrix, size: (num_samples x y_dim) or (E, y_dim) if
         * using any dimension reduction. Input should be mat_y, where each column is a function
         * output dimension. Output will be overwritten if necessary.
         * @return Shape of mat_k: (N, N) or (E, E) if using any dimension reduction. N =
         * num_samples + x_dim * num_samples_with_grad.
         */
        [[nodiscard]] virtual std::pair<long, long>
        ComputeKtrainWithGradient(
            const Eigen::Ref<const MatrixX> &mat_x,
            long num_samples,
            Dtype exp_bias,
            Eigen::VectorXl &vec_grad_flags,
            MatrixX &mat_k,
            MatrixX &mat_alpha);

        /**
         * @brief Compute kernel matrix between train samples with gradient. A bias term added to
         * the pairwise scaled distances: exp(d') = exp(d + exp_bias) = exp(d) * exp(exp_bias). This
         * is useful when the kernel scale is very small, causing the kernel values to be extremely
         * close to zero. By adding a bias term to the distances before exponentiation, we can
         * increase the kernel values to a more reasonable range. This bias term should be then
         * considered when interpreting the results.
         * @param mat_x The input data matrix, size: (x_dim x num_samples).
         * @param num_samples Number of samples.
         * @param exp_bias The bias term added to the pairwise scaled distances before exp.
         * @param vec_grad_flags The vector indicating which training samples have gradient
         * observations, size: num_samples. If an element is <=0, no gradient observation for that
         * sample; if >0, there is gradient observed for that sample. Positive elements will be
         * modified to the start index of df/dx in the kernel matrix.
         * @param mat_k Output kernel matrix.
         * @return Shape of mat_k: (N, N) or (E, E) if using any dimension reduction. N =
         * num_samples + x_dim * num_samples_with_grad.
         */
        [[nodiscard]] std::pair<long, long>
        ComputeKtrainWithGradient(
            const Eigen::Ref<const MatrixX> &mat_x,
            long num_samples,
            Dtype exp_bias,
            Eigen::VectorXl &vec_grad_flags,
            MatrixX &mat_k);

        /**
         * @brief compute kernel matrix between train samples with gradient and variable noise
         * levels. For ordinary covariance functions, calling another overload without mat_alpha is
         * sufficient.
         * @param mat_x The input data matrix, size: (x_dim x num_samples).
         * @param num_samples Number of samples.
         * @param vec_grad_flags The vector indicating which training samples have gradient
         * observations, size: num_samples. If an element is <=0, no gradient observation for that
         * sample; if >0, there is gradient observed for that sample. Positive elements will be
         * modified to the start index of df/dx in the kernel matrix.
         * @param vec_var_x The vector of input noise variances for each sample, size: num_samples.
         * @param vec_var_y The vector of output noise variances for each sample, size: num_samples.
         * @param vec_var_grad The vector of gradient noise variances for each sample, size:
        num_samples.
         * @param mat_k Output kernel matrix.
         * @param mat_alpha Input-output alpha matrix, size: (num_samples x y_dim) or (E, y_dim) if
         * using any dimension reduction. Input should be mat_y, where each column is a function
         * output dimension. Output will be overwritten if necessary.
         * @return Shape of mat_k: (N, N) or (E, E) if using any dimension reduction. N =
         * num_samples + x_dim * num_samples_with_grad.
         */
        [[nodiscard]] virtual std::pair<long, long>
        ComputeKtrainWithGradient(
            const Eigen::Ref<const MatrixX> &mat_x,
            long num_samples,
            Eigen::VectorXl &vec_grad_flags,
            const Eigen::Ref<const VectorX> &vec_var_x,
            const Eigen::Ref<const VectorX> &vec_var_y,
            const Eigen::Ref<const VectorX> &vec_var_grad,
            MatrixX &mat_k,
            MatrixX &mat_alpha) = 0;

        /**
         * @brief Compute kernel matrix between train samples with gradient and variable noise
         * levels.
         * @param mat_x The input data matrix, size: (x_dim x num_samples).
         * @param num_samples Number of samples.
         * @param vec_grad_flags The vector indicating which training samples have gradient
         * observations, size: num_samples. If an element is <=0, no gradient observation for that
         * sample; if >0, there is gradient observed for that sample. Positive elements will be
         * modified to the start index of df/dx in the kernel matrix.
         * @param vec_var_x The vector of input noise variances for each sample, size: num_samples.
         * @param vec_var_y The vector of output noise variances for each sample, size: num_samples.
         * @param vec_var_grad The vector of gradient noise variances for each sample, size:
        num_samples.
         * @param mat_k Output kernel matrix, size: (num_samples x num_samples) or (E, E) if using
         * any dimension reduction.
         * @return Shape of mat_k: (N, N) or (E, E) if using any dimension reduction. N =
         * num_samples + x_dim * num_samples_with_grad.
         */
        [[nodiscard]] std::pair<long, long>
        ComputeKtrainWithGradient(
            const Eigen::Ref<const MatrixX> &mat_x,
            long num_samples,
            Eigen::VectorXl &vec_grad_flags,
            const Eigen::Ref<const VectorX> &vec_var_x,
            const Eigen::Ref<const VectorX> &vec_var_y,
            const Eigen::Ref<const VectorX> &vec_var_grad,
            MatrixX &mat_k);

        /**
         * @brief compute kernel matrix between train samples with gradient and variable noise
         * levels. For ordinary covariance functions, calling another overload without mat_alpha is
         * sufficient. A bias term added to the pairwise scaled distances: exp(d') = exp(d +
         * exp_bias) = exp(d) * exp(exp_bias). This is useful when the kernel scale is very small,
         * causing the kernel values to be extremely close to zero. By adding a bias term to the
         * distances before exponentiation, we can increase the kernel values to a more reasonable
         * range. This bias term should be then considered when interpreting the results.
         * @param mat_x The input data matrix, size: (x_dim x num_samples).
         * @param num_samples Number of samples.
         * @param exp_bias The bias term added to the pairwise scaled distances before exp.
         * @param vec_grad_flags The vector indicating which training samples have gradient
         * observations, size: num_samples. If an element is <=0, no gradient observation for that
         * sample; if >0, there is gradient observed for that sample. Positive elements will be
         * modified to the start index of df/dx in the kernel matrix.
         * @param vec_var_x The vector of input noise variances for each sample, size: num_samples.
         * @param vec_var_y The vector of output noise variances for each sample, size: num_samples.
         * @param vec_var_grad The vector of gradient noise variances for each sample, size:
        num_samples.
         * @param mat_k Output kernel matrix, size: (num_samples x num_samples) or (E, E) if using
         * any dimension reduction.
         * @param mat_alpha Input-output alpha matrix, size: (num_samples x y_dim) or (E, y_dim) if
         * using any dimension reduction. Input should be mat_y, where each column is a function
         * output dimension. Output will be overwritten if necessary.
         * @return Shape of mat_k: (N, N) or (E, E) if using any dimension reduction. N =
         * num_samples + x_dim * num_samples_with_grad.
         */
        [[nodiscard]] virtual std::pair<long, long>
        ComputeKtrainWithGradient(
            const Eigen::Ref<const MatrixX> &mat_x,
            long num_samples,
            Dtype exp_bias,
            Eigen::VectorXl &vec_grad_flags,
            const Eigen::Ref<const VectorX> &vec_var_x,
            const Eigen::Ref<const VectorX> &vec_var_y,
            const Eigen::Ref<const VectorX> &vec_var_grad,
            MatrixX &mat_k,
            MatrixX &mat_alpha);

        /**
         * @brief Compute kernel matrix between train samples with gradient and variable noise
         * levels. A bias term added to the pairwise scaled distances: exp(d') = exp(d + exp_bias) =
         * exp(d) * exp(exp_bias). This is useful when the kernel scale is very small, causing the
         * kernel values to be extremely close to zero. By adding a bias term to the distances
         * before exponentiation, we can increase the kernel values to a more reasonable range. This
         * bias term should be then considered when interpreting the results.
         * @param mat_x The input data matrix, size: (x_dim x num_samples).
         * @param num_samples Number of samples.
         * @param exp_bias The bias term added to the pairwise scaled distances before exp.
         * @param vec_grad_flags The vector indicating which training samples have gradient
         * observations, size: num_samples. If an element is <=0, no gradient observation for that
         * sample; if >0, there is gradient observed for that sample. Positive elements will be
         * modified to the start index of df/dx in the kernel matrix.
         * @param vec_var_x The vector of input noise variances for each sample, size: num_samples.
         * @param vec_var_y The vector of output noise variances for each sample, size: num_samples.
         * @param vec_var_grad The vector of gradient noise variances for each sample, size:
         * num_samples.
         * @param mat_k Output kernel matrix, size: (num_samples x num_samples) or (E, E) if using
         * any dimension reduction.
         * @return Shape of mat_k: (N, N) or (E, E) if using any dimension reduction. N =
         * num_samples + x_dim * num_samples_with_grad.
         */
        [[nodiscard]] std::pair<long, long>
        ComputeKtrainWithGradient(
            const Eigen::Ref<const MatrixX> &mat_x,
            long num_samples,
            Dtype exp_bias,
            Eigen::VectorXl &vec_grad_flags,
            const Eigen::Ref<const VectorX> &vec_var_x,
            const Eigen::Ref<const VectorX> &vec_var_y,
            const Eigen::Ref<const VectorX> &vec_var_grad,
            MatrixX &mat_k);

        /**
         * @brief Compute kernel matrix between training samples and test queries with gradient.
         * @param mat_x1 The input data matrix for training samples, size: (x_dim x num_samples).
         * @param num_samples Number of training samples.
         * @param vec_grad1_flags The vector indicating which training samples have gradient
         * observations, size: num_samples. If an element is <=0, no gradient observation for that
         * sample; if >0, the number of gradient dimensions observed for that sample.
         * @param mat_x2 The input data matrix for test queries, size: (x_dim x num_queries).
         * @param num_queries Number of test queries.
         * @param predict_gradient Whether to predict gradient.
         * @param mat_k Output kernel matrix.
         * @return Shape of mat_k: (N, num_queries) or (E, num_queries) if using any dimension
         * reduction. N = num_samples + x_dim * num_samples_with_grad.
         */
        [[nodiscard]] virtual std::pair<long, long>
        ComputeKtestWithGradient(
            const Eigen::Ref<const MatrixX> &mat_x1,
            long num_samples,
            const Eigen::Ref<const Eigen::VectorXl> &vec_grad1_flags,
            const Eigen::Ref<const MatrixX> &mat_x2,
            long num_queries,
            bool predict_gradient,
            MatrixX &mat_k) const = 0;

        /**
         * @brief Compute sparse kernel matrix between training samples and test queries with
         * gradient.
         * @param mat_x1 The input data matrix for training samples, size: (x_dim x num_samples).
         * @param num_samples Number of training samples.
         * @param vec_grad1_flags The vector indicating which training samples have gradient
         * observations, size: num_samples. If an element is <=0, no gradient observation for that
         * sample; if >0, there is gradient observed for that sample.
         * @param mat_x2 The input data matrix for test queries, size: (x_dim x num_queries).
         * @param num_queries Number of test queries.
         * @param predict_gradient Whether to predict gradient.
         * @param zero_threshold Threshold below which values are considered zero.
         * @param mat_k Output kernel matrix.
         * @return Shape of mat_k: (N, num_queries) or (E, num_queries) if using any dimension
         * reduction. N = num_samples + x_dim * num_samples_with_grad.
         */
        [[nodiscard]] virtual std::pair<long, long>
        ComputeKtestWithGradientSparse(
            const Eigen::Ref<const MatrixX> &mat_x1,
            long num_samples,
            const Eigen::Ref<const Eigen::VectorXl> &vec_grad1_flags,
            const Eigen::Ref<const MatrixX> &mat_x2,
            long num_queries,
            bool predict_gradient,
            Dtype zero_threshold,
            SparseMatrix &mat_k) const;

        /**
         * @brief Compute kernel matrix between training samples and test queries with gradient.
         * A bias term added to the pairwise scaled distances: exp(d') = exp(d + exp_bias) = exp(d)
         * * exp(exp_bias). This is useful when the kernel scale is very small, causing the kernel
         * values to be extremely close to zero. By adding a bias term to the distances before
         * exponentiation, we can increase the kernel values to a more reasonable range. This bias
         * term should be then considered when interpreting the results.
         * @param mat_x1 The input data matrix for training samples, size: (x_dim x num_samples).
         * @param num_samples Number of training samples.
         * @param vec_grad1_flags The vector indicating which training samples have gradient
         * observations, size: num_samples. If an element is <=0, no gradient observation for that
         * sample; if >0, there is gradient observed for that sample.
         * @param mat_x2 The input data matrix for test queries, size: (x_dim x num_queries).
         * @param num_queries Number of test queries.
         * @param predict_gradient Whether to predict gradient.
         * @param exp_bias The bias term added to the pairwise scaled distances before exp.
         * @param mat_k Output kernel matrix.
         * @return Shape of mat_k: (N, num_queries) or (E, num_queries) if using any dimension
         * reduction. N = num_samples + x_dim * num_samples_with_grad.
         */
        [[nodiscard]] virtual std::pair<long, long>
        ComputeKtestWithGradient(
            const Eigen::Ref<const MatrixX> &mat_x1,
            long num_samples,
            const Eigen::Ref<const Eigen::VectorXl> &vec_grad1_flags,
            const Eigen::Ref<const MatrixX> &mat_x2,
            long num_queries,
            bool predict_gradient,
            Dtype exp_bias,
            MatrixX &mat_k) const;

        /**
         * @brief Compute sparse kernel matrix between training samples and test queries with
         * gradient. A bias term added to the pairwise scaled distances: exp(d') = exp(d + exp_bias)
         * = exp(d) * exp(exp_bias). This is useful when the kernel scale is very small, causing the
         * kernel values to be extremely close to zero. By adding a bias term to the distances
         * before exponentiation, we can increase the kernel values to a more reasonable range. This
         * bias term should be then considered when interpreting the results.
         * @param mat_x1 The input data matrix for training samples, size: (x_dim x num_samples).
         * @param num_samples Number of training samples.
         * @param vec_grad1_flags The vector indicating which training samples have gradient
         * observations, size: num_samples. If an element is <=0, no gradient observation for that
         * sample; if >0, there is gradient observed for that sample.
         * @param mat_x2 The input data matrix for test queries, size: (x_dim x num_queries).
         * @param num_queries Number of test queries.
         * @param predict_gradient Whether to predict gradient.
         * @param exp_bias The bias term added to the pairwise scaled distances before exp.
         * @param zero_threshold Threshold below which values are considered zero.
         * @param mat_k Output kernel matrix.
         * @param mat_k Output kernel matrix.
         * @return Shape of mat_k: (N, num_queries) or (E, num_queries) if using any dimension
         * reduction. N = num_samples + x_dim * num_samples_with_grad.
         */
        [[nodiscard]] virtual std::pair<long, long>
        ComputeKtestWithGradientSparse(
            const Eigen::Ref<const MatrixX> &mat_x1,
            long num_samples,
            const Eigen::Ref<const Eigen::VectorXl> &vec_grad1_flags,
            const Eigen::Ref<const MatrixX> &mat_x2,
            long num_queries,
            bool predict_gradient,
            Dtype exp_bias,
            Dtype zero_threshold,
            SparseMatrix &mat_k) const;

        [[nodiscard]] bool
        operator==(const Covariance &other) const;

        [[nodiscard]] bool
        operator!=(const Covariance &other) const;

        [[nodiscard]] virtual bool
        Write(std::ostream &s) const;

        [[nodiscard]] virtual bool
        Read(std::istream &s);

    protected:
        explicit Covariance(std::shared_ptr<Setting> setting);
    };

    extern template class Covariance<double>;
    extern template class Covariance<float>;
}  // namespace erl::covariance
