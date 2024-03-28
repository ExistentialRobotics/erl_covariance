#pragma once

#include <memory>

#include "erl_common/eigen.hpp"
#include "erl_common/assert.hpp"
#include "erl_common/yaml.hpp"
#include "erl_common/exception.hpp"

namespace erl::covariance {

    // ref1: https://peterroelants.github.io/posts/gaussian-process-kernels/
    // ref2: https://www.cs.toronto.edu/~duvenaud/cookbook/

    class Covariance {

    public:
        enum class Type {
            kOrnsteinUhlenbeck = 0,
            kMatern32 = 1,
            kRadialBiasFunction = 2,  // Gaussian kernel, exponential quadratic
            kRationalQuadratic = 3,
            kCustomKernelV1 = 4,
            kCustomKernelV2 = 5,
            kCustomKernelV3 = 6,
            kCustomKernelV4 = 7,
            kUnknown
        };

        // structure for holding the parameters
        struct Setting : public common::Yamlable<Setting> {
            Type type = Type::kUnknown;
            long x_dim = 2;           // dimension of input space
            double alpha = 1.;        // overall covariance magnitude
            double scale = 1.;        // m_scale_ length
            double scale_mix = 1.;    // used by RationalQuadratic, decreasing this value allows more local variations, inf --> Gaussian kernel
            Eigen::VectorXd weights;  // used by some custom kernels

            Setting() = default;

            explicit Setting(Type t)
                : type(t) {}
        };

    protected:
        std::shared_ptr<Setting> m_setting_ = nullptr;

    public:
        static inline const char *
        GetTypeName(const Type &type) {
            static const char *names[int(Type::kUnknown) + 1] = {
                ERL_AS_STRING(kOrnsteinUhlenbeck),
                ERL_AS_STRING(kMatern32),
                ERL_AS_STRING(kRadialBiasFunction),
                ERL_AS_STRING(kRationalQuadratic),
                ERL_AS_STRING(kCustomKernelV1),
                ERL_AS_STRING(kCustomKernelV2),
                ERL_AS_STRING(kCustomKernelV3),
                ERL_AS_STRING(kCustomKernelV4),
                ERL_AS_STRING(kUnknown),
            };
            return names[static_cast<int>(type)];
        }

        static inline Type
        GetTypeFromName(const std::string &type_name) {
            if (type_name == ERL_AS_STRING(kOrnsteinUhlenbeck)) { return Type::kOrnsteinUhlenbeck; }
            if (type_name == ERL_AS_STRING(kMatern32)) { return Type::kMatern32; }
            if (type_name == ERL_AS_STRING(kRadialBiasFunction)) { return Type::kRadialBiasFunction; }
            if (type_name == ERL_AS_STRING(kRationalQuadratic)) { return Type::kRationalQuadratic; }
            if (type_name == ERL_AS_STRING(kCustomKernelV1)) { return Type::kCustomKernelV1; }
            if (type_name == ERL_AS_STRING(kCustomKernelV2)) { return Type::kCustomKernelV2; }
            if (type_name == ERL_AS_STRING(kCustomKernelV3)) { return Type::kCustomKernelV3; }
            if (type_name == ERL_AS_STRING(kCustomKernelV4)) { return Type::kCustomKernelV4; }
            return Type::kUnknown;
        }

        [[nodiscard]] std::shared_ptr<Setting>
        GetSetting() const {
            return m_setting_;
        }

        [[nodiscard]] static inline std::pair<long, long>
        GetMinimumKtrainSize(long num_samples, long num_samples_with_gradient, long num_gradient_dimensions) {
            long n = num_samples + num_samples_with_gradient * num_gradient_dimensions;
            return {n, n};
        }

        [[nodiscard]] static inline std::pair<long, long>
        GetMinimumKtestSize(long num_train_samples, long num_train_samples_with_gradient, long num_gradient_dimensions, long num_test_queries) {
            return {num_train_samples + num_train_samples_with_gradient * num_gradient_dimensions, num_test_queries * (1 + num_gradient_dimensions)};
        }

        [[nodiscard]] virtual std::pair<long, long>
        ComputeKtrain(Eigen::Ref<Eigen::MatrixXd> k_mat, const Eigen::Ref<const Eigen::MatrixXd> &mat_x) const = 0;

        [[nodiscard]] virtual std::pair<long, long>
        ComputeKtrain(Eigen::Ref<Eigen::MatrixXd> k_mat, const Eigen::Ref<const Eigen::MatrixXd> &mat_x, const Eigen::Ref<const Eigen::VectorXd> &vec_var_y)
            const = 0;

        [[nodiscard]] virtual std::pair<long, long>
        ComputeKtest(Eigen::Ref<Eigen::MatrixXd> k_mat, const Eigen::Ref<const Eigen::MatrixXd> &mat_x1, const Eigen::Ref<const Eigen::MatrixXd> &mat_x2)
            const = 0;

        [[nodiscard]] virtual std::pair<long, long>
        ComputeKtrainWithGradient(
            Eigen::Ref<Eigen::MatrixXd> k_mat,
            const Eigen::Ref<const Eigen::MatrixXd> &mat_x,
            const Eigen::Ref<const Eigen::VectorXb> &vec_grad_flags) const = 0;

        [[nodiscard]] virtual std::pair<long, long>
        ComputeKtrainWithGradient(
            Eigen::Ref<Eigen::MatrixXd> k_mat,
            const Eigen::Ref<const Eigen::MatrixXd> &mat_x,
            const Eigen::Ref<const Eigen::VectorXb> &vec_grad_flags,
            const Eigen::Ref<const Eigen::VectorXd> &vec_var_x,
            const Eigen::Ref<const Eigen::VectorXd> &vec_var_y,
            const Eigen::Ref<const Eigen::VectorXd> &vec_var_grad) const = 0;

        /**
         * @brief compute kernel matrix between train samples and test queries with gradient.
         * @param k_mat
         * @param dim number of gradient dimensions, i.e. rows of mat_x1. Pass constant to enable compile-time optimization.
         * @param mat_x1
         * @param vec_grad1_flags
         * @param mat_x2
         * @return
         */
        [[nodiscard]] virtual std::pair<long, long>
        ComputeKtestWithGradient(
            Eigen::Ref<Eigen::MatrixXd> k_mat,
            const Eigen::Ref<const Eigen::MatrixXd> &mat_x1,
            const Eigen::Ref<const Eigen::VectorXb> &vec_grad1_flags,
            const Eigen::Ref<const Eigen::MatrixXd> &mat_x2) const = 0;

        static std::shared_ptr<Covariance>
        Create(const std::shared_ptr<Setting> &setting);

        virtual ~Covariance() = default;

    protected:
        explicit Covariance(std::shared_ptr<Setting> setting);
    };
}  // namespace erl::covariance

namespace YAML {

    template<>
    struct convert<erl::covariance::Covariance::Type> {
        inline static Node
        encode(const erl::covariance::Covariance::Type &type) {
            return Node(erl::covariance::Covariance::GetTypeName(type));
        }

        inline static bool
        decode(const Node &node, erl::covariance::Covariance::Type &type) {
            if (!node.IsScalar()) { return false; }
            type = erl::covariance::Covariance::GetTypeFromName(node.as<std::string>());
            return true;
        }
    };

    template<>
    struct convert<erl::covariance::Covariance::Setting> {
        inline static Node
        encode(const erl::covariance::Covariance::Setting &setting) {
            Node node(NodeType::Map);
            node["type"] = setting.type;
            node["x_dim"] = setting.x_dim;
            node["alpha"] = setting.alpha;
            node["scale"] = setting.scale;
            node["scale_mix"] = setting.scale_mix;
            node["weights"] = setting.weights;
            return node;
        }

        inline static bool
        decode(const Node &node, erl::covariance::Covariance::Setting &setting) {
            if (!node.IsMap()) { return false; }
            setting.type = node["type"].as<erl::covariance::Covariance::Type>();
            setting.x_dim = node["x_dim"].as<int>();
            setting.alpha = node["alpha"].as<double>();
            setting.scale = node["scale"].as<double>();
            setting.scale_mix = node["scale_mix"].as<double>();
            setting.weights = node["weights"].as<Eigen::VectorXd>();
            return true;
        }
    };
}  // namespace YAML
