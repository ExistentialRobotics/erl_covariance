#pragma once

#include "erl_common/eigen.hpp"
#include "erl_common/exception.hpp"
#include "erl_common/logging.hpp"
#include "erl_common/yaml.hpp"

#include <functional>
#include <memory>

namespace erl::covariance {

    // ref1: https://peterroelants.github.io/posts/gaussian-process-kernels/
    // ref2: https://www.cs.toronto.edu/~duvenaud/cookbook/

    class Covariance {
    public:
        // structure for holding the parameters
        struct Setting : public common::Yamlable<Setting> {
            long x_dim = 2;           // dimension of input space
            double alpha = 1.;        // overall covariance magnitude
            double scale = 1.;        // scale length
            double scale_mix = 1.;    // used by RationalQuadratic, decreasing this value allows more local variations, inf --> Gaussian kernel
            Eigen::VectorXd weights;  // used by some custom kernels
        };

        inline static const volatile bool kSettingRegistered = common::YamlableBase::Register<Setting>();

    protected:
        inline static std::map<std::string, std::function<std::shared_ptr<Covariance>(std::shared_ptr<Setting>)>> s_class_id_mapping_ = {};
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
        GetMemoryUsage() const {
            std::size_t memory_usage = sizeof(*this);
            if (m_setting_ != nullptr) { memory_usage += sizeof(Setting); }
            return memory_usage;
        }

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
        static std::enable_if_t<std::is_base_of_v<Covariance, Derived>, bool>
        Register(std::string covariance_type = "") {
            if (covariance_type.empty()) { covariance_type = demangle(typeid(Derived).name()); }
            if (s_class_id_mapping_.find(covariance_type) != s_class_id_mapping_.end()) {
                ERL_WARN("{} is already registered.", covariance_type);
                return false;
            }

            s_class_id_mapping_[covariance_type] = [](std::shared_ptr<Setting> setting) {
                auto covariance_setting = std::dynamic_pointer_cast<typename Derived::Setting>(setting);
                if (setting == nullptr) { covariance_setting = std::make_shared<typename Derived::Setting>(); }
                ERL_ASSERTM(covariance_setting != nullptr, "setting is nullptr.");
                return std::make_shared<Derived>(covariance_setting);
            };
            ERL_DEBUG("{} is registered.", covariance_type);
            return true;
        }

        [[nodiscard]] std::shared_ptr<Setting>
        GetSetting() const {
            return m_setting_;
        }

        [[nodiscard]] virtual std::pair<long, long>
        GetMinimumKtrainSize(const long num_samples, const long num_samples_with_gradient, const long num_gradient_dimensions) const {
            long n = num_samples + num_samples_with_gradient * num_gradient_dimensions;
            return {n, n};
        }

        [[nodiscard]] virtual std::pair<long, long>
        GetMinimumKtestSize(
            const long num_train_samples,
            const long num_train_samples_with_gradient,
            const long num_gradient_dimensions,
            const long num_test_queries,
            const bool predict_gradient) const {
            return {
                num_train_samples + num_train_samples_with_gradient * num_gradient_dimensions,
                predict_gradient ? num_test_queries * (1 + num_gradient_dimensions) : num_test_queries};
        }

        [[nodiscard]] virtual std::pair<long, long>
        ComputeKtrain(const Eigen::Ref<const Eigen::MatrixXd> &mat_x, long num_samples, Eigen::MatrixXd &mat_k, Eigen::VectorXd &vec_alpha) const = 0;

        [[nodiscard]] virtual std::pair<long, long>
        ComputeKtrain(
            const Eigen::Ref<const Eigen::MatrixXd> &mat_x,
            const Eigen::Ref<const Eigen::VectorXd> &vec_var_y,
            long num_samples,
            Eigen::MatrixXd &mat_k,
            Eigen::VectorXd &vec_alpha) const = 0;

        [[nodiscard]] virtual std::pair<long, long>
        ComputeKtest(
            const Eigen::Ref<const Eigen::MatrixXd> &mat_x1,
            long num_samples1,
            const Eigen::Ref<const Eigen::MatrixXd> &mat_x2,
            long num_samples2,
            Eigen::MatrixXd &mat_k) const = 0;

        [[nodiscard]] virtual std::pair<long, long>
        ComputeKtrainWithGradient(
            const Eigen::Ref<const Eigen::MatrixXd> &mat_x,
            long num_samples,
            Eigen::VectorXl &vec_grad_flags,
            Eigen::MatrixXd &mat_k,
            Eigen::VectorXd &vec_alpha) const = 0;

        [[nodiscard]] virtual std::pair<long, long>
        ComputeKtrainWithGradient(
            const Eigen::Ref<const Eigen::MatrixXd> &mat_x,
            long num_samples,
            Eigen::VectorXl &vec_grad_flags,
            const Eigen::Ref<const Eigen::VectorXd> &vec_var_x,
            const Eigen::Ref<const Eigen::VectorXd> &vec_var_y,
            const Eigen::Ref<const Eigen::VectorXd> &vec_var_grad,
            Eigen::MatrixXd &mat_k,
            Eigen::VectorXd &vec_alpha) const = 0;

        /**
         * @brief compute kernel matrix between train samples and test queries with gradient.
         * @param mat_x1
         * @param num_samples1
         * @param vec_grad1_flags
         * @param mat_x2
         * @param num_samples2
         * @param mat_k output kernel matrix
         * @return
         */
        [[nodiscard]] virtual std::pair<long, long>
        ComputeKtestWithGradient(
            const Eigen::Ref<const Eigen::MatrixXd> &mat_x1,
            long num_samples1,
            const Eigen::Ref<const Eigen::VectorXl> &vec_grad1_flags,
            const Eigen::Ref<const Eigen::MatrixXd> &mat_x2,
            long num_samples2,
            bool predict_gradient,
            Eigen::MatrixXd &mat_k) const = 0;

        [[nodiscard]] bool
        operator==(const Covariance &other) const;

        [[nodiscard]] bool
        operator!=(const Covariance &other) const {
            return !(*this == other);
        }

        [[nodiscard]] virtual bool
        Write(const std::string &filename) const;

        [[nodiscard]] virtual bool
        Write(std::ostream &s) const;

        [[nodiscard]] virtual bool
        Read(const std::string &filename);

        [[nodiscard]] virtual bool
        Read(std::istream &s);

    protected:
        explicit Covariance(std::shared_ptr<Setting> setting)
            : m_setting_(std::move(setting)) {}
    };

#define ERL_REGISTER_COVARIANCE(Derived) inline const volatile bool kRegistered##Derived = erl::covariance::Covariance::Register<Derived>()
}  // namespace erl::covariance

template<>
struct YAML::convert<erl::covariance::Covariance::Setting> {
    static Node
    encode(const erl::covariance::Covariance::Setting &setting) {
        Node node(NodeType::Map);
        node["x_dim"] = setting.x_dim;
        node["alpha"] = setting.alpha;
        node["scale"] = setting.scale;
        node["scale_mix"] = setting.scale_mix;
        node["weights"] = setting.weights;
        return node;
    }

    static bool
    decode(const Node &node, erl::covariance::Covariance::Setting &setting) {
        if (!node.IsMap()) { return false; }
        setting.x_dim = node["x_dim"].as<int>();
        setting.alpha = node["alpha"].as<double>();
        setting.scale = node["scale"].as<double>();
        setting.scale_mix = node["scale_mix"].as<double>();
        setting.weights = node["weights"].as<Eigen::VectorXd>();
        return true;
    }
};
