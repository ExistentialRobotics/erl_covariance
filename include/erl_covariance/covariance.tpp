#pragma once

namespace erl::covariance {
    template<typename Dtype>
    YAML::Node
    Covariance<Dtype>::Setting::YamlConvertImpl::encode(const Setting &setting) {
        YAML::Node node(YAML::NodeType::Map);
        node["x_dim"] = setting.x_dim;
        node["alpha"] = setting.alpha;
        node["scale"] = setting.scale;
        node["scale_mix"] = setting.scale_mix;
        node["weights"] = setting.weights;
        return node;
    }

    template<typename Dtype>
    bool
    Covariance<Dtype>::Setting::YamlConvertImpl::decode(const YAML::Node &node, Setting &setting) {
        if (!node.IsMap()) { return false; }
        setting.x_dim = node["x_dim"].as<int>();
        setting.alpha = node["alpha"].as<Dtype>();
        setting.scale = node["scale"].as<Dtype>();
        setting.scale_mix = node["scale_mix"].as<Dtype>();
        setting.weights = node["weights"].as<VectorX>();
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
    Covariance<Dtype>::CreateCovariance(const std::string &covariance_type, std::shared_ptr<Setting> setting) {
        return Factory::GetInstance().Create(covariance_type, std::move(setting));
    }

    template<typename Dtype>
    template<typename Derived>
    bool
    Covariance<Dtype>::Register(std::string covariance_type) {
        return Factory::GetInstance().template Register<Derived>(covariance_type, [](std::shared_ptr<Setting> setting) {
            auto covariance_setting = std::dynamic_pointer_cast<typename Derived::Setting>(setting);
            if (setting == nullptr) { covariance_setting = std::make_shared<typename Derived::Setting>(); }
            ERL_ASSERTM(covariance_setting != nullptr, "Failed to cast setting for derived Covariance of type {}.", typeid(Derived).name());
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
    Covariance<Dtype>::GetMinimumKtrainSize(const long num_samples, const long num_samples_with_gradient, const long num_gradient_dimensions) const {
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
    bool
    Covariance<Dtype>::operator==(const Covariance &other) const {
        if (m_setting_ == nullptr && other.m_setting_ != nullptr) { return false; }
        if (m_setting_ != nullptr && (other.m_setting_ == nullptr || *m_setting_ != *other.m_setting_)) { return false; }
        return true;
    }

    template<typename Dtype>
    bool
    Covariance<Dtype>::operator!=(const Covariance &other) const {
        return !(*this == other);
    }

    template<typename Dtype>
    bool
    Covariance<Dtype>::Write(const std::string &filename) const {
        ERL_INFO("Writing Covariance to file: {}", filename);
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

    template<typename Dtype>
    bool
    Covariance<Dtype>::Write(std::ostream &s) const {
        s << kFileHeader << std::endl  //
          << "# (feel free to add / change comments, but leave the first line as it is!)" << std::endl
          << "setting" << std::endl;
        // write setting
        if (!m_setting_->Write(s)) {
            ERL_WARN("Failed to write setting.");
            return false;
        }
        s << "end_of_Covariance" << std::endl;
        return s.good();
    }

    template<typename Dtype>
    bool
    Covariance<Dtype>::Read(const std::string &filename) {
        ERL_INFO("Reading Covariance from file: {}", std::filesystem::absolute(filename));
        std::ifstream file(filename.c_str(), std::ios_base::in | std::ios_base::binary);
        if (!file.is_open()) {
            ERL_WARN("Failed to open file: {}", filename.c_str());
            return false;
        }
        const bool success = Read(file);
        file.close();
        return success;
    }

    template<typename Dtype>
    bool
    Covariance<Dtype>::Read(std::istream &s) {
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
            "end_of_Covariance",
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
                case 1: {  // end_of_Covariance
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

    template<typename Dtype>
    Covariance<Dtype>::Covariance(std::shared_ptr<Setting> setting)
        : m_setting_(std::move(setting)) {}
}  // namespace erl::covariance
