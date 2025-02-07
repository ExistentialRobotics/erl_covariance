#pragma once

template<typename Dtype>
YAML::Node
Covariance<Dtype>::Setting::Encode() const {
    YAML::Node node(YAML::NodeType::Map);
    node["x_dim"] = x_dim;
    node["alpha"] = alpha;
    node["scale"] = scale;
    node["scale_mix"] = scale_mix;
    node["weights"] = weights;
    return node;
}

template<typename Dtype>
bool
Covariance<Dtype>::Setting::Decode(const YAML::Node &node) {
    if (!node.IsMap()) { return false; }
    x_dim = node["x_dim"].as<int>();
    alpha = node["alpha"].as<Dtype>();
    scale = node["scale"].as<Dtype>();
    scale_mix = node["scale_mix"].as<Dtype>();
    weights = node["weights"].as<Vector>();
    return true;
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
