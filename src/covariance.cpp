#include "erl_covariance/covariance.hpp"

namespace erl::covariance {
    std::shared_ptr<Covariance>
    Covariance::CreateCovariance(const std::string &covariance_type, std::shared_ptr<Setting> setting) {
        const auto it = s_class_id_mapping_.find(covariance_type);
        if (it == s_class_id_mapping_.end()) {
            ERL_WARN("Unknown covariance type: {}. Here are the registered covariance types:", covariance_type);
            for (const auto &pair: s_class_id_mapping_) { ERL_WARN("  - {}", pair.first); }
            return nullptr;
        }
        return it->second(std::move(setting));
    }
}  // namespace erl::covariance
