#pragma once

namespace erl::covariance {

    extern bool initialized;

    /**
     * @brief Initialize the library.
     */
    bool
    Init();

    inline const static bool kAutoInitialized = Init();

}  // namespace erl::covariance
