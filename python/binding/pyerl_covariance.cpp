//#include "erl_common/pybind11_erl_common.hpp"
#include "erl_covariance/pybind11_erl_covariance.hpp"

PYBIND11_MODULE(PYBIND_MODULE_NAME, m) {
    m.doc() = "Python 3 Interface of erl_covariance";
//    BindCommon(m);
    BindCovariance(m);
}
