#include <string>
#include <vector>

namespace debug_utils {

    
    template <typename T>
    std::string pretty(T &elem) {
        return std::to_string(elem);
    }

    template <typename T>
    std::string pretty(std::vector<T> &vec) {
        std::string result = "[";
        for (auto &elem : vec) {
            result += pretty(elem) + ", ";
        }
        result.pop_back();
        result.pop_back();
        result += "]";
        return result;
    }
}