#include <map>
#include <string>
#include <vector>

namespace lexicon {

using CellState = bool;
using Pattern = std::vector<std::vector<CellState>>;

class PatternDict {
    static constexpr CellState _ = false;
    static constexpr CellState X = true;

  public:
    static Pattern empty_pattern() {
        return {{}};
    }

    std::map<std::string, Pattern> all_patterns() {
        return {
            {"block",
             {
                 {X, X},
                 {X, X},
             }},
            {"glider",
             {
                 {_, X, _},
                 {_, _, X},
                 {X, X, X},
             }},
            {"beehive",
             {
                 {_, X, X, _},
                 {X, _, _, X},
                 {_, X, X, _},
             }},
            {"loaf",
             {
                 {_, X, X, _},
                 {X, _, _, X},
                 {_, X, _, X},
                 {_, _, X, _},
             }},
            {"boat",
             {
                 {X, X, _},
                 {X, _, X},
                 {_, X, _},
             }},
            {"tub",
             {
                 {_, X, _},
                 {X, _, X},
                 {_, X, _},
             }},
            {"blinker",
             {
                 {X},
                 {X},
                 {X},
             }},
            {"toad",
             {
                 {_, X, X, X},
                 {X, X, X, _},
             }},
            {"beacon",
             {
                 {X, X, _, _},
                 {X, X, _, _},
                 {_, _, X, X},
                 {_, _, X, X},
             }},
        };
    }
};

} // namespace lexicon