#pragma once
#include <charconv>
#include <cstring>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

inline std::vector<std::string> split_ws(const std::string &s) {
  std::istringstream ss(s);
  return {std::istream_iterator<std::string>(ss), {}};
}

inline std::vector<std::string_view> split(std::string_view s, char delim) {
  std::vector<std::string_view> tokens;
  size_t start = 0, pos;
  while ((pos = s.find(delim, start)) != std::string_view::npos) {
    tokens.emplace_back(s.substr(start, pos - start));
    start = pos + 1;
  }
  tokens.emplace_back(s.substr(start));
  return tokens;
}

// Parse int — from_chars is the modern, fast, no-exception way
inline int parse_int(std::string_view s) {
  int result;
  auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), result);
  if (ec != std::errc{})
    throw std::invalid_argument("Invalid int: \"" + std::string(s) + "\"");
  return result;
}

inline char *strstrip(char *s) {
  size_t size;
  char *end;

  size = strlen(s);

  if (!size)
    return s;

  end = s + size - 1;
  while (end >= s && isspace(*end))
    end--;
  *(end + 1) = '\0';

  while (*s && isspace(*s))
    s++;

  return s;
}
