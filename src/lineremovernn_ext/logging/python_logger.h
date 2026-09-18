#pragma once

#include <string>

#include <nanobind/nanobind.h>

namespace nb = nanobind;

class PythonLoggerBridge {
public:
  explicit PythonLoggerBridge(nb::object logger);
  ~PythonLoggerBridge();

  void debug(const std::string &message);
  void info(const std::string &message);
  void warning(const std::string &message);
  void error(const std::string &message);

private:
  enum class Level { Debug, Info, Warning, Error };

  void write(Level level, const std::string &message);

  nb::object logger;
};
