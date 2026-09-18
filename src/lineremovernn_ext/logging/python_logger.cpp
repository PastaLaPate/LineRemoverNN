#include "python_logger.h"

#include <iostream>

PythonLoggerBridge::PythonLoggerBridge(nb::object logger)
    : logger(std::move(logger)) {}

PythonLoggerBridge::~PythonLoggerBridge() {
  if (!logger.is_none()) {
    nb::gil_scoped_acquire acquire;
    logger = nb::none();
  }
}

void PythonLoggerBridge::debug(const std::string &message) {
  write(Level::Debug, message);
}

void PythonLoggerBridge::info(const std::string &message) {
  write(Level::Info, message);
}

void PythonLoggerBridge::warning(const std::string &message) {
  write(Level::Warning, message);
}

void PythonLoggerBridge::error(const std::string &message) {
  write(Level::Error, message);
}

void PythonLoggerBridge::write(Level level, const std::string &message) {
  if (logger.is_none()) {
    std::clog << message << '\n';
    return;
  }

  try {
    nb::gil_scoped_acquire acquire;
    const char *method = "info";
    switch (level) {
    case Level::Debug:
      method = "debug";
      break;
    case Level::Info:
      method = "info";
      break;
    case Level::Warning:
      method = "warning";
      break;
    case Level::Error:
      method = "error";
      break;
    }
    logger.attr(method)(nb::str(message.c_str()));
  } catch (const std::exception &error) {
    std::clog << "[Logger] Python logger failed: " << error.what() << '\n';
  }
}
