#include "iam.h"
#include "../utils.hpp"
#include "datasets/datasets.h"
#include "fcntl.h"
#include <cstdint>
#include <fcntl.h>
#include <filesystem>
#include <format>
#include <fstream>
#include <iosfwd>
#include <iostream>
#include <iterator>
#include <numeric>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/core/mat.hpp>
#include <ostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

using namespace cv;

constexpr uint64_t BLOB_MAGIC =
    0x1A571A733ABCDEF1ULL; // 1A571A733ABCDEF + version

bool IAM::valid() {
  std::filesystem::path words_index_path = this->path / "words.txt";
  std::filesystem::path words_path = this->path / "words";
  return std::filesystem::exists(words_index_path) &&
         std::filesystem::is_regular_file(words_index_path) &&
         std::filesystem::exists(words_path) &&
         std::filesystem::is_directory(words_path);
}

void IAM::load() {
  std::ifstream WordsIndex(this->path / "words.txt");
  std::string line;

  auto start_time = std::chrono::high_resolution_clock::now();

  while (std::getline(WordsIndex, line)) {
    if (line.starts_with("#"))
      continue;
    std::vector<std::string> tokens = split_ws(line);
    if (tokens.size() != 9)
      continue;
    if (tokens[1] == "err")
      continue;

    std::vector<std::string_view> parts = split(tokens[0], '-');
    std::span<std::string_view> second_path = {parts.begin(), 2};
    std::string joined_parts = std::accumulate(
        std::next(second_path.begin()), second_path.end(),
        std::string(second_path[0]), // Explicitly start with a std::string
        [](std::string a, std::string_view b) {
          return std::move(a) + "-" + std::string(b);
        });
    std::filesystem::path img_path =
        this->path / "words" / parts[0] / joined_parts / (tokens[0] + ".png");
    this->words.push_back(
        {.path{img_path},
         .bbox{{parse_int(tokens[3]), parse_int(tokens[4]),
                parse_int(tokens[5]), parse_int(tokens[6])}},
         .transcript{tokens[8]},
         .gray_scale = static_cast<uint8_t>(parse_int(tokens[2]))});
  }

  WordsIndex.close();
  auto end_time = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> duration_ms = end_time - start_time;

  uint64_t parsed_count = this->words.size();
  if (parsed_count > 0) {
    double avg_time = duration_ms.count() / parsed_count;
    std::cout << std::format(
        "[IAM::load] Loaded {} words in {:.2f} ms ({:.4f} ms/word)\n",
        parsed_count, duration_ms.count(), avg_time);
  } else {
    std::cout << "[IAM::load] No words were loaded." << std::endl;
  }

  if (this->index) {
    std::cout << "[IAM::load] Starting blob generation." << std::endl;
    this->generate_blob();
    std::cout << "[IAM::load] Starting blob reading." << std::endl;
    this->read_blob();
  }

  if (this->preload) {

    auto start_time = std::chrono::high_resolution_clock::now();
    std::cout << "[IAM::load] Preloading images..." << std::endl;
    this->preloaded_images.resize(parsed_count);
    for (uint64_t i = 0; i < parsed_count; i++) {
      this->read_image(i, true);
    }
    if (this->index) {
      ::close(this->blob_fd);
      this->blob_fd = -1;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_ms =
        end_time - start_time;
    double avg_time = duration_ms.count() / parsed_count;
    std::cout << std::format(
        "[IAM::load] Preloaded {} words in {:.2f} ms ({:.4f} ms/word)\n",
        parsed_count, duration_ms.count(), avg_time);
    size_t usage = sizeof(this->preloaded_images) +
                   (this->preloaded_images.capacity() * sizeof(cv::Mat));
    for (const auto &mat : this->preloaded_images) {
      if (!mat.empty() && mat.data) {
        usage += (mat.step * mat.rows);
      }
    }

    std::cout << std::format("[IAM::load] Total ram usage: {:.2f} MiB",
                             usage / (1024.0 * 1024.0))
              << std::endl;
  }
}

void IAM::generate_blob() {
  uint64_t parsed_count = this->words.size();
  auto start_time = std::chrono::high_resolution_clock::now();
  std::cout << "[IAM::load::indexer] Starting indexing..." << std::endl;
  std::filesystem::path blobFile = this->path / "words.blob";
  if (std::filesystem::exists(blobFile) &&
      !std::filesystem::is_directory(blobFile)) {
    std::ifstream in(blobFile, std::ios::binary);
    uint64_t magic, count;
    in.read(
        reinterpret_cast<char *>(&magic),
        sizeof(magic)); // do like magic was a list of bytes instead of an int.
    in.read(reinterpret_cast<char *>(&count), sizeof(count));

    if (magic == BLOB_MAGIC && count == parsed_count) {
      std::cout
          << "[IAM::load::indexer] Found existing blob. aborting generation"
          << std::endl;
      return;
    }
    std::cout << "[IAM::load::indexer] Existing blob is outdated, regenerating"
              << std::endl;
    // Version changed or new words added for some reason, redo indexing
  }
  std::ofstream out(blobFile, std::ios::binary |
                                  std::ios::trunc); // Delete existing data.
  out.write(reinterpret_cast<const char *>(&BLOB_MAGIC), sizeof(BLOB_MAGIC));
  out.write(reinterpret_cast<char *>(&parsed_count), sizeof(parsed_count));

  std::streampos index_start =
      out.tellp(); // Get current offset (index_start) to write index when
                   // finished loading / writing all of the data.

  // Index structure: uint64 (8 bytes) byte offset + uint32 (4 bytes) len of
  // data

  out.seekp(
      parsed_count * (sizeof(uint64_t) + sizeof(uint32_t)),
      std::ios::cur); // Cur: relative to current, else can be rewritten as
                      // sizeof(magic) + sizeof(count) + parsed_count...

  // Fill datas
  std::vector<uint64_t> offsets(parsed_count);
  std::vector<uint32_t> lengths(parsed_count);

  uint64_t offset_pos = 0; // Next image's data offset

  for (uint64_t i = 0; i < parsed_count; i++) {
    IAMWordEntry word = this->words[i];
    std::ifstream img_in(word.path, std::ios::binary);
    if (!img_in.is_open()) {
      // Cant find image
      std::cerr << std::format(
          "[IAM::load::blob_generator] Missing file: {}, ignoring\n",
          word.path.string());
      offsets[i] = 0;
      lengths[i] = 0;
      continue;
    }
    std::vector<char> bytes((std::istreambuf_iterator<char>(img_in)),
                            std::istreambuf_iterator<char>());
    offsets[i] = offset_pos;
    lengths[i] = static_cast<uint32_t>(bytes.size());
    if (!bytes.empty())
      out.write(bytes.data(), bytes.size());

    offset_pos += bytes.size();
  }

  out.seekp(index_start);

  for (uint64_t i = 0; i < parsed_count; i++) {
    out.write(reinterpret_cast<char *>(&offsets[i]), sizeof(uint64_t));
    out.write(reinterpret_cast<char *>(&lengths[i]), sizeof(uint32_t));
  }

  auto end_time = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> duration_ms = end_time - start_time;

  std::cout << "[IAM::load::blob_generator] Blob of size " << offset_pos
            << " generated in " << duration_ms.count() << "ms" << std::endl;
}

void IAM::read_blob() {
  std::filesystem::path blobFile = this->path / "words.blob";
  this->blob_fd = ::open(blobFile.c_str(), O_RDONLY);
  if (this->blob_fd < 0) {
    throw std::runtime_error("Couldnt open file " + blobFile.string());
  }

  std::ifstream in(blobFile, std::ios::binary);
  uint64_t magic, count;
  in.read(reinterpret_cast<char *>(&magic), sizeof(magic));
  in.read(reinterpret_cast<char *>(&count), sizeof(count));

  if (magic != BLOB_MAGIC || count != this->words.size()) {
    throw std::runtime_error("Blob is outdated and/or corrupted " +
                             blobFile.string());
  }

  uint64_t data_start =
      sizeof(magic) + sizeof(count) +
      count * (sizeof(uint64_t) + sizeof(uint32_t)); // to get absolute offset

  for (uint64_t i = 0; i < count; i++) {
    uint64_t offset;
    uint32_t len;

    in.read(reinterpret_cast<char *>(&offset), sizeof(offset));
    in.read(reinterpret_cast<char *>(&len), sizeof(len));

    this->words[i].blob_offset = data_start + offset;
    this->words[i].blob_length = len;
  }
}

long IAM::len() { return this->words.size(); }

cv::Mat IAM::read_image(int idx, bool preloading) {
  if (this->preload && !preloading) {
    return this->preloaded_images[idx];
  }

  IAMWordEntry word = this->words[idx];

  cv::Mat img;

  if (this->index && this->blob_fd) {
    std::vector<uchar> buf(word.blob_length);
    ssize_t bytes_read =
        ::pread(this->blob_fd, buf.data(), word.blob_length, word.blob_offset);

    if (bytes_read != word.blob_length) {
      std::cerr << "Short read for image " << word.path << std::endl;
      return img;
    }

    if (buf.empty()) {
      std::cerr << std::format("[IAM] Empty file: {}\n", word.path.string());
      return img;
    }
    img = cv::imdecode(buf, cv::IMREAD_GRAYSCALE);
  } else {
    std::ifstream f(word.path, std::ios::binary);
    if (!f.is_open()) {

      std::cerr << std::format("[IAM] Missing file: {}\n", word.path.string());
      return img;
    }

    std::vector<uchar> buf((std::istreambuf_iterator<char>(f)),
                           std::istreambuf_iterator<char>());

    if (buf.empty()) {
      std::cerr << std::format("[IAM] Empty file: {}\n", word.path.string());
      return img;
    }
    img = cv::imdecode(buf, cv::IMREAD_GRAYSCALE);
  }

  if (preloading) {
    this->preloaded_images[idx] = img;
  }

  if (!img.empty())
    img.setTo(255, img > 160);

  return img;
}

cv::Mat IAM::get_image(int idx) {
  return this->read_image(idx, false);
} // keep base signature

AssetRow IAM::get_asset(int idx) {
  IAMWordEntry word = this->words[idx];
  cv::Mat img = this->get_image(idx);

  return {.idx = idx,
          .dataset = "iam",
          .image = img,
          .transcript = word.transcript};
}

std::array<int, 2> IAM::get_size(int idx) {
  IAMWordEntry word = this->words[idx];
  return {word.bbox[2], word.bbox[3]};
}