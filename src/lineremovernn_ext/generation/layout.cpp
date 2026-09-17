#include "layout.h"
#include "page_gen.h"

Layout generate_document_layout(const PageSettings &settings) {}

Layout generate_page_layout(const PageSettings &settings) {}

Layout generate_layout(const PageSettings &settings) {
  return settings.document ? generate_document_layout(settings)
                           : generate_page_layout(settings);
}
