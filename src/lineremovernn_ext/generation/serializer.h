#pragma once

#include "generation/layout.h"
#include "page_gen.h"
#include "pugixml/pugixml.hpp"

pugi::xml_document serialize_xml(int page_idx, const PageSettings &settings,
                                 const Layout &layout);
