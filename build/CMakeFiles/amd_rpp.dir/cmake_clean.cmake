file(REMOVE_RECURSE
  "cppcheck-build"
  "fixits"
  "libamd_rpp.pdb"
  "libamd_rpp.so"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/amd_rpp.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
