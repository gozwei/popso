#include "milp/core/memory.hpp"

#if defined(__APPLE__)
// Recent macOS SDKs use C11 _Static_assert in the Mach headers, which g++ does not accept in C++.
#if defined(__GNUC__) && !defined(__clang__) && !defined(_Static_assert)
#define _Static_assert static_assert
#define POPSO_DEFINED_STATIC_ASSERT
#endif
#include <mach/mach.h>
#include <malloc/malloc.h>
#ifdef POPSO_DEFINED_STATIC_ASSERT
#undef _Static_assert
#undef POPSO_DEFINED_STATIC_ASSERT
#endif
#elif defined(__linux__)
#include <unistd.h>

#include <cstdio>
#if defined(__GLIBC__)
#include <malloc.h>
#endif
#elif defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#if defined(_MSC_VER)
#pragma comment(lib, "psapi.lib")  // also linked by CMake; keeps other build systems working
#endif
#endif

namespace milp {

long long resident_bytes() {
#if defined(__APPLE__)
  // The physical footprint: what the OS charges the process. The resident size also counts pages
  // malloc has freed and marked reusable (they stay resident until reused or reclaimed), so after
  // large frees it overstates the memory in use by up to the freed amount.
  task_vm_info_data_t vm;
  mach_msg_type_number_t vcount = TASK_VM_INFO_COUNT;
  if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&vm), &vcount) == KERN_SUCCESS) {
    return static_cast<long long>(vm.phys_footprint);
  }
  mach_task_basic_info_data_t info;
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
  if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, reinterpret_cast<task_info_t>(&info), &count) != KERN_SUCCESS) {
    return 0;
  }
  return static_cast<long long>(info.resident_size);
#elif defined(__linux__)
  std::FILE* f = std::fopen("/proc/self/statm", "r");
  if (!f) return 0;
  long long pages = 0, resident = 0;
  const int n = std::fscanf(f, "%lld %lld", &pages, &resident);
  std::fclose(f);
  return n == 2 ? resident * static_cast<long long>(sysconf(_SC_PAGESIZE)) : 0;
#elif defined(_WIN32)
  PROCESS_MEMORY_COUNTERS pmc;
  if (!GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) return 0;
  return static_cast<long long>(pmc.WorkingSetSize);
#else
  return 0;
#endif
}

void release_free_memory() {
#if defined(__APPLE__)
  malloc_zone_pressure_relief(nullptr, 0);
#elif defined(__linux__) && defined(__GLIBC__)
  malloc_trim(0);
#endif
}

}  // namespace milp
