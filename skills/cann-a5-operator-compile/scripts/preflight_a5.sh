#!/usr/bin/env bash

# Read-only preflight for a CANN 9.1 AscendC direct-launch build on Ascend A5.
set -uo pipefail

repo_root="${1:-.}"
soc_version="${2:-${SOC_VERSION:-}}"
cann_root="${3:-${ASCEND_HOME_PATH:-${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/cann-9.1.0}}}"
failures=0
risks=0

pass() { printf 'PASS: %s\n' "$1"; }
warn() { printf 'RISK: %s\n' "$1"; risks=$((risks + 1)); }
fail() { printf 'FAIL: %s\n' "$1"; failures=$((failures + 1)); }

if [[ -d "$repo_root/csrc" && -f "$repo_root/build.sh" ]]; then
    pass "repository layout found at $repo_root"
else
    fail "expected csrc/ and build.sh under $repo_root"
fi

soc_lower=$(printf '%s' "$soc_version" | tr '[:upper:]' '[:lower:]')
if [[ "$soc_lower" == ascend950* ]]; then
    pass "SOC_VERSION=$soc_version maps to the Ascend950/A5 family"
elif [[ -z "$soc_version" ]]; then
    fail "SOC_VERSION is empty; pass it as the second argument"
else
    fail "SOC_VERSION=$soc_version is not an Ascend950/A5 target"
fi

if [[ -d "$cann_root" ]]; then
    pass "CANN root exists at $cann_root"
else
    fail "CANN root does not exist at $cann_root"
fi

if [[ "$cann_root" == *9.1* ]]; then
    pass "CANN path indicates a 9.1 toolkit"
else
    warn "CANN path does not identify version 9.1; verify the active toolkit version"
fi

bisheng_path=""
for candidate in \
    "$cann_root/bin/bisheng" \
    "$cann_root/compiler/ccec_compiler/bin/bisheng" \
    "$cann_root/x86_64-linux/ccec_compiler/bin/bisheng" \
    "$cann_root/aarch64-linux/ccec_compiler/bin/bisheng"; do
    if [[ -x "$candidate" ]]; then
        bisheng_path="$candidate"
        break
    fi
done
if [[ -n "$bisheng_path" ]]; then
    pass "bisheng found at $bisheng_path"
else
    fail "bisheng was not found below $cann_root"
fi

if find "$cann_root" -name ASCConfig.cmake -print -quit 2>/dev/null | grep -q .; then
    pass "ASCConfig.cmake found"
else
    warn "ASCConfig.cmake not found; confirm configure still creates AscendC targets"
fi

if [[ -d "$repo_root/csrc" ]]; then
    while IFS= read -r source_file; do
        [[ -n "$source_file" ]] || continue
        if grep -qE 'MicroAPI|arch35/' "$source_file" && ! grep -q '__NPU_ARCH__' "$source_file"; then
            warn "A5 entry source may expose device-only headers to host-bisheng: $source_file"
        fi
    done < <(find "$repo_root/csrc" -type f -name '*a5*.cpp' -print 2>/dev/null)

    while IFS= read -r source_file; do
        [[ -n "$source_file" ]] || continue
        warn "GET_TILING_DATA appears in an A5-named source; verify this is not direct launch: $source_file"
    done < <(grep -R -l --include='*a5*.cpp' 'GET_TILING_DATA' "$repo_root/csrc" 2>/dev/null || true)

    if grep -R -n --include='CMakeLists.txt' -- '--op_relocatable_kernel_binary=true' "$repo_root/csrc" >/dev/null 2>&1; then
        warn "relocatable-kernel option found; verify it is consumed by an OPP API, not raw bisheng options"
    fi
fi

printf 'SUMMARY: failures=%d risks=%d\n' "$failures" "$risks"
if (( failures > 0 )); then
    exit 1
fi
