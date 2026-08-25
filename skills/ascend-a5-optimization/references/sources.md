# Source register and refresh guide

Last verified: **2026-08-22**.

## Authority levels

1. Huawei architecture whitepaper and product specification page
2. Matching CANN version documentation and official release notes
3. Huawei corporate roadmap/keynote for future availability
4. Official open-source code for implementation behavior
5. Third-party analysis only as a lead; do not use it to establish a hardware fact when a primary source exists

## Primary sources

- [CANN community release page for the Ascend 950 NPU Architecture Whitepaper](https://gitcode.com/cann) — find “昇腾 950 NPU 架构白皮书” and its document download.
- [Direct Ascend 950 NPU Architecture Whitepaper PDF](https://public-download.obs.cn-east-2.myhuaweicloud.com/ascend/%E6%98%87%E8%85%BE950%20NPU%E6%9E%B6%E6%9E%84%E7%99%BD%E7%9A%AE%E4%B9%A6.pdf) — product bins, architecture, compute, local memory, CV, NDDMA, L2, DVPP, and interconnect.
- [Huawei Ascend processor page](https://www.hiascend.com/hardware/processor) — current 950PR/950DT product positioning and product whitepaper link.
- [Atlas accelerator-card page](https://www.hiascend.com/hardware/accelerator-card) — Atlas 350 concrete card specification.
- [Huawei Connect 2025 keynote](https://www.huawei.com/cn/news/2025/9/hc-xu-keynote-speech) — original roadmap, PR/DT workload split, 128 B access granularity statement, and planned availability.

## CANN 9.1 documentation

- [CANN 9.1 documentation index](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/index/index.html) — release scope and current product support.
- [AI Core SIMD model overview](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/programug/Ascendcopdevg/docs/guide/%E7%BC%96%E7%A8%8B%E6%8C%87%E5%8D%97/%E7%BC%96%E7%A8%8B%E6%A8%A1%E5%9E%8B/AI-Core-SIMD%E7%BC%96%E7%A8%8B/%E6%A6%82%E8%BF%B0.md) — `__mix__` ratios, A5 register layer, dataflow.
- [AI Core SIMT abstract architecture](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta3/programug/Ascendcopdevg/docs/guide/%E7%BC%96%E7%A8%8B%E6%8C%87%E5%8D%97/%E7%BC%96%E7%A8%8B%E6%A8%A1%E5%9E%8B/AI-Core-SIMT%E7%BC%96%E7%A8%8B/%E6%8A%BD%E8%B1%A1%E7%A1%AC%E4%BB%B6%E6%9E%B6%E6%9E%84.md) — registers, UB shared memory, 32–128 KB Data Cache.
- [DataCopy API (A5 paths)](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta2/API/ascendcopapi/atlasascendc_api_07_0103.html) — SSBuffer CV path and path-specific units/constraints.
- [SIMD API common alignment and TPosition mapping](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta2/API/ascendcopapi/atlasascendc_api_07_0004.html) — UB/L1/L0/Fixpipe mapping and alignment.
- [PlatformAscendC](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_00059.html) — runtime tiling queries for core counts, memory size/bandwidth, and register length.
- [A5 op_summary profiling fields](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/devaids/Profiling/atlasprofiling_16_0067.html) — available counters and documented gaps.
- [msOpProf roofline guide](https://www.hiascend.com/document/detail/zh/mindstudio/2610/msOT/Operatordevelopmenttools/docs/zh/user_guide/msopprof_user_guide.md) — A5 roofline interpretation and latency-bound classification.

## Refresh procedure

Use the `web-access` skill and search/open primary sources directly. For every changed value record:

- retrieval date
- product/SKU and software version scope
- exact source URL and table/section
- whether the value is theoretical, sustained, or measured
- superseded value and why it changed

Do not copy a full whitepaper into this skill. Distill only optimization-relevant facts and retain the primary link.

