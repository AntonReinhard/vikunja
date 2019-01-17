//
// Created by hauke on 17.01.19.
//

#pragma once

#include <alpaka/alpaka.hpp>
#include <algorithm>
#include <thread>
namespace vikunja {
    namespace workdiv {
        namespace policies {
            struct BlockBasedSequentialPolicy {
                template<typename TAcc, typename TIdx = alpaka::idx::Idx<TAcc>>
                static constexpr TIdx getBlockSize() noexcept {
                    return 1;
                }

                template<typename TAcc, typename TDevAcc, typename TIdx = alpaka::idx::Idx<TAcc>>
                static constexpr TIdx getGridSize(TDevAcc const &devAcc) noexcept {
                    return 1;
                }
            };
            // for accelerators that parallelize on the grid-block-level
            struct BlockBasedGridBlockPolicy {
                template<typename TAcc, typename TIdx = alpaka::idx::Idx<TAcc>>
                static constexpr TIdx getBlockSize() {
                    return 1;
                }

                template<typename TAcc, typename TDevAcc, typename TIdx = alpaka::idx::Idx<TAcc>>
                static TIdx getGridSize(TDevAcc const &devAcc) {
                    return std::max(static_cast<TIdx>(1), static_cast<TIdx>(std::thread::hardware_concurrency()));
                }

            };
            // TODO: ok, here we have a problem. getBlockSize cannot be dynamic as it is currently a compile-time variable, but it cannot be set higher than the thread limit for some reason. I dont really know what happens, but nevermind.
            // for accelerators that parallelize on the block-thread level
            struct BlockBasedBlockThreadPolicy {
                template<typename TAcc, typename TIdx = alpaka::idx::Idx<TAcc>>
                static constexpr TIdx getBlockSize() noexcept {
                    // TODO: figure out if this really needs to be a constexpr
                    // probably yes.
                    return 2;
                }
                template<typename TAcc, typename TDevAcc, typename TIdx = alpaka::idx::Idx<TAcc>>
                static constexpr TIdx getGridSize(TDevAcc const &devAcc) noexcept {
                    return 4;
                }
            };
            // for CUDA
            struct BlockBasedCudaPolicy {
                template<typename TAcc, typename TIdx = alpaka::idx::Idx<TAcc>>
                static constexpr TIdx getBlockSize() noexcept {
                    // TODO
                    return 1024;
                }
                template<typename TAcc, typename TDevAcc, typename TIdx = alpaka::idx::Idx<TAcc>>
                static TIdx getGridSize(TDevAcc const &devAcc) {
                    // Jonas Schenke calculated this for CUDA
                    static_cast<TIdx>(
                            alpaka::acc::getAccDevProps<TAcc>(devAcc).m_multiProcessorCount * 8);
                }

            };
        } // policies

        // TODO: implement for other accelerators
        namespace traits {
            template<typename TAcc, typename TSfinae = void>
            struct GetBlockBasedPolicy{
                // use sequential as default
                using type = policies::BlockBasedSequentialPolicy;
            };
#ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED
            template<typename... TArgs>
            struct GetBlockBasedPolicy<alpaka::acc::AccCpuOmp2Blocks<TArgs...>> {
                using type = policies::BlockBasedGridBlockPolicy;
            };
#endif
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED
            template<typename... TArgs>
            struct GetBlockBasedPolicy<alpaka::acc::AccCpuOmp2Threads<TArgs...>> {
                using type = policies::BlockBasedBlockThreadPolicy;
            };
#endif
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED
            template<typename... TArgs>
            struct GetBlockBasedPolicy<alpaka::acc::AccCpuThreads<TArgs...>> {
                using type = policies::BlockBasedBlockThreadPolicy;
            };
#endif
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
            template<typename... TArgs>
            struct GetBlockBasedPolicy<alpaka::acc::AccGpuCudaRt<TArgs...>> {
                using type = policies::BlockBasedCudaPolicy;
            };
#endif
        } // traits

        template<typename TAcc>
        using BlockBasedPolicy = typename traits::GetBlockBasedPolicy<TAcc>::type;
    } // workdiv
} // vikunja