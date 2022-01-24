/* Copyright 2021 Simeon Ehrig
 *
 * This file is part of vikunja.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <vector>

namespace vikunja::bench
{
    template<typename TData>
    class IotaFunctor
    {
    private:
        TData const m_begin;
        TData const m_increment;

    public:
        //! Functor for iota implementation with generic data type.
        //!
        //! \tparam TData Type of each element
        //! \param begin Value of the first element.
        //! \param increment Distance between two elements.
        IotaFunctor(TData const begin, TData const increment) : m_begin(begin), m_increment(increment)
        {
        }

        //! Writes the result of `begin + index * increment` to each element of the output vector.
        //!
        //! \tparam TAcc The accelerator environment to be executed on.
        //! \tparam TElem The element type.
        //! \param acc The accelerator to be executed on.
        //! \param output The destination vector.
        //! \param numElements The number of elements.
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename TAcc, typename TIdx>
        ALPAKA_FN_ACC auto operator()(TAcc const& acc, TData* const output, TIdx const& numElements) const -> void
        {
            static_assert(alpaka::Dim<TAcc>::value == 1, "The VectorAddKernel expects 1-dimensional indices!");

            TIdx const gridThreadIdx(alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u]);
            TIdx const threadElemExtent(alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u]);
            TIdx const threadFirstElemIdx(gridThreadIdx * threadElemExtent);

            if(threadFirstElemIdx < numElements)
            {
                // Calculate the number of elements for this thread.
                // The result is uniform for all but the last thread.
                TIdx const threadLastElemIdx(threadFirstElemIdx + threadElemExtent);
                TIdx const threadLastElemIdxClipped(
                    (numElements > threadLastElemIdx) ? threadLastElemIdx : numElements);

                for(TIdx i(threadFirstElemIdx); i < threadLastElemIdxClipped; ++i)
                {
                    output[i] = m_begin + static_cast<TData>(i) * m_increment;
                }
            }
        }
    };


    //! Allocates memory and initialises each value with `begin + index * increment`,
    //! where index is the position in the output vector. The allocation is done with `setup.devAcc`.
    //!
    //! \tparam TData Data type of the memory buffer.
    //! \tparam TSetup Fully specialized type of `vikunja::test::TestAlpakaSetup`.
    //! \tparam Type of the extent.
    //! \tparam TBuf Type of the alpaka memory buffer.
    //! \param setup Instance of `vikunja::test::TestAlpakaSetup`. The `setup.devAcc` and `setup.queueDev` is used
    //! for allocation and initialization of the the memory.
    //! \param extent Size of the memory buffer. Needs to be 1 dimensional.
    //! \param begin Value of the first element. Depending of TData, it can be negative.
    //! \param increment Distance between two elements of the vector. If the value is negative, the value of an
    //! element is greater than its previous element.
    template<
        typename TData,
        typename TSetup,
        typename TExtent,
        typename TBuf = alpaka::Buf<typename TSetup::DevAcc, TData, alpaka::DimInt<1u>, typename TSetup::Idx>>
    TBuf allocate_mem_iota(
        TSetup& setup,
        TExtent const& extent,
        TData const begin = TData{0},
        TData const increment = TData{1})
    {
        // TODO: test also 2 and 3 dimensional memory
        static_assert(TExtent::Dim::value == 1);

        // TODO: optimize utilization for CPU backends
        typename TSetup::Idx const elementsPerThread = 1;
        typename TSetup::Idx linSize = extent.prod();

        TBuf devMem(alpaka::allocBuf<TData, typename TSetup::Idx>(setup.devAcc, extent));

        alpaka::WorkDivMembers<typename TSetup::Dim, typename TSetup::Idx> const workDiv(
            alpaka::getValidWorkDiv<typename TSetup::Acc>(
                setup.devAcc,
                extent,
                elementsPerThread,
                false,
                alpaka::GridBlockExtentSubDivRestrictions::Unrestricted));

        IotaFunctor iotaFunctor(begin, increment);

        alpaka::exec<typename TSetup::Acc>(
            setup.queueAcc,
            workDiv,
            iotaFunctor,
            alpaka::getPtrNative(devMem),
            linSize);

        return devMem;
    }
} // namespace vikunja::bench
