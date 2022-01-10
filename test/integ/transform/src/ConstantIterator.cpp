/* Copyright 2022 Anton Reinhard
 *
 * This file is part of vikunja.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <vikunja/mem/iterator/ConstantIterator.hpp>
#include <vikunja/transform/transform.hpp>

#include <alpaka/alpaka.hpp>

#include <catch2/catch.hpp>

TEST_CASE()
{
    // Define the accelerator here. Must be one of the enabled accelerators.
    using TAcc = alpaka::AccGpuCudaRt<alpaka::DimInt<3u>, std::uint64_t>;

    // Type of the data that will be reduced
    using TRed = uint64_t;

    // Alpaka index type
    using Idx = alpaka::Idx<TAcc>;
    // Alpaka dimension type
    using Dim = alpaka::Dim<TAcc>;
    // Type of the extent vector
    using Vec = alpaka::Vec<Dim, Idx>;
    // Find the index of the CUDA blockIdx.x component. Alpaka somehow reverses
    // these, i.e. the x component of cuda is always the last value in the vector
    constexpr Idx xIndex = Dim::value - 1u;
    // number of elements to reduce
    const Idx n = static_cast<Idx>(1000000);

    // create output
    std::vector<TRed> output(n);
    auto v_begin = output.begin();

    // Value for constant iterator
    const TRed constantIterVal = 10;

    // define device, platform, and queue types.
    using DevAcc = alpaka::Dev<TAcc>;
    using PltfAcc = alpaka::Pltf<DevAcc>;
    // using QueueAcc = alpaka::test::queue::DefaultQueue<alpaka::Dev<TAcc>>;
    using PltfHost = alpaka::PltfCpu;
    using DevHost = alpaka::Dev<PltfHost>;
    using QueueAcc = alpaka::Queue<TAcc, alpaka::Blocking>;

    // Get the host device.
    DevHost devHost(alpaka::getDevByIdx<PltfHost>(0u));
    // Select a device to execute on.
    DevAcc devAcc(alpaka::getDevByIdx<PltfAcc>(0u));
    // Get a queue on the accelerator device.
    QueueAcc queueAcc(devAcc);

    auto times_two = [](TRed v) { return v * 2; };

    vikunja::mem::iterator::ConstantIterator c_begin(constantIterVal);
    /*
        // Allocate memory for the device.
        auto deviceMem(alpaka::allocBuf<Data, Idx>(devAcc, n));
        // The memory is accessed via a pointer.
        auto deviceNativePtr = alpaka::getPtrNative(deviceMem);
        // Allocate memory for the host.
        auto hostMem(alpaka::allocBuf<Data, Idx>(devHost, n));
        auto hostNativePtr = alpaka::getPtrNative(hostMem);
    */
    vikunja::transform::deviceTransform(devAcc, queueAcc, n, c_begin, v_begin, times_two);
}
