# starcross toolchain: CUDA -> OpenCL -> SPIR-V -> GLSL / WGSL
#
#   bin/starcross-container build                     # podman or docker, whichever is installed
#   bin/starcross-container convert examples/vecadd.cu
#
# The clspv build compiles its own copy of LLVM/clang; expect 30-90 minutes on the
# first build depending on cores. It's in its own stage so rebuilding the tool is fast.

# ---------------------------------------------------------------- build stage
FROM docker.io/library/ubuntu:24.04 AS build
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential clang lld cmake ninja-build python3 git ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

ARG CLSPV_REF=main
RUN git clone --depth 1 --branch "${CLSPV_REF}" https://github.com/google/clspv.git /src/clspv && \
    cd /src/clspv && python3 utils/fetch_sources.py --shallow && \
    cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_USE_LINKER=lld && \
    ninja -C build clspv clspv-reflection && \
    install -D -m755 build/bin/clspv /out/clspv && \
    install -D -m755 build/bin/clspv-reflection /out/clspv-reflection

# SPIRV-Cross from source (Ubuntu's package is years old)
RUN git clone --depth 1 https://github.com/KhronosGroup/SPIRV-Cross.git /src/spirv-cross && \
    cmake -S /src/spirv-cross -B /src/spirv-cross/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
          -DSPIRV_CROSS_ENABLE_TESTS=OFF -DSPIRV_CROSS_SHARED=OFF && \
    ninja -C /src/spirv-cross/build spirv-cross && \
    install -D -m755 /src/spirv-cross/build/spirv-cross /out/spirv-cross

# naga (SPIR-V -> WGSL)
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal && \
    /root/.cargo/bin/cargo install naga-cli --locked --root /out-naga && \
    install -D -m755 /out-naga/bin/naga /out/naga

# -------------------------------------------------------------- runtime stage
FROM docker.io/library/ubuntu:24.04
LABEL org.opencontainers.image.source="https://github.com/PDUBZ3R0/STAR-CROSS" \
      org.opencontainers.image.title="starcross-toolchain" \
      org.opencontainers.image.description="CUDA -> OpenCL -> SPIR-V -> GLSL/WGSL (cu2cl + clspv + spirv-cross + naga)"
ENV DEBIAN_FRONTEND=noninteractive
# clang is used by cu2cl for the syntax check and the __constant address-space fixups
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 clang spirv-tools glslang-tools ca-certificates && \
    rm -rf /var/lib/apt/lists/*
COPY --from=build /out/ /usr/local/bin/
COPY starcross /opt/starcross/starcross
COPY bin/starcross /opt/starcross/bin/starcross
RUN chmod +x /opt/starcross/bin/starcross && ln -s /opt/starcross/bin/starcross /usr/local/bin/starcross && \
    starcross doctor
WORKDIR /work
ENTRYPOINT ["starcross"]
CMD ["--help"]
