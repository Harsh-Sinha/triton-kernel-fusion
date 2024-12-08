# Triton

This is our forked development repository of Triton, a language and compiler for writing highly efficient custom Deep-Learning primitives. The aim of Triton is to provide an open-source environment to write fast code at higher productivity than CUDA, but also with higher flexibility than other existing DSLs.

# Install from source

```
git clone https://github.com/triton-lang/triton.git;
cd triton;

pip install torch

pip install ninja cmake wheel pybind11; # build-time dependencies
sudo pip install -e python
```

Or with a virtualenv:

```
git clone https://github.com/triton-lang/triton.git;
cd triton;

pip install torch

python -m venv .venv --prompt triton;
source .venv/bin/activate;

pip install ninja cmake wheel pybind11; # build-time dependencies
sudo pip install -e python
```

# Tips for building

- vscode intellisense has some difficulty figuring out how to build Triton's C++
  (probably because, in our build, users don't invoke cmake directly, but
  instead use setup.py).  Teach vscode how to compile Triton as follows.

    - Do a local build. Run command `pip install -e python`
    - Get the full path to the `compile_commands.json` file produced by the build:
      `find python/build -name 'compile_commands.json' | xargs readlink -f`.
      You might get a full path similar to `/Users/{username}/triton/python/build/cmake.macosx-11.1-arm64-cpython-3.12/compile_commands.json`
    - In vscode, install the
      [C/C++
      extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools),
      then open the command palette (`Shift + Command + P` on Mac, or `Shift +
      Ctrl + P` on Windows/Linux) and open `C/C++: Edit Configurations (UI)`.
    - Open "Advanced Settings" and paste the full path to
      `compile_commands.json` into the "Compile Commands" textbox.

# Add triton binaries to your path variable
Once you have succesfully built you will have the following to your environment path.

`/full/path/to/your/triton/root/dir/python/build/cmake.OS-ARCH-cpython-VERSION/bin`  

Where OS, ARCH, and VERSION are going to be specific to your machine. Please locate this folder and make sure it exists. Place the following at the bottom in .bashrc  

`export PATH=/full/path/to/your/triton/root/dir/python/build/cmake.OS-ARCH-cpython-VERSION/bin${PATH:+:${PATH}}`  

NOTE: this should be an actual path in your system and not the pseudo path above

# Running tests

There currently isn't a turnkey way to run all the Triton tests, but you can
follow the following recipe.

```shell
# One-time setup.  Note we have to reinstall local Triton because torch
# overwrites it with the public version.
$ pip install scipy numpy torch pytest lit pandas matplotlib && pip install -e python

# Run Python tests using your local GPU.
$ python3 -m pytest python/test/unit

# Move to builddir.  Fill in <...> with the full path, e.g.
# `cmake.linux-x86_64-cpython-3.11`.
$ cd python/build/cmake<...>

# Run C++ unit tests.
$ ctest -j32

# Run lit tests.
$ lit test
```

You may find it helpful to make a symlink to the builddir and tell your local
git to ignore it.

```
$ ln -s python/build/cmake<...> build
$ echo build >> .git/info/exclude
```

Then you can e.g. rebuild and run lit with the following command.

```
$ ninja -C build && ( cd build ; lit test )
```

# Tips for hacking

For detailed instructions on how to debug Triton's frontend, please refer to this [tutorial](https://triton-lang.org/main/programming-guide/chapter-3/debugging.html). The following includes additional tips for hacking on Triton's backend.

**Helpful environment variables**

- `MLIR_ENABLE_DUMP=1` dumps the IR before every MLIR pass Triton runs, for all
   kernels. Use `MLIR_ENABLE_DUMP=kernelName` to dump for a specific kernel only.
  - Triton cache can interfere with the dump. In cases where `MLIR_ENABLE_DUMP=1` does not work, try cleaning your triton cache: `rm -r ~/.triton/cache/*`
- `LLVM_IR_ENABLE_DUMP=1` dumps the IR before every pass run over the LLVM IR.
- `TRITON_INTERPRET=1` uses the Triton interpreter instead of running on the
  GPU.  You can insert Python breakpoints in your kernel code!
- `TRITON_ENABLE_LLVM_DEBUG=1` passes `-debug` to LLVM, printing a lot of
  debugging information to stdout.  If this is too noisy, run with just
  `TRITON_LLVM_DEBUG_ONLY` instead to limit the output.

  An alternative way to reduce output noisiness is running with
  `LLVM_IR_ENABLE_DUMP=1`, extract the IR before the LLVM pass of interest, and
  then run LLVM's `opt` standalone, perhaps passing `-debug-only=foo` on the
  command line.
- `TRITON_LLVM_DEBUG_ONLY=<comma-separated>` is the equivalent of LLVM's
  `-debug-only` command-line option. This limits the LLVM debug output to
  specific pass or component names (which are specified using `#define
  DEBUG_TYPE` throughout LLVM and Triton) in order to allow the debug output to
  be less noisy. `TRITON_LLVM_DEBUG_ONLY` allows for one or more comma
  separated values to be specified (eg
  `TRITON_LLVM_DEBUG_ONLY="tritongpu-remove-layout-conversions` or
  `TRITON_LLVM_DEBUG_ONLY="tritongpu-remove-layout-conversions,regalloc"`).
- `USE_IR_LOC={ttir,ttgir}` reparses the IR such that the location information
  will be the line number of the IR file with that particular extension,
  instead of line number of the python file. This can provide a direct mapping
  from the IR to llir/ptx. When used with performance tools, it can provide a
  breakdown on IR instructions.
- `TRITON_PRINT_AUTOTUNING=1` prints out the best autotuning config and total time
  spent for each kernel after autotuning is complete.
- `DISABLE_LLVM_OPT` will disable llvm optimizations for make_llir and make_ptx
  if its value is true when parsing as Bool. Otherwise, it will be parsed as a list
  of flags to disable llvm optimizations. One usage case is
  `DISABLE_LLVM_OPT="disable-lsr"`
  Loop strength reduction is known to cause up to 10% performance changes for
  certain kernels with register pressure.
- `TRITON_ALWAYS_COMPILE=1` forces to compile kernels regardless of cache hit.
- `MLIR_ENABLE_TIMING` dumps the timing information for each MLIR pass.
- `LLVM_ENABLE_TIMING` dumps the timing information for each LLVM pass.
- `TRITON_DEFAULT_FP_FUSION` overrides the default behavior of allowing fp fusion (mul+add->fma).
- `MLIR_ENABLE_REMARK` enables the performance warnings that are emitted as remarks.

# Changelog

Version 2.0 is out! New features include:
- Many, many bug fixes
- Performance improvements
- Backend rewritten to use MLIR
- Support for kernels that contain back-to-back matmuls (e.g., flash attention)

# Contributing

Community contributions are more than welcome, whether it be to fix bugs or to add new features at [github](https://github.com/triton-lang/triton/). For more detailed instructions, please visit our [contributor's guide](CONTRIBUTING.md).


# Compatibility

Supported Platforms:
  * Linux

Supported Hardware:
  * NVIDIA GPUs (Compute Capability 7.0+)
  * AMD GPUs (ROCm 5.2+)
  * Under development: CPUs
