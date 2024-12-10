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

If you have a GPU you can run the following file: `examples/comm_buffer_int.py`. This will output all the optimization passes ran on the kernel code and the results before and after. You can parse the file to find the output of our optimization pass.  

If you do NOT have a GPU you can navigate to the following directory and run the following command:

dir: `cd test/TritonNvidiaGPU`  
cmd: `triton-opt hello_world.mlir -split-input-file --triton-nvidia-hello-world --allocate-shared-memory`  
  
You will see in between every instruction originally in `hello_world.mlir` there is the following "no op instruction":  
`%c42_i32_0 = arith.constant 42 : i32`  

# Modifed Code Location
Our optimization pass is located here: `lib/Dialect/TritonNvidiaGPU/Transforms/HelloWorld.cpp`  
  
Here are other files we had to modify as well:  
`include/triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h`  
`include/triton/Dialect/TritonNvidiaGPU/Transforms/Passes.td`  
`lib/Dialect/TritonNvidiaGPU/Transforms/CMakeLists.txt`  
`third_party/nvidia/triton_nvidia.cc`  
  
Additionally, here are test files we wrote:  
`test/TritonNvidiaGPU/hello_world.mlir`  
`examples/comm_buffer_float.py`  
`examples/comm_buffer_int.py`

