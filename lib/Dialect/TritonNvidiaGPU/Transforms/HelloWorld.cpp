#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"


#include <iostream>
#include <memory>


#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"


namespace
{
    using namespace mlir;
    using namespace triton;
    using namespace triton::gpu;
    using namespace triton::nvidia_gpu;


    struct TritonNvidiaGPUHelloWorldPass 
        : public TritonNvidiaGPUHelloWorldPassBase<TritonNvidiaGPUHelloWorldPass>
    {
        virtual void runOnOperation() override
        {
            std::cout << "Hello World" << std::endl;
        }
    };
}


std::unique_ptr<Pass> mlir::createTritonNvidiaGPUHelloWorldPass()
{
    return std::make_unique<TritonNvidiaGPUHelloWorldPass>();
}