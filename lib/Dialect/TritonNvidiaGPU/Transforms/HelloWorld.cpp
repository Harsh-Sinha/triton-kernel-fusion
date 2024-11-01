#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Builders.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Attributes.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Format.h"
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
            auto module = getOperation();
            SmallVector<Operation *, 8> targetOps;
            module.walk([&](Operation *op) {
                targetOps.push_back(op);
            });
            for (Operation *op : targetOps) {
                OpBuilder builder(op);
                builder.setInsertionPoint(op);
                auto newConstOp = builder.create<arith::ConstantOp>(
                    op->getLoc(),
                    builder.getI32Type(),
                    builder.getI32IntegerAttr(42)
                );
            }
        }
    };
}

std::unique_ptr<Pass> mlir::createTritonNvidiaGPUHelloWorldPass()
{
    return std::make_unique<TritonNvidiaGPUHelloWorldPass>();
}