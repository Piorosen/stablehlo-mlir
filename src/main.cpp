#include <mlir/IR/AsmState.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Support/FileUtilities.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/IR/Verifier.h>

using namespace mlir;

int main(int argc, char ** argv) {
  MLIRContext ctx;
  ctx.loadDialect<func::FuncDialect, arith::ArithDialect>();

  OpBuilder builder(&ctx);
  auto mod = builder.create<ModuleOp>(builder.getUnknownLoc());
  
  // 设置插入点
  builder.setInsertionPointToEnd(mod.getBody());

  auto i32 = builder.getI32Type();
  auto awkf16 = builder.getFloat8E5M2Type();

  auto funcType = builder.getFunctionType({i32, awkf16}, {i32});
  auto func = builder.create<func::FuncOp>(builder.getUnknownLoc(), "test", funcType);

  auto entry = func.addEntryBlock();
  builder.setInsertionPointToEnd(entry);

  auto args = entry->getArguments();

  auto casted = builder.create<arith::FPToSIOp>(mod.getLoc(),
                                                     i32,
                                                     args[1]);

  auto addi = builder.create<arith::AddIOp>(mod.getLoc(), casted, args[0]);

  builder.create<func::ReturnOp>(mod.getLoc(), addi.getResult());
  mod->print(llvm::outs());
  return 0;
}