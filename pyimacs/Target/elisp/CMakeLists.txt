add_mlir_translation_library(PyIMacsElisp
        translateToElisp.cc

        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRLLVMIR
        MLIRSupport
        MLIRTargetLLVMIRExport
        )
