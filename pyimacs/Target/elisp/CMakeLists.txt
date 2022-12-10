add_mlir_translation_library(ElispTranslation
        translateToElisp.cc

        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRLLVMIR
        MLIRSupport
        MLIRTargetLLVMIRExport
        )
add_dependencies(ElispTranslation LispTableGen)
