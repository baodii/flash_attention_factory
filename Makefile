# ============================================================
# Makefile for Paged Attention Extensions
# ============================================================

# ------------------------------------------------------------
# Compiler and Flags
# ------------------------------------------------------------
CXX       := icpx
CXXSTD    := -std=c++20
OPTFLAGS  := -O2 -fPIC
SYCLFLAGS := -fsycl -fsycl-targets=spir64_gen
CXXFLAGS  := $(CXXSTD) $(OPTFLAGS) $(SYCLFLAGS)

# Debug build (uncomment to enable)
# CXXFLAGS := $(CXXSTD) -O0 -g -fPIC $(SYCLFLAGS)

# Ahead-Of-Time (AOT) Compilation Flags
AOTFLAGS  := -Xsycl-target-backend=spir64_gen \
             "-device bmg-g21-a0 -options '-doubleGRF -vc-codegen -Xfinalizer -printregusage'"

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
# TORCH_DIR      := /home/baodi/workspace/pytorch/torch
TORCH_DIR 		 := $(shell python3 -c "import torch; import os; print(os.path.dirname(torch.__file__))")
LIB_DIR        := $(TORCH_DIR)/lib
ESIMD_PATH     := $(CMPLR_ROOT)/include/sycl
PYTHON_PATH    := $(shell python3 -c 'import sysconfig; print(sysconfig.get_paths()["include"])')
IPEX_XETLA_DIR := /home/baodi/ipex/csrc/gpu/aten/operators/xetla/kernels

# ------------------------------------------------------------
# Includes and Libraries
# ------------------------------------------------------------
INCLUDES := -I. \
            -I$(CXXOPTS_PATH)/include \
            -I$(TORCH_DIR)/include \
            -I$(TORCH_DIR)/include/torch/csrc/api/include \
            -I$(ESIMD_PATH) \
            -I$(PYTHON_PATH) \
            -I$(IPEX_XETLA_DIR)/include \
            -I$(IPEX_XETLA_DIR)

LIBS := -L$(LIB_DIR) \
        -ltorch_python -ltorch -ltorch_xpu -ltorch_cpu \
        -lc10 -lc10_xpu \
        -Wl,-rpath,$(LIB_DIR)

# ------------------------------------------------------------
# Extension Definitions
# ------------------------------------------------------------
EXTENSIONS := paged_attention_reduce paged_attention_loop

# Auto-derive file names
SRCS := $(addsuffix .cpp,$(EXTENSIONS))
OUTS := $(addsuffix .so,$(EXTENSIONS))

# Auto-generate per-extension defines
$(foreach ext,$(EXTENSIONS), \
  $(eval $(ext)_DEFINES := -DTORCH_EXTENSION_NAME=$(ext)) \
)

# ------------------------------------------------------------
# Targets
# ------------------------------------------------------------
.PHONY: all clean

all: $(OUTS)

%.so: %.cpp
	$(CXX) $(CXXFLAGS) $(AOTFLAGS) $($(basename $@)_DEFINES) \
	      $(INCLUDES) -shared $< -o $@ $(LIBS)

clean:
	@echo "Cleaning build artifacts..."
	@rm -f $(OUTS)

