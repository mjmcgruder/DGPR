MAKEFLAGS += -r -j

CXXFLAGS := -std=c++11 -Isolver -Irender
CUFLAGS  := -std=c++14 -Isolver -DSINGLE_PRECISION

include local.mk

shaders := graphics_vertex.spv graphics_fragment.spv compute_gemm.spv compute_precomp_basis.spv

sol   := $(wildcard solver/*.cpp)
solcu := $(wildcard solver/*.cu)
cuda  := $(wildcard solver/*.cu)
tst   := $(wildcard test/*.cpp)
tstcu := $(wildcard test/*.cu)
rnd   := $(wildcard render/*.cpp)

BIN  := bin

DG     := $(BIN)/dg
CUDG   := $(BIN)/cudg
TEST   := $(BIN)/test
TESTVK := $(BIN)/testvk
TESTCU := $(BIN)/testcu
REND   := $(BIN)/rend
SHDR   := $(patsubst %,$(BIN)/%,$(shaders))

# ui

.PHONY : dg
dg: $(DG)

.PHONY : cudg
cudg: $(CUDG)

.PHONY : rend
rend: $(REND)

.PHONY : test
test: $(TEST)

.PHONY : testvk
testvk: $(TESTVK)

.PHONY : testcu
testcu: $(TESTCU)

.PHONY : clean
clean:
	rm -f $(DG) $(CUDG) $(TEST) $(TESTVK) $(TESTCU) $(REND) $(SHDR)

# main targets

$(DG): $(sol)
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) -o$@ solver/dg.cpp

$(CUDG): $(sol) $(cuda)
	@mkdir -p $(BIN)
	nvcc $(CUFLAGS) -o $@ solver/cudg.cu

$(REND): $(rnd) $(sol) $(SHDR)
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) -DSINGLE_PRECISION -o$@ render/rend.cpp $(VULKAN_LDFLAGS)

$(TEST): $(sol) $(tst)
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) $(TEST_CXXFLAGS) -o$@ test/test.cpp $(TEST_LDFLAGS)

$(TESTVK): $(sol) $(rnd) $(tst) $(SHDR) 
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) $(TEST_CXXFLAGS) -DSINGLE_PRECISION -o$@ test/vkmath_test.cpp $(TEST_LDFLAGS) $(VULKAN_LDFLAGS)

$(TESTCU): $(sol) $(solcu) $(tst) $(tstcu)
	@mkdir -p $(BIN)
	nvcc $(CUFLAGS) $(MPI_CXXFLAGS) $(TEST_CXXFLAGS) -o $@ test/cumath_test.cu $(TEST_LDFLAGS) $(MPI_LDFLAGS)

# shaders

$(BIN)/graphics_vertex.spv: shaders/graphics_vertex.glsl
	glslc -o$@ -fshader-stage=vert $^
$(BIN)/graphics_fragment.spv: shaders/graphics_fragment.glsl
	glslc -o$@ -fshader-stage=frag $^
$(BIN)/compute_precomp_basis.spv: shaders/compute.glsl
	glslc -o$@ -DBASIS -fshader-stage=comp $^
$(BIN)/compute_gemm.spv: shaders/compute.glsl
	glslc -o$@ -DGEMM -fshader-stage=comp $^
