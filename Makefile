MAKEFLAGS += -r -j

CXXFLAGS := -std=c++11 -Isolver -Irender
CUFLAGS  := -std=c++14 -Isolver -DSINGLE_PRECISION

include local.mk

shaders = graphics_vertex.glsl graphics_fragment.glsl compute_precomp_sol_basis.glsl compute_test.glsl

sol  := $(wildcard solver/*.cpp)
cuda := $(wildcard solver/*.cu)
tst  := $(wildcard test/*.cpp)
rnd  := $(wildcard render/*.cpp)
shdr := $(patsubst %,shaders/%,$(shaders))

BIN  := bin

DG   := $(BIN)/dg
CUDG := $(BIN)/cudg
TEST := $(BIN)/test
REND := $(BIN)/rend
SHDR := $(patsubst shaders/%.glsl,$(BIN)/%.spv,$(shdr))

# ui

.PHONY : dg
dg: $(DG)

.PHONY : cudg
cudg: $(CUDG)

.PHONY : rend
rend: $(REND)

.PHONY : test
test: $(TEST)

.PHONY : clean
clean:
	rm -f $(DG) $(CUDG) $(TEST) $(REND) $(SHDR)

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

# shaders

$(BIN)/graphics_vertex.spv: shaders/graphics_vertex.glsl
	glslc -o$@ -fshader-stage=vert $^
$(BIN)/graphics_fragment.spv: shaders/graphics_fragment.glsl
	glslc -o$@ -fshader-stage=frag $^
$(BIN)/compute_precomp_sol_basis.spv: shaders/compute_precomp_sol_basis.glsl
	glslc -o$@ -fshader-stage=comp -Ishaders $^
$(BIN)/compute_test.spv: shaders/compute_test.glsl
	glslc -o$@ -fshader-stage=comp $^
