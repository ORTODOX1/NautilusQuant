VERILATOR ?= verilator
VFLAGS    ?= --binary --build -Wall -Wno-fatal -j 0 --top-module tb_nqx
SV_SRC    := \
    givens_lane.sv \
    golden_rom.sv  \
    polar_unit.sv  \
    quant_unit.sv  \
    nqx_top.sv     \
    tb_nqx.sv

PYTHON    ?= python3
ROM_MEM   := golden_rom.mem
DUMP_HEX  := python_dump.hex

.PHONY: all sim rom dump clean

all: sim

rom: $(ROM_MEM)

$(ROM_MEM): ../tools/gen_rom.py
	$(PYTHON) ../tools/gen_rom.py --dim 128 --out $(ROM_MEM) --verify

dump: $(DUMP_HEX)

$(DUMP_HEX): ../tools/dump_for_rtl.py
	$(PYTHON) ../tools/dump_for_rtl.py --dim 128 --out $(DUMP_HEX)

sim: rom dump $(SV_SRC)
	$(VERILATOR) $(VFLAGS) $(SV_SRC)
	./obj_dir/Vtb_nqx

clean:
	rm -rf obj_dir $(ROM_MEM) $(DUMP_HEX)
