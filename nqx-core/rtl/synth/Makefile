YOSYS ?= yosys

.PHONY: synth clean

synth:
	$(YOSYS) -q -l synth.log synth.ys
	@echo "----"
	@grep -E "Number of (cells|wires|wire bits)" synth.log || true
	@echo "----"
	@echo "Netlist: nqx_top_synth.v   JSON: nqx_top_synth.json   Log: synth.log"

clean:
	rm -f nqx_top_synth.v nqx_top_synth.json synth.log
