SBY ?= sby

.PHONY: formal bmc prove clean

formal: bmc

bmc:
	$(SBY) -f orthogonality.sby

prove:
	$(SBY) -f induction.sby

clean:
	rm -rf orthogonality induction
