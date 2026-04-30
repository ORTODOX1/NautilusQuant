# NQX boot ROM build
ROOT = ../../

all: boot.bin boot.hex

boot.bin: boot.nqasm
	cd $(ROOT) && python -c "from nqx.assembler import assemble; prog = assemble(open('firmware/boot/boot.nqasm').read()); out = bytes([b for i in prog for b in i.encode()]); open('firmware/boot/boot.bin','wb').write(out); print(f'assembled {len(prog)} insns -> {len(out)} bytes')"

boot.hex: boot.bin
	xxd -p boot.bin > boot.hex

clean:
	rm -f boot.bin boot.hex

test:
	python -m pytest firmware/boot/test_boot.py -v

.PHONY: all clean test
