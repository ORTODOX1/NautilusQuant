# NQX Boot ROM

Bringup firmware that runs on NQX core after reset.

## Sequence

1. **BARRIER** — wait for any pending DMA
2. **Clear VRF** — load zeros into all 16 vector registers
3. **Orthogonality self-test** — encode test vector
4. **HALT** — boot complete

## Build

```bash
cd firmware/boot
make          # → boot.bin + boot.hex
make test     # assemble + run on emulator
```

## Boot protocol

| Step | Address | Data |
|------|---------|------|
| Reset | — | PC = 0 |
| Boot ROM | 0x0000 | Execute boot.nqasm |
| HALT | — | PC stops, STATUS = 0x1 |
| Host reads | STATUS | Checks boot OK |
