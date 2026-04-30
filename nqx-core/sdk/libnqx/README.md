# libnqx — C ABI for NQX-Core

Prototype C-language ABI for integrating NQX acceleration into C/C++ applications.

## Files

| File | Purpose |
|------|---------|
| `libnqx.h` | C header with function declarations and struct types |
| `libnqx.py` | Python implementation backed by `NQXCore` |
| `test_abi.py` | pytest: open/close, encode/decode roundtrip, version |

## Usage (Python prototype)

```python
from sdk.libnqx.libnqx import nqx_open, nqx_encode, nqx_decode, nqx_close
import numpy as np

hid = nqx_open('{"dim": 128, "bits": 3}')
x = np.random.standard_normal((8, 128)).astype(np.float32)
enc = nqx_encode(hid, x)
dec = nqx_decode(hid, enc["packed"], enc["sign_bits"], enc["mins"], enc["maxs"],
                 enc["n"], enc["dim"], enc["bits"])
nqx_close(hid)
```

## Future: real .so

Planned via pybind11 or ctypes:
```c
gcc my_app.c -lnqx -o my_app
```

The header `libnqx.h` is the stable API — the implementation can switch from Python
to C/Rust without changing callers.
