# NQX Linux kernel driver

Skeleton PCIe driver for the NQX accelerator.

## Build

```bash
cd firmware/driver
make                    # builds against running kernel
```

To cross-compile against a custom kernel tree:

```bash
make KDIR=/path/to/linux/build
```

## Usage

```bash
make install   # insmod + chmod /dev/nqx0
ls -l /dev/nqx0
cat /dev/nqx0  # read status register
```

## Interface

| Operation | Description |
|-----------|-------------|
| `open` | Opens `/dev/nqxN` |
| `read` | Reads 4 bytes from DATA register |
| `write` | Writes 4 bytes to DATA register |
| `ioctl 0x00` | Reset accelerator (CTRL=1) |
| `ioctl 0x01` | Read STATUS register into arg |

## DKMS

```bash
sudo dkms add .
sudo dkms install nqx_driver/1.0
```

## Kernel API dependencies

- `pci_register_driver` / `pcim_enable_device`
- `devm_kzalloc` / `pcim_iomap_regions`
- `cdev` / `class_create` / `device_create`
- Compatible with Linux 5.10+ (tested on 6.x)
