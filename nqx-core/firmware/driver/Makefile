# SPDX-License-Identifier: GPL-2.0-only
# NQX kernel driver — cross-compile against Linux headers.
# Usage:
#   make           # build against running kernel
#   make KDIR=/path/to/kernel  # custom kernel tree

obj-m += nqx_driver.o

KDIR ?= /lib/modules/$(shell uname -r)/build
PWD  := $(shell pwd)

all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean

install:
	sudo insmod nqx_driver.ko || true
	sudo chmod 666 /dev/nqx0 2>/dev/null || true

remove:
	sudo rmmod nqx_driver 2>/dev/null || true
