// SPDX-License-Identifier: GPL-2.0-only
/* NQX-Core PCIe accelerator driver skeleton.
 *
 * PCIe device probe/remove, BAR MMIO mapping, char device /dev/nqx0.
 * Skeleton — intended for development and verification, not production.
 */

#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/fs.h>
#include <linux/io.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/pci.h>
#include <linux/slab.h>
#include <linux/uaccess.h>

#define NQX_PCI_VENDOR 0x1d6b  /* Linux Foundation (placeholder) */
#define NQX_PCI_DEVICE 0x0001
#define NQX_DRIVER_NAME "nqx"
#define NQX_CLASS_NAME "nqx"
#define NQX_MAX_DEVS   4

/* MMIO register offsets (see docs/architecture.md §3.4) */
#define NQX_REG_CTRL     0x00
#define NQX_REG_STATUS   0x04
#define NQX_REG_COMMAND  0x08
#define NQX_REG_DATA     0x0C

struct nqx_dev {
    struct pci_dev    *pdev;
    void __iomem      *bar;
    unsigned long      bar_len;
    struct cdev        cdev;
    struct device     *device;
    int                dev_id;
};

static dev_t nqx_dev_num;
static struct class *nqx_class;
static struct nqx_dev *nqx_devices[NQX_MAX_DEVS];
static int nqx_dev_count;

/* Char device ops */
static int nqx_open(struct inode *inode, struct file *filp)
{
    struct nqx_dev *nd = container_of(inode->i_cdev, struct nqx_dev, cdev);
    filp->private_data = nd;
    return 0;
}

static int nqx_release(struct inode *inode, struct file *filp)
{
    return 0;
}

static ssize_t nqx_read(struct file *filp, char __user *buf,
                         size_t count, loff_t *f_pos)
{
    struct nqx_dev *nd = filp->private_data;
    u32 val;
    if (*f_pos >= 4)
        return 0;
    val = ioread32(nd->bar + NQX_REG_DATA);
    if (copy_to_user(buf, &val, sizeof(val)))
        return -EFAULT;
    *f_pos += sizeof(val);
    return sizeof(val);
}

static ssize_t nqx_write(struct file *filp, const char __user *buf,
                          size_t count, loff_t *f_pos)
{
    struct nqx_dev *nd = filp->private_data;
    u32 val;
    if (count < sizeof(val))
        return -EINVAL;
    if (copy_from_user(&val, buf, sizeof(val)))
        return -EFAULT;
    iowrite32(val, nd->bar + NQX_REG_DATA);
    return sizeof(val);
}

static long nqx_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)
{
    struct nqx_dev *nd = filp->private_data;
    switch (cmd) {
    case 0x00: /* NQX_IOCTL_RESET — reset accelerator */
        iowrite32(0x01, nd->bar + NQX_REG_CTRL);
        return 0;
    case 0x01: /* NQX_IOCTL_STATUS — read status register */
    {
        u32 status = ioread32(nd->bar + NQX_REG_STATUS);
        if (copy_to_user((void __user *)arg, &status, sizeof(status)))
            return -EFAULT;
        return 0;
    }
    default:
        return -ENOTTY;
    }
}

static const struct file_operations nqx_fops = {
    .owner          = THIS_MODULE,
    .open           = nqx_open,
    .release        = nqx_release,
    .read           = nqx_read,
    .write          = nqx_write,
    .unlocked_ioctl = nqx_ioctl,
};

/* PCIe probe */
static int nqx_pci_probe(struct pci_dev *pdev, const struct pci_device_id *ent)
{
    struct nqx_dev *nd;
    int ret;

    if (nqx_dev_count >= NQX_MAX_DEVS)
        return -ENOMEM;

    ret = pcim_enable_device(pdev);
    if (ret < 0)
        return ret;

    ret = pcim_iomap_regions(pdev, BIT(0), NQX_DRIVER_NAME);
    if (ret < 0)
        return ret;

    nd = devm_kzalloc(&pdev->dev, sizeof(*nd), GFP_KERNEL);
    if (!nd)
        return -ENOMEM;

    nd->pdev = pdev;
    nd->bar = pcim_iomap_table(pdev)[0];
    nd->bar_len = pci_resource_len(pdev, 0);
    nd->dev_id = nqx_dev_count;

    dev_info(&pdev->dev, "NQX bar len=%lu bytes\n", nd->bar_len);

    /* Create char device */
    cdev_init(&nd->cdev, &nqx_fops);
    nd->cdev.owner = THIS_MODULE;
    ret = cdev_add(&nd->cdev, nqx_dev_num + nd->dev_id, 1);
    if (ret < 0) {
        dev_err(&pdev->dev, "cdev_add failed: %d\n", ret);
        return ret;
    }

    nd->device = device_create(nqx_class, &pdev->dev,
                                nqx_dev_num + nd->dev_id, nd, "nqx%d", nd->dev_id);
    if (IS_ERR(nd->device)) {
        cdev_del(&nd->cdev);
        return PTR_ERR(nd->device);
    }

    pci_set_drvdata(pdev, nd);
    nqx_devices[nqx_dev_count++] = nd;

    dev_info(&pdev->dev, "/dev/nqx%d ready\n", nd->dev_id);
    return 0;
}

static void nqx_pci_remove(struct pci_dev *pdev)
{
    struct nqx_dev *nd = pci_get_drvdata(pdev);
    if (!nd)
        return;
    device_destroy(nqx_class, nqx_dev_num + nd->dev_id);
    cdev_del(&nd->cdev);
    nqx_devices[nd->dev_id] = NULL;
    dev_info(&pdev->dev, "removed\n");
}

static const struct pci_device_id nqx_pci_ids[] = {
    { PCI_DEVICE(NQX_PCI_VENDOR, NQX_PCI_DEVICE) },
    { 0, }
};
MODULE_DEVICE_TABLE(pci, nqx_pci_ids);

static struct pci_driver nqx_pci_driver = {
    .name     = NQX_DRIVER_NAME,
    .id_table = nqx_pci_ids,
    .probe    = nqx_pci_probe,
    .remove   = nqx_pci_remove,
};

static int __init nqx_init(void)
{
    int ret;

    ret = alloc_chrdev_region(&nqx_dev_num, 0, NQX_MAX_DEVS, NQX_DRIVER_NAME);
    if (ret < 0)
        return ret;

    nqx_class = class_create(THIS_MODULE, NQX_CLASS_NAME);
    if (IS_ERR(nqx_class)) {
        unregister_chrdev_region(nqx_dev_num, NQX_MAX_DEVS);
        return PTR_ERR(nqx_class);
    }

    ret = pci_register_driver(&nqx_pci_driver);
    if (ret < 0) {
        class_destroy(nqx_class);
        unregister_chrdev_region(nqx_dev_num, NQX_MAX_DEVS);
    }
    return ret;
}

static void __exit nqx_exit(void)
{
    pci_unregister_driver(&nqx_pci_driver);
    class_destroy(nqx_class);
    unregister_chrdev_region(nqx_dev_num, NQX_MAX_DEVS);
}

module_init(nqx_init);
module_exit(nqx_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("NautilusQuant");
MODULE_DESCRIPTION("NQX-Core PCIe accelerator driver skeleton");
