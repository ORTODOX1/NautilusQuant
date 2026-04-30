# NQX-Core: ISA, микроархитектура, тайминг

## 1. Цель

Hardware accelerator для NautilusQuant pipeline:
encode = `T·x → polar → quant → QJL → pack` за один проход через SRAM.

## 2. Системная диаграмма

```
        Host (LLM runtime)
              |
              v   command queue
   +---------------------------+
   |     NQX Frontend          |
   |  Fetch -> Decode -> Issue |
   +-------------+-------------+
                 |
        +--------+--------+
        |   NQX Backend   |
        | (dataflow core) |
        +--------+--------+
                 |
   +-------------+-----------------+
   |                               |
   v   DMA / HBM2e (16 GB/s+)      |
   +-------------------------------+
```

## 3. ISA (NQ-ISA v2)

### 3.1 Форматы

```
 31      24 23      16 15       8 7        0
+---------+---------+---------+---------+
| OPCODE  |  rd/imm |   rs1   |  rs2/im |   "R-form"  (32-bit)
+---------+---------+---------+---------+

 31      24 23           16 15                                0
+---------+----------------+---------------------------------+
| OPCODE  |     reg        |          imm16 / addr           |   "I-form"
+---------+----------------+---------------------------------+
```

`rd, rs1, rs2 ∈ {V0..V15}` (vector regs) или `{S0..S7}` (scalar regs) в
зависимости от opcode. Шифр scalar/vector — старший бит поля.

### 3.2 Список opcodes

| Hex | Mnem      | Форма | Операнды             | Эффект                                                   |
|-----|-----------|-------|----------------------|----------------------------------------------------------|
| 00  | NOP       | R     | —                    | пропустить такт                                          |
| 01  | LDV       | I     | vd, [addr]           | DMA HBM[addr..addr+dim*2] → SRAM_in → vd (FP16→FP32)     |
| 02  | STV       | I     | vs, [addr]           | pack 3+1bit → SRAM_out → DMA → HBM[addr..]               |
| 03  | MOV       | R     | vd, vs               | копировать vector reg                                    |
| 10  | GVNS      | R     | vd, layer_id (imm)   | Givens rotation layer (0=L1, 1=L2, 2=L3)                 |
| 11  | GVNS_INV  | R     | vd, layer_id (imm)   | inverse Givens (negate sin)                              |
| 20  | POLAR     | R     | vd                   | (x,y) -> (r,θ) попарно                                   |
| 21  | IPOLAR    | R     | vd                   | (r,θ) -> (x,y) попарно                                   |
| 30  | QUANT     | R     | vd, bits (imm)       | Lloyd-Max in-place quantize, scale/zero -> S regs        |
| 31  | DEQUANT   | R     | vd, bits, sd, sz     | dequant используя scalar регистры                        |
| 40  | QJL       | R     | vd_orig, vd_q, alpha | sign-bit correction (alpha — Q1.7 unorm в imm)           |
| 41  | UNQJL     | R     | vd, alpha            | reverse QJL (для decode)                                 |
| 50  | PACK3     | R     | vd, vsign            | упаковать 3+1 бит в SRAM_out                             |
| 51  | UNPACK3   | R     | vd                   | распаковать из SRAM_in                                   |
| 52  | MXPACK    | R     | vd, fmt              | OCP MX block-quant (block=32, fmt: MXFP4/6/8/MXINT8)     |
| 53  | MXUNPACK  | R     | vd, fmt              | dequant из MX-метаданных                                 |
| 54  | SUBBIT_ENC| R     | vd, r_bits, θ_bits   | раздельная квант. radius (r_bits) и angle (θ_bits)       |
| 55  | SUBBIT_DEC| R     | vd                   | reconstruct после SUBBIT_ENC                             |
| 60  | ENC       | I     | [src], [dst], cnt    | макрос: LDV+GVNS×3+POLAR+QUANT+QJL+PACK3+STV (cnt раз)   |
| 61  | DEC       | I     | [src], [dst], cnt    | обратный макрос                                          |
| 70  | BARRIER   | R     | —                    | дождаться завершения всех in-flight инструкций           |
| 7F  | HALT      | R     | —                    | остановить ядро                                          |
| 80  | ATTN_DOT  | R     | vq, vk               | dot-product в polar домене → CPU.last_attn_dot, S0       |

### 3.3 Регистровые файлы

- **VRF** (Vector Register File): 16 × `dim` × FP32 = 16 × 512 B = **8 KB** для dim=128
- **SRF** (Scalar Register File): 8 × FP32 = 32 B (хранят scale, zero-point, mins, maxs)
- **PC** (Program Counter): 16-bit
- **CSR** (Control/Status):
  - `cycle_count` — счётчик тактов
  - `dim` — текущая размерность вектора (128/256/...)
  - `bits` — точность квантизации (по умолчанию 3)
  - `phi` — константа золотого сечения (фиксирована в ROM)

### 3.4 Адресное пространство

| Регион     | Базовый адрес | Размер  | Доступ            |
|------------|---------------|---------|-------------------|
| HBM        | 0x0000_0000   | до 16 GB| DMA, через LDV/STV |
| SRAM_in    | 0x1000_0000   | 24 KB   | внутренний        |
| SRAM_out   | 0x1001_0000   | 24 KB   | внутренний        |
| LUT_ROM    | 0x2000_0000   | 4 KB    | read-only         |
| MMIO/CSR   | 0x3000_0000   | 4 KB    | read-write        |

#### MMIO performance counters (`0x3000_0000` … `0x3000_001C`)

| Address     | Counter                | Width | Notes |
|-------------|------------------------|------:|-------|
| 0x3000_0000 | `cycle_count`          | 32    | total executed cycles |
| 0x3000_0004 | `stall_cycles`         | 32    | DMA-wait cycles inserted by `BARRIER` |
| 0x3000_0008 | `gu_busy_cycles`       | 32    | Givens unit active cycles |
| 0x3000_000C | `pu_busy_cycles`       | 32    | Polar unit active cycles |
| 0x3000_0010 | `qu_busy_cycles`       | 32    | Quant unit active cycles |
| 0x3000_0014 | `dma_in_bytes`         | 32    | bytes pulled from HBM via LDV/LDV_ASYNC |
| 0x3000_0018 | `dma_out_bytes`        | 32    | bytes pushed to HBM via STV |
| 0x3000_001C | `prng_cycles_baseline` | 32    | hypothetical PRNG cost = `4 × dim × dim` (for honest random comparison) |

Software mirrors counters into S0..S7 via `core.perf.write_to_srf(srf)`.

## 4. Микроархитектура

### 4.1 Dataflow pipeline

```
Stage:    F     D    IS   GU.L1   GU.L2  GU.L3   PU    QU.r   QU.q   QJL  PACK  WB/STV
                                                          ^
                                                          | min/max
                                                          | reduce 7c
```

- Каждая **vector-stage** обрабатывает 128 FP32 элементов параллельно
- 64 SIMD lane × 2 elements per lane = 128 wide
- Steady-state throughput: **1 вектор на такт** (после prefill)
- Pipeline depth (latency первой иттерации): **18 тактов** при dim=128

### 4.2 Functional Units

#### Givens Unit (GU)

```
                +---+    +---+
   in[i] ---*---|×c |--+ |   |   +---+
            |   +---+  |-|+/-|---|out[i]|
            |   +---+  |-|   |   +---+
   in[j] -*-|---|×s |--+ +---+
          | |   +---+
          | |   +---+
          | +---|×c |--+      +---+
          |     +---+  |  +---|out[j]|
          |     +---+  |--|+|--+
          +-----|×s |--+  +---+
                +---+
```

- 4 multipliers + 2 adders per lane = 6 ops/cycle/lane
- 64 lanes => **384 ops/cycle/layer**
- 3 layer-stages подряд = **1152 ops/cycle**

Lookup `(c, s)` для пары `k` приходит из ROM_LUT в том же такте через
broadcast bus. Пары и их углы предкомпилированы для каждого слоя
(L1: adjacent, L2: shifted-by-1, L3: butterfly stride dim/4).

#### Polar Unit (PU)

Per pair `(2k, 2k+1)`:
- `r = sqrt(x² + y²)` — fixed CORDIC sqrt, 4 cycles, pipelined
- `θ = atan2(y, x)` — CORDIC vectoring mode, 8 stages, pipelined

Throughput: 1 пара/такт/lane × 64 lanes = 64 пары/такт = весь vector за 1 такт steady state.

#### Quantizer Unit (QU)

Stage QU.r (range): in-place min/max reduce по batch (не по элементам вектора —
квантуем per-feature по batch, как в reference). Дерево compare 7-уровневое =
log₂(128 features) = 7 тактов, но batched через все элементы.

Stage QU.q (round): `q = round((x - min) / range × (2^bits - 1))`. 1 такт.

#### QJL Unit

`error = original - dequant; sign = (error >= 0); corrected = dequant + sign·|error|·α`.

α — фиксированный или через imm. 1 такт (sign + mul-add).

#### Pack Unit

3-bit q-values + 1-bit sign packed bit-serially. 1 такт.

### 4.3 ROM_LUT layout

Для каждого слоя L ∈ {1,2,3}:
- `pair_i[k]`, `pair_j[k]`: int8 индексы (`i,j ∈ [0, dim)`)
- `cos[k]`, `sin[k]`: FP32

Всего для dim=128:
- L1: 64 пар × (2 idx + 2 trig) = 64 × 10 B = 640 B
- L2: 63 × 10 = 630 B
- L3: ≤32 × 10 = 320 B
- Forward + inverse (но inverse = negate sin runtime, ROM один)
- **Итого: ≤2 KB**

## 5. Энергомодель

| Операция                  | pJ        | Источник                          |
|---------------------------|-----------|-----------------------------------|
| FP32 multiply             | 3.7 pJ    | Horowitz 2014, 45nm               |
| FP32 add                  | 0.9 pJ    |                                   |
| HBM2e read (1 byte)       | 5 pJ      | NVIDIA H100 (per-byte amortized)  |
| HBM2e write (1 byte)      | 5 pJ      |                                   |
| SRAM (read + write 1 B)   | 0.05 pJ   |                                   |
| ROM read                  | 0.02 pJ   |                                   |

Стоимость одного encode (dim=128):
- HBM in: 256 B × 5 = 1280 pJ = **1.28 nJ**
- HBM out: 64 B × 5 = 320 pJ = **0.32 nJ** (4× компрессия)
- Compute (3 layers × 64 pairs × 6 ops × ~3 pJ): ≈ 3.5 nJ
- **Total: ≈ 5.1 nJ/vector**

Сравнение с FP16 без сжатия (только trip туда-обратно):
- 256 B × 2 × 5 = 2.56 nJ — но это **на каждое чтение** в attention loop
- Поверх 100 chunks attention reads: **256 nJ vs ~32 nJ** = 8× выигрыш

## 6. Cycle accuracy

Pipeline diagram для batch из 4 векторов (dim=128):

```
cycle:    0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21
vec 0:   F  D  I  L1 L2 L3 P  Qr Qr Qr Qr Qr Qr Qr Qq QJL PK WB
vec 1:      F  D  I  L1 L2 L3 P  Qr Qr Qr Qr Qr Qr Qr Qq QJL PK WB
vec 2:         F  D  I  L1 L2 L3 P  Qr ...                    PK WB
vec 3:            F  D  I  L1 L2 L3 P                            PK WB
```

Latency vec 0: 18 cycles. Throughput steady: 1/cycle. Для 4 vec: 21 cycles.
Для N vec (N >> depth): ≈ N cycles. **Throughput-bound на DMA HBM**, не на compute.

## 7. Программная модель

```nqasm
; Полный encode batch
LDV V0, [0x0]           ; load N FP16 vectors → V0 (FP32)
GVNS V0, 0              ; Layer 1 forward
GVNS V0, 1              ; Layer 2
GVNS V0, 2              ; Layer 3
POLAR V0                ; (x,y) → (r,θ)
QUANT V0, 3             ; 3-bit Lloyd-Max, scale → S0, zero → S1
QJL V1, V0, 0x80        ; α = 0.5 (Q1.7 = 0x80)
PACK3 V1, V_SIGN        ; → SRAM_out
STV V1, [0x10000000]    ; → HBM packed
HALT
```

В макроформе всё то же = `ENC [src], [dst], N`.

## 8. NQ-ISA v2 расширения

### 8.1 MX-форматы (Concept 3)

OCP Microscaling: блок из `block_size=32` элементов делит один 8-bit shared
exponent. Поддержка форматов `MXFP4` (E1M2), `MXFP6` (E2M3), `MXFP8` (E4M3),
`MXINT8`. Биты на значение = `total_bits + 8/block_size`.

```nqasm
; MXFP4 квантование вектора
LDV V0, [0x0]
MXPACK V0, MXFP4         ; in-place dequant + meta+SRAM_out
MXUNPACK V0, MXFP4       ; восстановление
HALT
```

`fmt` может быть мнемоникой (`MXFP4`/`MXFP6`/`MXFP8`/`MXINT8`) или индексом 0..3.

### 8.2 Sub-bit квантование (Concept 4)

После `POLAR` отдельно квантуем radius (`r_bits`) и angle (`θ_bits`). Углы
концентрированы за счёт золотого сечения, поэтому 1-2 бит достаточно.

```nqasm
; r=3 бит, θ=1 бит
LDV V0, [0x0]
GVNS V0, 0
GVNS V0, 1
GVNS V0, 2
POLAR V0
SUBBIT_ENC V0, 3, 1
SUBBIT_DEC V0
HALT
```

`bits_per_value = (r_bits + θ_bits) / 2`. Compression vs FP16 = `16 / bits_per_value`.

### 8.3 Attention-fused (Concept 5)

`ATTN_DOT Vq, Vk` считает `<q, k>` без `IPOLAR`-обратного:
для пары `(2k, 2k+1)` в polar `q[i]·k[i] + q[i+1]·k[i+1] = r_q·r_k·cos(θ_q − θ_k)`.

```nqasm
; Q и K уже в polar (после ENC->dequantize пути)
LDV V0, [0x0000_0000]    ; Q vectors
LDV V1, [0x0001_0000]    ; K vectors
POLAR V0
POLAR V1
ATTN_DOT V0, V1          ; → core.last_attn_dot, scalar в S0
HALT
```

Результат — матрица `(n_q, n_k)`, доступная через `core.last_attn_dot`. Скаляр
(или первые `dim` элементов) дублируется в `S0`.

## 9. Полная NQ-ISA v2 opcode table

| Hex | Mnem        | Форма | Стадия pipeline             |
|-----|-------------|-------|-----------------------------|
| 00  | NOP         | R     | ctrl                        |
| 01  | LDV         | I     | DMA in                      |
| 02  | STV         | I     | DMA out                     |
| 03  | MOV         | R     | VRF copy                    |
| 10  | GVNS        | R     | GU.Lk forward               |
| 11  | GVNS_INV    | R     | GU.Lk inverse               |
| 20  | POLAR       | R     | PU.cart→polar               |
| 21  | IPOLAR      | R     | PU.polar→cart               |
| 30  | QUANT       | R     | QU.minmax + round           |
| 31  | DEQUANT     | R     | QU.round                    |
| 40  | QJL         | R     | QJL fused                   |
| 41  | UNQJL       | R     | reverse QJL                 |
| 50  | PACK3       | R     | PACK 3+1 bit                |
| 51  | UNPACK3     | R     | UNPACK 3+1                  |
| 52  | MXPACK      | R     | MX block quant              |
| 53  | MXUNPACK    | R     | MX dequant                  |
| 54  | SUBBIT_ENC  | R     | radius+angle split-quant    |
| 55  | SUBBIT_DEC  | R     | reconstruct polar           |
| 60  | ENC         | I     | macro encode pipeline       |
| 61  | DEC         | I     | macro decode pipeline       |
| 70  | BARRIER     | R     | sync                        |
| 7F  | HALT        | R     | stop                        |
| 80  | ATTN_DOT    | R     | Polar dot fused             |
