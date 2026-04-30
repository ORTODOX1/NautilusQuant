```
NQX-Core encode latency:  min 2.49 ms  max 4.75 ms  jitter 2.26 ms
    2.49 ms |████████████████████████ 6
    2.68 ms |████████████████████████████████████████ 10
    2.86 ms |████████████████ 4
    3.05 ms |████████ 2
    3.24 ms |████████████ 3
    3.43 ms |████████████ 3
    3.62 ms |█ 0
    3.81 ms |████████████████ 4
    3.99 ms |████████ 2
    4.18 ms |████████████████████ 5
    4.37 ms |████████████████████████ 6
    4.56 ms |████████████████████ 5

TurboQuant encode latency:  min 5.90 ms  max 15.66 ms  jitter 9.75 ms
    5.90 ms |██████████████████████████████████████ 15
    6.72 ms |████████████ 5
    7.53 ms |██████████████████ 7
    8.34 ms |█████ 2
    9.16 ms |████████████████████████████████████████ 16
    9.97 ms |██████████ 4
   10.78 ms |█ 0
   11.59 ms |█ 0
   12.41 ms |█ 0
   13.22 ms |█ 0
   14.03 ms |█ 0
   14.84 ms |██ 1

Cycle breakdown by stage (lower is better)

Stage         NQX  Turbo  scaled bar
------------------------------------------------------------
load            1      1  N|█
                          T|█
pack            1      1  N|█
                          T|█
polar           1      1  N|█
                          T|█
prng            0    655  N|█
                          T|████████████████
qjl             1      1  N|█
                          T|█
quant           8      8  N|█
                          T|█
rotate          3    128  N|█
                          T|███
store           1      1  N|█
                          T|█

Total cycles  NQX = 16  Turbo = 796 (scaled)

Gantt — pipelined NQX-Core encode

         01234567890123456
vec  0  █████████        
vec  1   █████████       
vec  2    █████████      
vec  3     █████████     
vec  4      █████████    
vec  5       █████████   
vec  6        █████████  
vec  7         █████████ 

Stages: LDV → GVNS.L1 → GVNS.L2 → GVNS.L3 → POLAR → QUANT → QJL → PACK → STV
```
