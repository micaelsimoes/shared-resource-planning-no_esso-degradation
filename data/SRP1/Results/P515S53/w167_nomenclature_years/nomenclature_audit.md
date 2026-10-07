# W167 Task A -- nomenclature audit of the submitted main.tex

main.tex sha256 `3cedbb6e37bfe31bf7812a76fcc93e7030053e9c7b80793a454ba7e11c965e19` (clone `manuscript/6a67305f25e8348fb71380c3`); nomenclature lines 185-267. Report only; nothing edited. Line numbers are main.tex lines.

## Normalisation

identity key = base + "^{label}" + "_{label}". Font wrappers (\text, \mathrm, \textrm, \mathit, \operatorname) removed; bold (\boldsymbol, \mathbf, \bm) removed; \textcolor transparent; \mathcal{X} -> cal(X); \mathbb -> excluded; decorations kept in the base (hat(S); widehat=hat, overline=bar, widetilde=tilde). Superscript: whitespace and braces removed, iteration markers removed ("(l)", "(k)", "(\ell)" and +-1, nested "^k"/"^{k+1}"/"^{\ell}", comma items k, l, \ell, k+-1, l+-1, \ell+-1 and digit strings such as 0 or 1 (LB^0, \hat{x}^1: iteration counters)); \max -> max. Subscript: dropped (index) unless it starts with an upper-case letter (\Omega_C) or is all digits (y_0). Letter runs split into single letters except SoH, SoC, LB, UB, CL, DSO, TSO, ESSO, NPV, EFC, DoD. An atom is an INDEX (reported separately, not as an undefined symbol) when its base is one of the letters this manuscript uses as indices or counters (c d e g i j k l m n o s t y, \ell), it is not bold, has no superscript label and no subscript at all; so x, u, f, h, r are symbols and g_i, x_{ij} are symbols. A label subscript (\Omega_C) is not scanned for atoms; an index subscript is. Scripts are also scanned: every subscript, and a superscript that is an expression (contains + - _ ( \times \cdot after iteration markers are removed), contribute their own atoms (e.g. y^{Inv}, y_0 inside indices).

## Counts

- nomenclature_entries: 34
- distinct_nomenclature_keys: 31
- body_atoms: 1370
- distinct_body_keys: 136
- distinct_body_symbol_keys: 124
- i_defined_not_used: 0
- ii_used_not_defined_all: 105
- ii_used_not_defined_by_category: {'symbol': 89, 'index': 12, 'decorated variant of a defined symbol': 4}
- iii_duplicate_nomenclature_keys: 3
- iii_index_subscript_overloads_of_defined_keys: 6
- iii_declared_overloads: 12
- iii_declared_conflicts: 5
- iii_same_description_different_keys: 1
- symbols_only_inside_algorithm1: 25
- iv_benders_keys_with_occurrences: 18
- v_required_items: 19
- v_required_used_in_paragraphs_v5: 16

## (i) Defined in the nomenclature, never used in the text

| key | nomenclature line | tex | description |
|---|---|---|---|

## (ii) Used in the text, not in the nomenclature

Category `symbol` is the substantive list; `index` are bound indices; the variant categories are forms of a defined symbol. Every line is in nomenclature_audit.json.

### symbol (89)

| key | first line | section of first use | n | lines (first 15) | raw forms (first 4) |
|---|---|---|---|---|---|
| `cal(X)` | 334 | subsection Planning--Operation Decomposition Architecture | 1 | 334 | `cal(X)` |
| `C^{Inv}` | 337 | subsection Planning--Operation Decomposition Architecture | 5 | 337, 364, 416, 424, 512 | `C^{\mathrm{Inv}}`; `C^{\mathrm{Inv}}_{c}`; `C^{\text{Inv}}_{e,y^\text{Inv}}` |
| `x` | 337 | subsection Planning--Operation Decomposition Architecture | 15 | 337, 339, 341, 352, 353, 363, 378, 383, 407, 414, 416, 424, 1712, 1713 | `bold x`; `x`; `x_{ij}` |
| `Q` | 339 | subsection Planning--Operation Decomposition Architecture | 2 | 339, 352 | `Q` |
| `C^{Salvage}` | 341 | subsection Planning--Operation Decomposition Architecture | 2 | 341, 363 | `C^{\text{Salvage}}` |
| `C^{Op}` | 352 | subsection Planning--Operation Decomposition Architecture | 1 | 352 | `C^{\mathrm{Op}}_{o}` |
| `u` | 352 | subsection Planning--Operation Decomposition Architecture | 4 | 352, 353, 379, 383 | `bold u_{o}` |
| `cal(U)` | 353 | subsection Planning--Operation Decomposition Architecture | 1 | 353 | `cal(U)_{o}` |
| `E` | 363 | subsection Planning--Operation Decomposition Architecture | 2 | 363, 1528 | `E` |
| `y^{Inv}` | 363 | subsection Planning--Operation Decomposition Architecture | 89 | 363, 364, 365, 366, 585, 591, 599, 752, 755, 759, 764, 769, 773, 776, 779 ... | `y^{\text{Inv}}` |
| `y^{End}` | 364 | subsection Planning--Operation Decomposition Architecture | 2 | 364, 366 | `y^{\text{End}}` |
| `y_{0}` | 364 | subsection Planning--Operation Decomposition Architecture | 8 | 364, 560, 585, 591, 640, 769, 773, 865 | `y_{0}` |
| `T^{Rem}` | 365 | subsection Planning--Operation Decomposition Architecture | 1 | 365 | `T^{\text{Rem}}_{e,y^\text{Inv}}` |
| `cal(E)` | 402 | subsection Planning--Operation Decomposition Architecture | 2 | 402, 425 | `cal(E)` |
| `cal(Y)` | 402 | subsection Planning--Operation Decomposition Architecture | 4 | 402, 426, 443, 473 | `cal(Y)` |
| `cal(D)` | 402 | subsection Planning--Operation Decomposition Architecture | 3 | 402, 443, 474 | `cal(D)` |
| `cal(C)` | 402 | subsection Planning--Operation Decomposition Architecture | 2 | 402, 427 | `cal(C)` |
| `Omega_{M}` | 402 | subsection Planning--Operation Decomposition Architecture | 2 | 402, 476 | `Omega_{M}` |
| `alpha` | 414 | subsection Planning--Operation Decomposition Architecture | 7 | 414, 416, 488, 569, 577, 689, 718 | `alpha`; `alpha^{(l)}` |
| `gamma` | 428 | subsection Planning--Operation Decomposition Architecture | 2 | 428, 475 | `gamma_{y}` |
| `c^{S}` | 430 | subsection Planning--Operation Decomposition Architecture | 1 | 430 | `c^{S}_{y,c}` |
| `c^{E}` | 432 | subsection Planning--Operation Decomposition Architecture | 1 | 432 | `c^{E}_{y,c}` |
| `LB` | 436 | subsection Planning--Operation Decomposition Architecture | 4 | 436, 519, 525, 531 | `LB^{0}`; `LB^{\ell}` |
| `UB` | 436 | subsection Planning--Operation Decomposition Architecture | 6 | 436, 510, 511, 525, 526, 531 | `UB^{0}`; `UB^{\ell-1}`; `UB^{\ell}` |
| `hat(x)` | 437 | subsection Planning--Operation Decomposition Architecture | 5 | 437, 440, 479, 512, 520 | `hat(x)^{1}`; `hat(x)^{\ell+1}`; `hat(x)^{\ell}` |
| `ell^{max}` | 439 | subsection Planning--Operation Decomposition Architecture | 1 | 439 | `ell^{\max}` |
| `r^{max}` | 448 | subsection Planning--Operation Decomposition Architecture | 1 | 448 | `r^{\max}` |
| `hat(Q)` | 472 | subsection Planning--Operation Decomposition Architecture | 3 | 472, 489, 512 | `hat(Q)^{\ell}` |
| `N` | 475 | subsection Planning--Operation Decomposition Architecture | 1 | 475 | `N_{y}` |
| `W` | 475 | subsection Planning--Operation Decomposition Architecture | 1 | 475 | `W_{d}` |
| `pi` | 478 | subsection Planning--Operation Decomposition Architecture | 2 | 478 | `pi_{m}`; `pi_{o}` |
| `Phi` | 479 | subsection Planning--Operation Decomposition Architecture | 1 | 479 | `Phi_{y,d,m,o}` |
| `g^{S}` | 483 | subsection Planning--Operation Decomposition Architecture | 2 | 483, 492 | `g^{S,\ell}_{e,y}` |
| `g^{E}` | 483 | subsection Planning--Operation Decomposition Architecture | 2 | 483, 500 | `g^{E,\ell}_{e,y}` |
| `gap` | 524 | subsection Planning--Operation Decomposition Architecture | 2 | 524, 529 | `gap^{\ell}` |
| `epsilon^{rel}` | 529 | subsection Planning--Operation Decomposition Architecture | 1 | 529 | `epsilon^{\mathrm{rel}}` |
| `epsilon^{abs}` | 531 | subsection Planning--Operation Decomposition Architecture | 1 | 531 | `epsilon^{\mathrm{abs}}` |
| `mu^{S}` | 692 | subsubsection Benders' Cuts | 3 | 692, 701, 709 | `mu^{S(k)}_{e,y}` |
| `mu^{E}` | 695 | subsubsection Benders' Cuts | 3 | 695, 701, 712 | `mu^{E(k)}_{e,y}` |
| `S^{Up}` | 740 | subsubsection Shared ESS Mathematical Formulation | 3 | 740, 750, 891 | `S^{\text{Up}}_{e,y}` |
| `S^{Down}` | 740 | subsubsection Shared ESS Mathematical Formulation | 3 | 740, 750, 892 | `S^{\text{Down}}_{e,y}` |
| `E^{Up}` | 744 | subsubsection Shared ESS Mathematical Formulation | 3 | 744, 750, 893 | `E^{\text{Up}}_{e,y}` |
| `E^{Down}` | 744 | subsubsection Shared ESS Mathematical Formulation | 3 | 744, 750, 894 | `E^{\text{Down}}_{e,y}` |
| `S^{Rated,Unit}` | 755 | subsubsection Shared ESS Mathematical Formulation | 4 | 755, 764, 769, 779 | `S^{\text{Rated,Unit}}_{e,y^\text{Inv},y}` |
| `E^{Rated,Unit}` | 759 | subsubsection Shared ESS Mathematical Formulation | 5 | 759, 764, 773, 783, 797 | `E^{\text{Rated,Unit}}_{e,y^\text{Inv},y}` |
| `E^{Ch,Dch}` | 796 | subsubsection Shared ESS Mathematical Formulation | 3 | 796, 801, 806 | `E^{\text{Ch,Dch}}_{e,y^\text{Inv},y}` |
| `Delta` | 808 | subsubsection Shared ESS Mathematical Formulation | 2 | 808, 819 | `Delta` |
| `S^{S,Comp}` | 854 | subsubsection Shared ESS Mathematical Formulation | 3 | 854, 859, 897 | `S^{\text{S,Comp}}_{e,y^\text{Inv},y,d,t}` |
| `S^{Net}` | 864 | subsubsection Shared ESS Mathematical Formulation | 2 | 864, 877 | `S^{\text{Net}}_{e,y,d,t}` |
| `E^{Av}` | 1117 | subsection Degradation and Available Energy Capacity | 4 | 1117, 1325, 1328, 1361 | `E^{\text{Av}}_{e,y}` |
| `N^{I}` | 1424 | subsection Updated Algorithm | 2 | 1424, 1441 | `N^{I}` |
| `V` | 1426 | subsection Updated Algorithm | 1 | 1426 | `V_{i,t}` |
| `V^{I}` | 1432 | subsection Updated Algorithm | 1 | 1432 | `V^{I,0}_{i,t}` |
| `P^{I}` | 1432 | subsection Updated Algorithm | 1 | 1432 | `P^{I,0}_{i,t}` |
| `Q^{I}` | 1432 | subsection Updated Algorithm | 1 | 1432 | `Q^{I,0}_{i,t}` |
| `k^{max}` | 1437 | subsection Updated Algorithm | 1 | 1437 | `k^{\text{max}}` |
| `hat(V)^{I}` | 1443 | subsection Updated Algorithm | 2 | 1443, 1465 | `hat(V)^{I^k}_{i,t}` |
| `hat(P)^{I}` | 1444 | subsection Updated Algorithm | 2 | 1444, 1466 | `hat(P)^{I^k}_{i,t}` |
| `hat(Q)^{I}` | 1445 | subsection Updated Algorithm | 2 | 1445, 1467 | `hat(Q)^{I^k}_{i,t}` |
| `hat(P)^{E}` | 1446 | subsection Updated Algorithm | 9 | 1446, 1468, 1486, 1516, 1546, 1558, 1559, 1560, 1580 | `hat(P)^{E^k}_{e, y, d, t}`; `hat(P)^{E^k}_{e,y,d,t}`; `hat(P)^{E^k}_{i,t}`; `hat(P)^{E^{k+1}}_{e,y,d,t}` |
| `hat(Q)^{E}` | 1447 | subsection Updated Algorithm | 9 | 1447, 1469, 1487, 1517, 1546, 1564, 1565, 1566, 1589 | `hat(Q)^{E^k}_{e, y, d, t}`; `hat(Q)^{E^k}_{e,y,d,t}`; `hat(Q)^{E^k}_{i,t}`; `hat(Q)^{E^{k+1}}_{e,y,d,t}` |
| `pi^{I,V}` | 1451 | subsection Updated Algorithm | 2 | 1451, 1473 | `pi^{I,V^k}_{i,t}` |
| `pi^{I,P}` | 1452 | subsection Updated Algorithm | 2 | 1452, 1474 | `pi^{I,P^k}_{i,t}` |
| `pi^{I,Q}` | 1453 | subsection Updated Algorithm | 2 | 1453, 1475 | `pi^{I,Q^k}_{i,t}` |
| `pi^{E,P}` | 1454 | subsection Updated Algorithm | 10 | 1454, 1476, 1491, 1516, 1547, 1558, 1559, 1576, 1577, 1594 | `pi^{E,P^k}_{i,t}`; `pi^{{E,P}^k}_{e, y, d, t}`; `pi^{{E,P}^k}_{e,y,d,t}`; `pi^{{E,P}^{k+1}}_{e,y,d,t}` |
| `pi^{E,Q}` | 1455 | subsection Updated Algorithm | 10 | 1455, 1477, 1492, 1517, 1547, 1564, 1565, 1585, 1586, 1594 | `pi^{E,Q^k}_{i,t}`; `pi^{{E,Q}^k}_{e, y, d, t}`; `pi^{{E,Q}^k}_{e,y,d,t}`; `pi^{{E,Q}^{k+1}}_{e,y,d,t}` |
| `f` | 1515 | subsection ADMM Implementation | 2 | 1515, 1537 | `f` |
| `X` | 1515 | subsection ADMM Implementation | 4 | 1515, 1518, 1533, 1537 | `X` |
| `cal(L)^{E,P}` | 1516 | subsection ADMM Implementation | 3 | 1516, 1541, 1558 | `cal(L)^{E,P}` |
| `P^{E}` | 1516 | subsection ADMM Implementation | 7 | 1516, 1518, 1545, 1558, 1559, 1560, 1580 | `P^{E^k}_{e, y, d, t}`; `P^{E^k}_{e,t}`; `P^{E^k}_{e,y,d,t}`; `P^{E^{k+1}}_{e,y,d,t}` |
| `cal(L)^{E,Q}` | 1517 | subsection ADMM Implementation | 3 | 1517, 1541, 1564 | `cal(L)^{E,Q}` |
| `Q^{E}` | 1517 | subsection ADMM Implementation | 7 | 1517, 1518, 1545, 1564, 1565, 1566, 1589 | `Q^{E^k}_{e, y, d, t}`; `Q^{E^k}_{e,t}`; `Q^{E^k}_{e,y,d,t}`; `Q^{E^{k+1}}_{e,y,d,t}` |
| `h` | 1518 | subsection ADMM Implementation | 2 | 1518, 1551 | `h` |
| `K` | 1532 | subsection ADMM Implementation | 7 | 1532, 1545, 1547, 1570, 1573, 1596 | `K` |
| `rho^{E,P}` | 1560 | subsection ADMM Implementation | 5 | 1560, 1570, 1578, 1599 | `rho^{{E,P}^k}`; `rho^{{E,P}^{k+1}}`; `rho^{{E,P}^{k}}` |
| `rho^{E,Q}` | 1566 | subsection ADMM Implementation | 5 | 1566, 1570, 1587, 1603 | `rho^{{E,Q}^k}`; `rho^{{E,Q}^{k+1}}`; `rho^{{E,Q}^{k}}` |
| `r^{E,P}` | 1599 | subsection ADMM Implementation | 2 | 1599, 1607 | `r^{E,P}` |
| `r^{E,Q}` | 1603 | subsection ADMM Implementation | 2 | 1603, 1607 | `r^{E,Q}` |
| `g` | 1677 | Appendix: section Distribution Networks | 7 | 1677, 1678, 1771 | `g`; `g_{i}` |
| `b` | 1677 | Appendix: section Distribution Networks | 2 | 1677, 1678 | `b_{i}` |
| `V^{Base}` | 1677 | Appendix: section Distribution Networks | 2 | 1677, 1678 | `V^{\text{Base}}_{i}` |
| `V^{max}` | 1677 | Appendix: section Distribution Networks | 2 | 1677, 1678 | `V^{\text{max}}_{i}` |
| `V^{min}` | 1677 | Appendix: section Distribution Networks | 2 | 1677, 1678 | `V^{\text{min}}_{i}` |
| `b^{Sh}` | 1712 | Appendix: section Distribution Networks | 2 | 1712, 1713 | `b^{\text{Sh}}_{ij}` |
| `P^{G,max}` | 1771 | Appendix: section Distribution Networks | 1 | 1771 | `P^{G,\text{max}}_{g}` |
| `P^{G,min}` | 1771 | Appendix: section Distribution Networks | 1 | 1771 | `P^{G,\text{min}}_{g}` |
| `Q^{G,max}` | 1771 | Appendix: section Distribution Networks | 1 | 1771 | `Q^{G,\text{max}}_{g}` |
| `Q^{G,min}` | 1771 | Appendix: section Distribution Networks | 1 | 1771 | `Q^{G,\text{min}}_{g}` |
| `V^{S}` | 1771 | Appendix: section Distribution Networks | 1 | 1771 | `V^{S}_{g}` |

### decorated variant of a defined symbol (4)

| key | first line | section of first use | n | lines (first 15) | raw forms (first 4) |
|---|---|---|---|---|---|
| `hat(S)^{Rated}` | 496 | subsection Planning--Operation Decomposition Architecture | 1 | 496 | `hat(S)^{\mathrm{Rated},\ell}_{e,y}` |
| `hat(E)^{Rated}` | 504 | subsection Planning--Operation Decomposition Architecture | 1 | 504 | `hat(E)^{\mathrm{Rated},\ell}_{e,y}` |
| `hat(S)^{Inv}` | 740 | subsubsection Shared ESS Mathematical Formulation | 2 | 740, 749 | `hat(S)^{\text{Inv}}_{e,y}` |
| `hat(E)^{Inv}` | 744 | subsubsection Shared ESS Mathematical Formulation | 2 | 744, 749 | `hat(E)^{\text{Inv}}_{e,y}` |

### label-subscript variant of a defined symbol (0)

| key | first line | section of first use | n | lines (first 15) | raw forms (first 4) |
|---|---|---|---|---|---|

### index (12)

| key | first line | section of first use | n | lines (first 15) | raw forms (first 4) |
|---|---|---|---|---|---|
| `c` | 336 | subsection Planning--Operation Decomposition Architecture | 22 | 336, 337, 375, 427, 428, 430, 432, 559, 560, 562, 565, 575, 621, 639, 640 ... | `c` |
| `o` | 352 | subsection Planning--Operation Decomposition Architecture | 17 | 352, 353, 377, 379, 383, 477, 478, 479, 733 | `o` |
| `e` | 363 | subsection Planning--Operation Decomposition Architecture | 234 | 363, 364, 365, 366, 407, 408, 425, 430, 432, 483, 491, 492, 494, 496, 499 ... | `e` |
| `y` | 407 | subsection Planning--Operation Decomposition Architecture | 239 | 407, 408, 426, 428, 430, 432, 443, 473, 475, 479, 483, 491, 492, 494, 496 ... | `y`; `y^{(k)}` |
| `ell` | 436 | subsection Planning--Operation Decomposition Architecture | 4 | 436, 439, 535 | `ell` |
| `d` | 443 | subsection Planning--Operation Decomposition Architecture | 75 | 443, 474, 475, 479, 807, 808, 810, 812, 818, 820, 821, 844, 847, 854, 859 ... | `d` |
| `m` | 476 | subsection Planning--Operation Decomposition Architecture | 3 | 476, 478, 479 | `m` |
| `l` | 581 | subsubsection Total Rated Power and Energy Capacity | 6 | 581, 598, 605, 616, 686, 703 | `l` |
| `k` | 686 | subsubsection Benders' Cuts | 13 | 686, 700, 703, 1436, 1437, 1498, 1545, 1547, 1570, 1573, 1596 | `k` |
| `t` | 807 | subsubsection Shared ESS Mathematical Formulation | 100 | 807, 808, 810, 812, 819, 820, 844, 847, 854, 859, 861, 864, 867, 869, 874 ... | `t` |
| `i` | 1424 | subsection Updated Algorithm | 54 | 1424, 1425, 1426, 1427, 1432, 1441, 1442, 1443, 1444, 1445, 1446, 1447, 1449, 1450, 1451 ... | `i` |
| `j` | 1712 | Appendix: section Distribution Networks | 8 | 1712, 1713 | `j` |

## (iii) Defined twice, overloaded or conflicting

### Duplicate nomenclature keys

- `Y`: l. 197 `$Y$` = Set of representative years; l. 219 `$Y_y$` = Number of calendar years represented by $y \in Y$
- `D`: l. 198 `$D$` = Set of representative days; l. 218 `$D_d$` = Number of calendar days represented by $d \in D$
- `omega`: l. 205 `$\omega_c$` = Probability of investment-cost scenario $c \in \Omega_C$; l. 206 `\textcolor{blue}{$\omega_o$}` = Probability of operation scenario $o \in \Omega_O$

### Defined keys that also occur with a dropped index subscript (same normalised key)

- `Y`: subscript forms ['', 'y']
- `r`: subscript forms ['', 'ij']
- `T^{Cal}`: subscript forms ['', 'e,y']
- `S^{Inv}`: subscript forms ['', 'e,y']
- `E^{Inv}`: subscript forms ['', 'e,y', 'e,y,c']
- `D`: subscript forms ['', 'd']

### Declared overloads (Worker reading; every use verified against the text)

- **`D`** -- D and D_d share the base letter; D_d is a count, D a set (both defined in the nomenclature)
  - set of representative days (nomenclature): body lines [807, 844, 861, 874, 896, 1516, 1517, 1530, 1545, 1547, 1573]; nomenclature lines [198]
  - number of calendar days represented by d, D_d (nomenclature): body lines [808, 818, 821]; nomenclature lines [218]
- **`Y`** -- same pattern as D / D_d
  - set of representative years (nomenclature): body lines [363, 558, 581, 605, 616, 639, 691, 694, 708, 711, 737, 752, 766, 776, 792, 803, 824, 835, 844, 861] ...; nomenclature lines [197]
  - number of calendar years represented by y, Y_y (nomenclature): body lines [827, 832]; nomenclature lines [219]
- **`y`** -- y^{(k)} reuses the year letter for a recourse value; removed with the Benders cuts
  - representative year (index): body lines [407, 408, 426, 428, 430, 432, 443, 473, 475, 479, 483, 491, 492, 494, 496, 499, 500, 502, 504, 558] ...; nomenclature lines []
  - subproblem objective value at iteration k, y^{(k)} (Benders cuts): body lines [689, 700, 706]; nomenclature lines []
- **`r`** -- r carries four meanings; only the discount rate is in the nomenclature
  - discount rate (nomenclature; (1+r)^(y-y0)): body lines [364, 560, 640]; nomenclature lines [221]
  - ADMM loop counter "r = 1 to r^max" (Algorithm 1): body lines [448]; nomenclature lines []
  - penalty update rate r^{E,P}, r^{E,Q} (Appendix A): body lines [1599, 1603, 1607]; nomenclature lines []
  - branch resistance r_ij (Appendix D branch table): body lines [1712, 1713]; nomenclature lines []
- **`x`** -- x is undefined in the nomenclature in both meanings
  - first-stage investment vector x (bold in eq. 1, plain in Algorithm 1): body lines [337, 339, 341, 352, 353, 363, 378, 383, 407, 414, 416, 424]; nomenclature lines []
  - branch reactance x_ij (Appendix D branch table): body lines [1712, 1713]; nomenclature lines []
- **`g`** -- three meanings of g; none in the nomenclature
  - generator index g (Appendix D generator table): body lines [1771]; nomenclature lines []
  - bus conductance g_i (Appendix D bus table): body lines [1677, 1678]; nomenclature lines []
  - cut sensitivities g^{S,ell}, g^{E,ell} (Algorithm 1): body lines [483, 492, 500]; nomenclature lines []
- **`pi`** -- the nomenclature uses omega for probabilities; paragraphs_v5 uses pi_t for the wholesale price
  - scenario probabilities pi_m pi_o in Algorithm 1: body lines [478]; nomenclature lines []
  - ADMM dual variables pi^{I,V}, pi^{E,P}, ... (Appendix A): body lines [1451, 1452, 1453, 1454, 1455, 1473, 1474, 1475, 1476, 1477, 1491, 1492, 1516, 1517, 1547, 1558, 1559, 1564, 1565, 1576] ...; nomenclature lines []
- **`E`** -- E is both a set name and the energy-capacity letter
  - set E with E^S subset of E (Appendix A list; salvage sum e in E): body lines [363, 1528]; nomenclature lines []
  - set of shared ESSs E^S (nomenclature): body lines [557, 581, 605, 616, 639, 691, 694, 708, 711, 737, 752, 766, 776, 792, 803, 824, 835, 844, 861, 874] ...; nomenclature lines [196]
  - energy variables E^{Inv}, E^{Rated}, ... (nomenclature): body lines [407, 408, 432, 502, 566, 576, 591, 598, 599, 608, 621, 644, 658, 695, 712, 744, 759, 764, 773, 783] ...; nomenclature lines [236, 240]
- **`C^{Inv} / c^{Inv}`** -- the budget (lower-case c) and the investment-cost function (upper-case C) differ only by case
  - total investment budget c^{Inv} (nomenclature): body lines [646, 650]; nomenclature lines [215]
  - investment cost C^{Inv}(x), C_c^{Inv} (eq. 1, Algorithm 1, salvage): body lines [337, 364, 416, 424, 512]; nomenclature lines []
- **`epsilon^{rel} / epsilon^{abs}`** -- paragraphs_v5 uses eps_abs = 1e-5 / eps_rel = 1e-4 for the ADMM residual test (different quantity)
  - Benders gap tolerances in Algorithm 1: body lines [529, 531]; nomenclature lines []
- **`hat(P)`** -- paragraphs_v5 uses P-hat for the measured objective period
  - local copy of the SO request, P-hat^{E^k}, P-hat^{I^k} (Appendix A): body lines [1444, 1446, 1466, 1468, 1486, 1516, 1546, 1558, 1559, 1560, 1580]; nomenclature lines []
- **`S^{Rated}`** -- the branch-table header reuses S^Rated with branch indices
  - rated power of the shared ESS (nomenclature, eqs.): body lines [408, 494, 585, 598, 755, 764, 769, 779, 1559, 1560, 1565, 1566]; nomenclature lines [238]
  - branch rating S^Rated_ij (Appendix D branch table): body lines [1712, 1713]; nomenclature lines []

### Declared conflicts between nomenclature and body (Worker reading; lines verified)

- **S_Rated_meaning**: nomenclature defines S^{Rated}_{e,y^Inv,y} as the rated power of the unit installed in y^Inv (per unit); the body uses S^{Rated}_{e,y} for the TOTAL rated power and S^{Rated,Unit}_{e,y^Inv,y} (not in the nomenclature) for the per-unit quantity (lines 238, 769, 769)
- **E_Rated_meaning**: nomenclature defines E^{Rated}_{e,y^Inv,y} as the AVAILABLE energy capacity of the unit (the same description as E^{Av,Unit}, l. 244); the body uses E^{Rated}_{e,y} for the TOTAL rated energy and E^{Av,Unit} for the available (SoH-scaled) energy (lines 240, 244, 773)
- **E_Inv_scenario_index**: the energy-to-power constraint writes E^{Inv(l)}_{e,y,c} with an investment-cost-scenario index c, while the plan is stated to be scenario-independent (S^{Inv(l)}_{e,y} on the same line) (lines 621)
- **phi_year_index**: phi^{Min}_e / phi^{Max}_e carry only the unit index but are described "of unit e in year y" (lines 213, 214)
- **SoH_min_index**: nomenclature SoH^{min}_{e,y} "in year y"; the body uses SoH^{min}_{e,y^Inv} "for unit e installed in year y^Inv" (lines 217, 842)

### Different nomenclature keys with the same description

- "Available energy capacity of unit $e$ installed in year $y^\text{Inv}$ and operated in year $y$": l. 240 `E^{Rated}`; l. 244 `E^{Av,Unit}`

### Local definitions in the body of nomenclature symbols (compare the wording)

| key | line | body text | nomenclature |
|---|---|---|---|
| `Omega_{C}` | 374 | \Omega_C the investment cost trajectories | Set of investment-cost scenarios |
| `omega` | 375 | \omega_c the probability of cost trajectory | Probability of investment-cost scenario $c \in \Omega_C$ / Probability of operation scenario $o \in \Omega_O$ |
| `Omega_{O}` | 376 | \Omega_O the operational scenarios | Set of operational scenarios |
| `omega` | 377 | \omega_o the probability of operational scenario | Probability of investment-cost scenario $c \in \Omega_C$ / Probability of operation scenario $o \in \Omega_O$ |
| `c^{Inv,E}` | 575 | c^{\text{Inv,E}}_{y,c} the scenario-dependent unit investment costs of converter power rating and energy capacity | Unit investment cost of energy capacity in year $y$ and investment scenario $c$ |
| `E^{Inv}` | 576 | E^{\text{Inv}(l)}_{e,y} the corresponding investment decisions} | Investment in energy capacity of ESS $e$ in year $y$ at planning iteration $l$ |
| `E^{Rated}` | 598 | E^{\text{Rated}(l)}_{e,y} the total rated power and energy capacity available at node | Available energy capacity of unit $e$ installed in year $y^\text{Inv}$ and operated in year $y$ |
| `E^{Inv}` | 599 | E^{\text{Inv}(l)}_{e,y^\text{Inv}} the incremental power and energy capacity investments made in year | Investment in energy capacity of ESS $e$ in year $y$ at planning iteration $l$ |
| `T^{Cal}` | 600 | T^\text{Cal}_{e,y} the calendar lifetime of the corresponding installation | Calendar lifetime of ESS $e$ installed in year $y$ |
| `E^{Max}` | 612 | E^{\text{Max}}_{e,y} the maximum installable capacity for ESS | Maximum installable energy capacity of unit $e$ in year $y$ |
| `phi^{Max}` | 628 | \phi^{\text{Max}}_e the minimum and maximum admissible energy-to-power ratios for shared ESS | Maximum admissible energy-to-power ratio of unit $e$ in year $y$ |
| `c^{Inv}` | 650 | c^{\text{Inv}} the total investment budget allocated to shared ESSs | Total investment budget |
| `alpha^{Down}` | 722 | \alpha^{\text{Down}} a predefined lower bound ensuring numerical stability | Lower bound on the Benders underestimator |
| `E^{Av,Unit}` | 788 | E^{\text{Av,Unit}}_{e,y^\text{Inv},y} the available rated power and energy capacity, respectively, of unit | Available energy capacity of unit $e$ installed in year $y^\text{Inv}$ and operated in year $y$ |
| `SoH` | 789 | SoH_{e,y^\text{Inv},y} \in [0,1] the corresponding SoH | SoH of unit $e$ installed in year $y^\text{Inv}$ and operated in year $y$ |
| `CL^{Nom}` | 801 | \mathrm{CL}^\text{Nom}_e its nominal cycle life | Nominal cycle life of unit $e \in E^S$ |
| `D` | 818 | D_d the number of calendar days represented by representative day | Set of representative days / Number of calendar days represented by $d \in D$ |
| `S^{Dch}` | 820 | S^\text{Dch}_{e,y^\text{Inv},y,d,t} the charging and discharging apparent powers, respectively | Discharging apparent power of unit $e$, installed in year $y^\text{Inv}$ and operated in year $y$, representative day $d$, time instant $t$ |
| `Y` | 832 | Y_y the number of years represented by representative year | Set of representative years / Number of calendar years represented by $y \in Y$ |
| `SoH^{min}` | 842 | SoH_{e,y^\text{Inv}}^{\text{min}} the minimum admissible SoH for unit | Minimum admissible SoH of unit $e$ in year $y$ |
| `Q^{Net}` | 883 | Q^{\text{Net}}_{e,y,d,t} the net active and reactive power requests, respectively, issued by the SOs participating | Net reactive power request to unit $e$ in year $y$, representative day $d$, and time instant $t$, issued by the SOs participating in the coordination procedure |
| `E^{S}` | 1528 | E^S \subseteq E set of shared ESSs installed at TN--DN interface nodes | Set of shared ESSs |
| `Y` | 1529 | Y set of representative years in the planning horizon | Set of representative years / Number of calendar years represented by $y \in Y$ |
| `D` | 1530 | D set of representative days | Set of representative days / Number of calendar days represented by $d \in D$ |
| `T` | 1531 | T set of intra-day time periods | Set of intra-day time instants |

## (iv) Symbols of the removed method (Benders)

| key | meaning | nomenclature lines | body lines | n |
|---|---|---|---|---|
| `L^{B}` | set of Benders iterations (nomenclature l. 202) | [202] | [581, 605, 616] | 3 |
| `alpha^{Down}` | lower bound on the Benders underestimator | [216] | [718, 722] | 2 |
| `alpha` | Benders underestimator alpha^(l) / alpha in the LP master of Algorithm 1 | [] | [414, 416, 488, 569, 577, 689, 718] | 7 |
| `mu^{S}` | optimality / feasibility cut multiplier (power), mu^{S(k)} | [] | [692, 701, 709] | 3 |
| `mu^{E}` | optimality / feasibility cut multiplier (energy), mu^{E(k)} | [] | [695, 701, 712] | 3 |
| `g^{S}` | cut sensitivity g^{S,ell} in Algorithm 1 | [] | [483, 492] | 2 |
| `g^{E}` | cut sensitivity g^{E,ell} in Algorithm 1 | [] | [483, 500] | 2 |
| `LB` | Benders lower bound LB^ell | [] | [436, 519, 525, 531] | 4 |
| `UB` | Benders upper bound UB^ell | [] | [436, 510, 511, 525, 526, 531] | 6 |
| `gap` | Benders relative gap gap^ell | [] | [524, 529] | 2 |
| `hat(Q)` | recourse value at the Benders iterate, Q-hat^ell | [] | [472, 489, 512] | 3 |
| `hat(x)` | Benders iterate x-hat^ell | [] | [437, 440, 479, 512, 520] | 5 |
| `hat(S)^{Rated}` | cut expansion point S-hat^{Rated,ell} | [] | [496] | 1 |
| `hat(E)^{Rated}` | cut expansion point E-hat^{Rated,ell} | [] | [504] | 1 |
| `ell` | Benders iteration counter ell (Algorithm 1) | [] | [436, 439, 535] | 4 |
| `ell^{max}` | Benders iteration cap ell^max | [] | [439] | 1 |
| `epsilon^{rel}` | Benders gap tolerance (Algorithm 1) | [] | [529] | 1 |
| `epsilon^{abs}` | Benders gap tolerance (Algorithm 1) | [] | [531] | 1 |

- iteration superscript (l) on investment / rated variables (planning iteration of the Benders master): nomenclature lines [234, 236]; body: `E^{Inv}` [566, 576, 591, 599, 621, 644, 658, 695, 712]; `E^{Rated}` [591, 598, 608]; `S^{Inv}` [563, 576, 585, 599, 620, 622, 642, 658, 692, 709]; `S^{Rated}` [585, 598]; `alpha` [569, 577, 689, 718]; `mu^{E}` [695, 701, 712]; `mu^{S}` [692, 701, 709]; `y` [689, 700, 706]
- y^{(k)} -- subproblem objective value at Benders iteration k (cut constant): nomenclature lines []; body: `y` [689, 700, 706]

Symbols occurring only inside Algorithm 1 (lines [397, 540], replaced wholesale per the map): `LB`, `N`, `Omega_{M}`, `Phi`, `UB`, `W`, `c^{E}`, `c^{S}`, `cal(C)`, `cal(D)`, `cal(E)`, `cal(Y)`, `ell^{max}`, `epsilon^{abs}`, `epsilon^{rel}`, `g^{E}`, `g^{S}`, `gamma`, `gap`, `hat(E)^{Rated}`, `hat(Q)`, `hat(S)^{Rated}`, `hat(x)`, `pi`, `r^{max}`

Lines containing the words Benders / cut(s) / underestimator / optimality gap / lower bound / upper bound: 21 -- 142, 161, 172, 202, 216, 299, 323, 394, 398, 420, 486, 515, 577, 605, 684, 686, 715, 722, 1131, 1136, 1140

## (v) Symbols the revision needs (map section B) and paragraphs_v5.md

| item | used in paragraphs_v5 | paragraphs_v5 notation (line: context) | main.tex today | STEP4 notation |
|---|---|---|---|---|
| lattice unit size 0.25 MVA | False | - | absent | P_{n,y} \in 0.25\,\mathbb{Z}_{\ge 0} MVA (STEP4_DFO_METHOD.md 1.1) |
| lattice unit size 0.5 MWh | False | - | absent | E_{n,y} \in 0.5\,\mathbb{Z}_{\ge 0} MWh (STEP4_DFO_METHOD.md 1.1) |
| duration (E/P) bounds 2-4 h | False | - | `phi^{Min}` l. [620, 628]; `phi^{Max}` l. [622, 628]; text l. 213; text l. 214; text l. 420 | 2 h <= E/P <= 4 h (STEP4_DFO_METHOD.md header) |
| certification threshold tau | True | 27: gs are not growing. Swings smaller than τ/10 = 453.91 € are treated as noise: the; 29: W = max(20, ⌈1.1 P̂⌉) cycles is at most τ. Here P̂ is the period; 31: rface-consensus gap satisfies /t_sum/ ≤ τ/2 = 2,269.53 €; | absent | - |
| first residual pass k0 | True | 20: fied them. From the first residual pass k₀ onwards the regime is; 25: st three turning points (extrema) since k₀, so that its period is measured rather; 34: nd the absence of residual lapses since k₀ | `k` l. [686, 700, 703, 1436, 1437, 1498]... | - |
| certification cycle k* | True | 22: luation is certified at the first cycle k\* at which all of the | `k` l. [686, 700, 703, 1436, 1437, 1498]... | - |
| settling window length W = max(20, ceil(1.1 P-hat)) | True | 29: he range of the objective over the last W = max(20, ⌈1.1 P̂⌉) cycles is at most τ | `W` l. [475] | - |
| measured period P-hat | True | 29: bjective over the last W = max(20, ⌈1.1 P̂⌉) cycles is at most τ. Here P̂ is the p; 29: 20, ⌈1.1 P̂⌉) cycles is at most τ. Here P̂ is the period | `hat(P)^{E}` l. [1446, 1468, 1486, 1516, 1546, 1558]...; `hat(P)^{I}` l. [1444, 1466] | - |
| longest measured period P_max | True | 7 (comment, not manuscript): more slowly" sentence replaced; P-hat, P_max, lambda and the settling slack defined; 37: on that shows no turning point within 2 P_max = 60 cycles, where P_max = 30; 37: point within 2 P_max = 60 cycles, where P_max = 30 | `P^{G,max}` l. [1771] | - |
| priced interface-consensus gap t_sum | True | 31: iced interface-consensus gap satisfies /t_sum/ ≤ τ/2 = 2,269.53 €;; 56: > - its consensus gap /t_sum/ at the cap; | `t` l. [807, 808, 810, 812, 819, 820]... | - |
| ratio resolution delta-R | True | 43: > **The threshold.** τ = δR·V/4, with δR = 0.07 and V = 259,375.33; 43: > **The threshold.** τ = δR·V/4, with δR = 0.07 and V = 259,375.33 €, the SRP1 v; 45: > most τ, so the ratio is resolved to δR. This gives τ = 4,539.07 €. | `delta` l. [795, 827] | - |
| reference unit value V (in tau = deltaR V / 4) | True | 43: > **The threshold.** τ = δR·V/4, with δR = 0.07 and V = 259,375.33 €,; 43: shold.** τ = δR·V/4, with δR = 0.07 and V = 259,375.33 €, the SRP1 value of the r | `V` l. [1426]; `V^{I}` l. [1432]; `hat(V)^{I}` l. [1443, 1465]; `V^{Base}` l. [1677, 1678]; `V^{max}` l. [1677, 1678]; `V^{min}` l. [1677, 1678]; `V^{S}` l. [1771] | - |
| Boyd residual tolerances eps_abs, eps_rel | True | 15: ive test [Boyd et al. 2011, §3.3], with ε_abs = 10⁻⁵ and; 16: > ε_rel = 10⁻⁴. Every local NLP must also solve | `epsilon^{abs}` l. [531]; `epsilon^{rel}` l. [529] | - |
| ADMM penalty rho (frozen from k0) | True | 22: > and the ADMM penalty parameters ρ frozen. An evaluation is certified at t | `rho^{E,P}` l. [1560, 1570, 1578, 1599]; `rho^{E,Q}` l. [1566, 1570, 1587, 1603] | - |
| interface price lambda (consensus dual of the interface power), lambda_t | True | 75: > non-smooth. The interface price λ (the consensus dual of the interface po; 75: he interface power) is then set-valued, λ ∈ [0, c_flex], and; 144: ibility against the TN's marginal value λ_t rather than | `pi^{I,P}` l. [1452, 1474] | - |
| wholesale price pi_t | True | 145: the wholesale price π_t; coordination — or a locational real-ti | `pi` l. [478] | - |
| flexibility price c_flex | True | 75: face power) is then set-valued, λ ∈ [0, c_flex], and | absent | - |
| earlier window L = 44 | True | 96: easured half-life at the earlier window L = 44. | `L^{B}` l. [581, 605, 616]; `cal(L)^{E,P}` l. [1516, 1541, 1558]; `cal(L)^{E,Q}` l. [1517, 1541, 1564] | - |
| elasticity of value to available energy eps_AE | True | 98: lasticity of value to available energy (ε_AE 1.04–1.85, | absent | - |

Unknown LaTeX commands met inside math (ignored by the tokeniser): []

