# P5.15 Addendum 27 W18 -- Phase A evidence tables for the review report (zero solves)

Generated 2026-09-21T05:48:44.846448+00:00 at git HEAD `5aef8cf65de92452a260a69395708a965dc6c1b9` by `p515_s45_phase_a_tables.py` (sha256 `3c6dabcb16de510acfecc3120bee5bb8f13f0a77f27eb13b31d4fbe6994edf97`). Zero solves: guard counts `{'permitted_solve': 0, 'permitted_exec': 0, 'blocked_solve': 0, 'blocked_exec': 0}`, verify(0) failures `[]`.

**Objective convention (all tables):** Q(x) = certified_cost_gross_settlement_excluded = GROSS operational cost, settlement-EXCLUDED (the oracle cost convention); F(x) = I(x) + Q(x) with I(x) from the corrected cost file; terminal_salvage_value and net_operational_recourse = gross - salvage are REPORTED, EXCLUDED from F; the settlement remainder (T_TSO + sum T_DSO) is REPORTED, EXCLUDED from Q and F. Currency EUR.

## Column formulas (T1, T2)

- **Q**: certified_cost_gross_settlement_excluded (campaign_results.json); equals the terminal per-cycle gross_operational_cost (checked)
- **I(x)**: I_new_eur of the candidate key in the W2 or W16 I(x) table (checked equal to the campaign I_x_eur)
- **F**: I(x) + Q
- **F - F0**: F(x) - F(x = 0), x = 0 = a0_c7:x0
- **res. bar**: resolution of F - F0: bar(x) + bar(x0)
- **bar**: max over the last 10 cycles of |Q[k] - Q[k-1]| (gross; campaign bar.value, recomputed and checked)
- **term. step**: |Q[K] - Q[K-1]|, K = last cycle (per_cycle_record.jsonl)
- **win. descent**: Q[K-10] - Q[K] (signed; > 0 = still descending)
- **mono**: last 10 gross steps Q[k] - Q[k-1], k = K-9..K, all <= 0 (non-increasing) or all >= 0 (non-decreasing); "no" = mixed signs
- **r10 net**: rule ten, production: terminal |net recourse step| / terminal objective tolerance
- **r10 gross**: rule ten, gross: terminal |gross step| / terminal objective tolerance
- **settl. rem.**: evaluation_record.json settlement_remainder.value = recourse_components.interface_settlement_total = T_TSO + sum T_DSO at the certified point
- **salvage**: evaluation_record.json recourse_components.terminal_salvage_value
- **net recourse**: recourse_components.net_operational_recourse = gross - salvage
- **EFC/day max**: storage_per_node[n].efc_per_day_max (max over the active cohort-years)
- **min SoH**: storage_per_node[n].terminal_soh_min_over_active_cohort_years
- **floor row**: any storage_per_node[*].soh_floor_rows_active_at_terminal non-empty
- **unfinished flag**: terminal step >= 3,000 EUR AND mono != "no"
- **T2 filter**: F - F0 < 150,000 EUR (x = 0 itself included as the reference row)
- **step<2000**: terminal step < 2,000 EUR

## T1 -- all 68 evaluations (64 distinct keys)

Objective convention: Q(x) = certified_cost_gross_settlement_excluded = GROSS operational cost, settlement-EXCLUDED (the oracle cost convention); F(x) = I(x) + Q(x) with I(x) from the corrected cost file; terminal_salvage_value and net_operational_recourse = gross - salvage are REPORTED, EXCLUDED from F; the settlement remainder (T_TSO + sum T_DSO) is REPORTED, EXCLUDED from Q and F. Currency EUR.

| # | stage:label | dup | key | x (P MVA/E MWh) | cyc | Q gross | I(x) | F | F - F0 | res. bar | bar | term. step | win. descent | mono | r10 net | r10 gross | settl. rem. | salvage | net recourse | EFC/day max (node) | min SoH (node) | floor row |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | a0_c7:lattice_plan_n7_p1.5_e3.0 | dup of a1a:n7_2h_e3 | 2cd1de8249cf | n7 1.5/3.0 @2025 | 112 | 653,099,301 | 1,146,109 | 654,245,410 | 385,948 | 12,320 | 2,690 | 2,475 | 2,454 | no | 0.0379 | 0.0379 | -12,438 | 0.00 | 653,099,301 | n7 1.244 | n7 0.617 | no |
| 2 | a0_c7:n5_p0.25_e0.5 |  | 33031c083b68 | n5 0.25/0.5 @2025 | 134 | 653,721,244 | 191,018 | 653,912,262 | 52,801 | 15,379 | 5,749 | 115 | 37,305 | non-increasing | 0.0018 | 0.0018 | 23,876 | 0.00 | 653,721,244 | n5 1.352 | n5 0.585 | no |
| 3 | a0_c7:n5_p0.25_e1.0 | dup of a1a:n5_4h_e1 | 1821665996ec | n5 0.25/1.0 @2025 | 128 | 653,614,718 | 317,957 | 653,932,675 | 73,214 | 13,446 | 3,816 | 1,008 | 5,004 | no | 0.0154 | 0.0154 | 10,815 | -0.00 | 653,614,718 | n5 1.141 | n5 0.624 | no |
| 4 | a0_c7:n7_p0.25_e0.5 |  | 56b9730bfb7b | n7 0.25/0.5 @2025 | 139 | 653,736,824 | 191,018 | 653,927,842 | 68,381 | 17,920 | 8,290 | 1,251 | 20,059 | no | 0.0191 | 0.0191 | 12,527 | 0.00 | 653,736,824 | n7 1.362 | n7 0.584 | no |
| 5 | a0_c7:n7_p0.25_e1.0 | dup of a1a:n7_4h_e1 | db77e1549af8 | n7 0.25/1.0 @2025 | 125 | 653,597,654 | 317,957 | 653,915,611 | 56,150 | 20,970 | 11,340 | 1,209 | 62,882 | no | 0.0185 | 0.0185 | 50,304 | -0.00 | 653,597,654 | n7 1.145 | n7 0.622 | no |
| 6 | a0_c7:n9_p0.25_e0.5 |  | c50a0e021a23 | n9 0.25/0.5 @2025 | 140 | 653,755,659 | 191,018 | 653,946,678 | 87,216 | 28,682 | 19,052 | 2,540 | -67,183 | non-decreasing | 0.0389 | 0.0389 | -21,909 | 0.00 | 653,755,659 | n9 1.356 | n9 0.589 | no |
| 7 | a0_c7:n9_p0.25_e1.0 | dup of a1a:n9_4h_e1 | a361ac932145 | n9 0.25/1.0 @2025 | 125 | 653,602,442 | 317,957 | 653,920,399 | 60,937 | 21,396 | 11,766 | 1,359 | 70,027 | non-increasing | 0.0208 | 0.0208 | 36,717 | 0.00 | 653,602,442 | n9 1.142 | n9 0.623 | no |
| 8 | a0_c7:x0 |  | 8435c71859dd | x = 0 @2025 | 132 | 653,859,461 | 0 | 653,859,461 | 0 | 0 | 9,630 | 220 | 39,896 | no | 0.0034 | 0.0034 | 23,789 | 0.00 | 653,859,461 | - | - | no |
| 9 | a1a:n5_2h_e1 |  | 2a1302016f89 | n5 0.5/1.0 @2025 | 138 | 653,587,843 | 382,036 | 653,969,880 | 110,418 | 14,242 | 4,612 | 1,679 | 33,363 | non-increasing | 0.0257 | 0.0257 | 22,844 | 0.00 | 653,587,843 | n5 1.339 | n5 0.593 | no |
| 10 | a1a:n5_2h_e2 |  | 5097b0c6b14e | n5 1.0/2.0 @2025 | 136 | 653,310,380 | 764,073 | 654,074,453 | 214,992 | 17,592 | 7,962 | 2,029 | 54,615 | non-increasing | 0.0311 | 0.0311 | 40,781 | 0.00 | 653,310,380 | n5 1.280 | n5 0.611 | no |
| 11 | a1a:n5_2h_e3 |  | bb375de471aa | n5 1.5/3.0 @2025 | 114 | 653,076,367 | 1,146,109 | 654,222,476 | 363,015 | 11,881 | 2,251 | 1,961 | 13,757 | no | 0.0300 | 0.0300 | -6,526 | -0.00 | 653,076,367 | n5 1.243 | n5 0.618 | no |
| 12 | a1a:n5_2h_e4 |  | f20fb813dfef | n5 2.0/4.0 @2025 | 117 | 652,796,656 | 1,528,145 | 654,324,801 | 465,340 | 19,077 | 9,447 | 2,080 | 37,738 | no | 0.0319 | 0.0319 | 34,930 | 0.00 | 652,796,656 | n5 1.176 | n5 0.630 | no |
| 13 | a1a:n5_2h_e5 |  | 302735fb1984 | n5 2.5/5.0 @2025 | 105 | 652,564,047 | 1,910,182 | 654,474,229 | 614,767 | 12,719 | 3,089 | 1,298 | 15,471 | no | 0.0199 | 0.0199 | 7,381 | 0.00 | 652,564,047 | n5 1.150 | n5 0.637 | no |
| 14 | a1a:n5_4h_e1 | dup of a0_c7:n5_p0.25_e1.0 | 1821665996ec | n5 0.25/1.0 @2025 | 128 | 653,614,718 | 317,957 | 653,932,675 | 73,214 | 13,446 | 3,816 | 1,008 | 5,004 | no | 0.0154 | 0.0154 | 10,815 | -0.00 | 653,614,718 | n5 1.141 | n5 0.624 | no |
| 15 | a1a:n5_4h_e2 |  | 3eed1a1ec1a7 | n5 0.5/2.0 @2025 | 111 | 653,337,271 | 635,914 | 653,973,185 | 113,724 | 20,492 | 10,862 | 197 | 65,106 | no | 0.0030 | 0.0030 | 55,097 | -0.00 | 653,337,271 | n5 1.130 | n5 0.631 | no |
| 16 | a1a:n5_4h_e3 |  | 32acd70c1b79 | n5 0.75/3.0 @2025 | 110 | 653,109,230 | 953,871 | 654,063,101 | 203,640 | 24,096 | 14,466 | 1,291 | 48,743 | no | 0.0198 | 0.0198 | 26,115 | -0.00 | 653,109,230 | n5 1.122 | n5 0.637 | no |
| 17 | a1a:n5_4h_e4 |  | f92d165248fe | n5 1.0/4.0 @2025 | 99 | 652,853,135 | 1,271,828 | 654,124,963 | 265,501 | 19,617 | 9,987 | 1,784 | 54,618 | no | 0.0273 | 0.0273 | 44,743 | 0.00 | 652,853,135 | n5 1.096 | n5 0.646 | no |
| 18 | a1a:n5_4h_e5 |  | f0bfdf1c8adb | n5 1.25/5.0 @2025 | 102 | 652,608,642 | 1,589,785 | 654,198,427 | 338,966 | 16,537 | 6,907 | 929 | 51,585 | non-increasing | 0.0142 | 0.0142 | 38,653 | 0.00 | 652,608,642 | n5 1.096 | n5 0.646 | no |
| 19 | a1a:n7_2h_e1 |  | 3993aebc641d | n7 0.5/1.0 @2025 | 138 | 653,598,723 | 382,036 | 653,980,759 | 121,298 | 15,583 | 5,953 | 873 | 40,707 | non-increasing | 0.0133 | 0.0133 | 24,095 | 0.00 | 653,598,723 | n7 1.317 | n7 0.600 | no |
| 20 | a1a:n7_2h_e2 |  | 423e5a7f623e | n7 1.0/2.0 @2025 | 136 | 653,325,523 | 764,073 | 654,089,596 | 230,135 | 16,431 | 6,801 | 2,849 | 55,559 | non-increasing | 0.0436 | 0.0436 | 40,020 | 0.00 | 653,325,523 | n7 1.281 | n7 0.611 | no |
| 21 | a1a:n7_2h_e3 | dup of a0_c7:lattice_plan_n7_p1.5_e3.0 | 2cd1de8249cf | n7 1.5/3.0 @2025 | 112 | 653,099,301 | 1,146,109 | 654,245,410 | 385,948 | 12,320 | 2,690 | 2,475 | 2,454 | no | 0.0379 | 0.0379 | -12,438 | 0.00 | 653,099,301 | n7 1.244 | n7 0.617 | no |
| 22 | a1a:n7_2h_e4 |  | 7048a00e1304 | n7 2.0/4.0 @2025 | 111 | 652,821,679 | 1,528,145 | 654,349,824 | 490,363 | 19,630 | 10,000 | 7,837 | 53,391 | no | 0.1200 | 0.1200 | 31,442 | 0.00 | 652,821,679 | n7 1.175 | n7 0.630 | no |
| 23 | a1a:n7_2h_e5 |  | 354b38814b9c | n7 2.5/5.0 @2025 | 108 | 652,589,463 | 1,910,182 | 654,499,645 | 640,183 | 14,351 | 4,721 | 3,426 | 1,637 | no | 0.0525 | 0.0525 | 2,400 | -0.00 | 652,589,463 | n7 1.150 | n7 0.637 | no |
| 24 | a1a:n7_4h_e1 | dup of a0_c7:n7_p0.25_e1.0 | db77e1549af8 | n7 0.25/1.0 @2025 | 125 | 653,597,654 | 317,957 | 653,915,611 | 56,150 | 20,970 | 11,340 | 1,209 | 62,882 | no | 0.0185 | 0.0185 | 50,304 | -0.00 | 653,597,654 | n7 1.145 | n7 0.622 | no |
| 25 | a1a:n7_4h_e2 |  | cc4eb5df3858 | n7 0.5/2.0 @2025 | 119 | 653,372,840 | 635,914 | 654,008,754 | 149,292 | 15,842 | 6,212 | 3,089 | -4,242 | no | 0.0473 | 0.0473 | 18,311 | 0.00 | 653,372,840 | n7 1.134 | n7 0.630 | no |
| 26 | a1a:n7_4h_e3 |  | fb21d822e145 | n7 0.75/3.0 @2025 | 112 | 653,123,597 | 953,871 | 654,077,468 | 218,007 | 18,054 | 8,424 | 68 | 50,569 | no | 0.0010 | 0.0010 | 27,094 | 0.00 | 653,123,597 | n7 1.126 | n7 0.636 | no |
| 27 | a1a:n7_4h_e4 |  | 11725e45371a | n7 1.0/4.0 @2025 | 100 | 652,873,847 | 1,271,828 | 654,145,675 | 286,214 | 20,960 | 11,330 | 3,284 | 39,972 | no | 0.0503 | 0.0503 | 37,398 | 0.00 | 652,873,847 | n7 1.099 | n7 0.645 | no |
| 28 | a1a:n7_4h_e5 |  | cab98853bb31 | n7 1.25/5.0 @2025 | 111 | 652,650,749 | 1,589,785 | 654,240,534 | 381,073 | 13,713 | 4,083 | 3,887 | -14,000 | no | 0.0596 | 0.0596 | 4,512 | 0.00 | 652,650,749 | n7 1.101 | n7 0.645 | no |
| 29 | a1a:n9_2h_e1 |  | 75271798795b | n9 0.5/1.0 @2025 | 151 | 653,590,205 | 382,036 | 653,972,242 | 112,781 | 15,390 | 5,760 | 2,284 | 41,219 | non-increasing | 0.0349 | 0.0349 | 29,428 | 0.00 | 653,590,205 | n9 1.340 | n9 0.592 | no |
| 30 | a1a:n9_2h_e2 |  | 46628fb108a5 | n9 1.0/2.0 @2025 | 136 | 653,307,647 | 764,073 | 654,071,720 | 212,258 | 19,770 | 10,140 | 1,169 | 62,311 | no | 0.0179 | 0.0179 | 55,088 | 0.00 | 653,307,647 | n9 1.281 | n9 0.611 | no |
| 31 | a1a:n9_2h_e3 |  | dd970259278d | n9 1.5/3.0 @2025 | 107 | 653,082,633 | 1,146,109 | 654,228,742 | 369,281 | 18,926 | 9,296 | 2,599 | 40,315 | non-increasing | 0.0398 | 0.0398 | 14,832 | -0.00 | 653,082,633 | n9 1.185 | n9 0.630 | no |
| 32 | a1a:n9_2h_e4 |  | 30d0f10d376b | n9 2.0/4.0 @2025 | 133 | 652,804,572 | 1,528,145 | 654,332,718 | 473,256 | 10,726 | 1,096 | 1,096 | 279 | no | 0.0168 | 0.0168 | 14,879 | -0.00 | 652,804,572 | n9 1.223 | n9 0.621 | no |
| 33 | a1a:n9_2h_e5 |  | a2a2bd2b7e9d | n9 2.5/5.0 @2025 | 108 | 652,566,422 | 1,910,182 | 654,476,604 | 617,143 | 12,394 | 2,764 | 2,055 | 19,597 | non-increasing | 0.0315 | 0.0315 | 12,747 | 0.00 | 652,566,422 | n9 1.150 | n9 0.637 | no |
| 34 | a1a:n9_4h_e1 | dup of a0_c7:n9_p0.25_e1.0 | a361ac932145 | n9 0.25/1.0 @2025 | 125 | 653,602,442 | 317,957 | 653,920,399 | 60,937 | 21,396 | 11,766 | 1,359 | 70,027 | non-increasing | 0.0208 | 0.0208 | 36,717 | 0.00 | 653,602,442 | n9 1.142 | n9 0.623 | no |
| 35 | a1a:n9_4h_e2 |  | 6ed97c64f357 | n9 0.5/2.0 @2025 | 121 | 653,402,724 | 635,914 | 654,038,638 | 179,177 | 20,995 | 11,365 | 4,554 | -85,661 | non-decreasing | 0.0697 | 0.0697 | -39,519 | 0.00 | 653,402,724 | n9 1.133 | n9 0.630 | no |
| 36 | a1a:n9_4h_e3 |  | ace7403b9b27 | n9 0.75/3.0 @2025 | 104 | 653,127,641 | 953,871 | 654,081,512 | 222,051 | 15,942 | 6,312 | 854 | 15,279 | no | 0.0131 | 0.0131 | 7,402 | 0.00 | 653,127,641 | n9 1.111 | n9 0.640 | no |
| 37 | a1a:n9_4h_e4 |  | 466fb2a6c3b1 | n9 1.0/4.0 @2025 | 97 | 652,852,486 | 1,271,828 | 654,124,314 | 264,853 | 19,602 | 9,972 | 461 | 66,644 | no | 0.0071 | 0.0071 | 53,505 | -0.00 | 652,852,486 | n9 1.097 | n9 0.646 | no |
| 38 | a1a:n9_4h_e5 |  | b2e9d339a0f7 | n9 1.25/5.0 @2025 | 109 | 652,639,191 | 1,589,785 | 654,228,976 | 369,514 | 14,550 | 4,921 | 3,918 | -20,564 | no | 0.0600 | 0.0600 | 50 | 0.00 | 652,639,191 | n9 1.099 | n9 0.646 | no |
| 39 | a1b:n7_2h_e1_y2030 |  | e7c94efae91d | n7 0.5/1.0 @2030 | 131 | 653,669,414 | 283,764 | 653,953,178 | 93,717 | 18,300 | 8,670 | 247 | 44,578 | no | 0.0039 | 0.0038 | 49,145 | 21,378.13 | 653,648,036 | n7 1.208 | n7 0.707 | no |
| 40 | a1b:n7_2h_e1_y2035 |  | 803d1a6cf8f2 | n7 0.5/1.0 @2035 | 139 | 653,790,388 | 237,891 | 654,028,279 | 168,818 | 24,312 | 14,682 | 2,544 | -57,935 | non-decreasing | 0.0392 | 0.0389 | -16,192 | 63,691.69 | 653,726,697 | n7 1.151 | n7 0.834 | no |
| 41 | a1b:n7_2h_e2_y2030 |  | 44cfc7e5b063 | n7 1.0/2.0 @2030 | 127 | 653,513,161 | 567,528 | 654,080,689 | 221,228 | 24,320 | 14,690 | 3,344 | 57,951 | non-increasing | 0.0509 | 0.0512 | 21,886 | 44,533.39 | 653,468,628 | n7 1.165 | n7 0.716 | no |
| 42 | a1b:n7_2h_e2_y2035 |  | 389e6e079d67 | n7 1.0/2.0 @2035 | 116 | 653,701,283 | 475,782 | 654,177,065 | 317,604 | 17,901 | 8,271 | 805 | -40,320 | non-decreasing | 0.0128 | 0.0123 | -22,361 | 132,712.53 | 653,568,570 | n7 1.046 | n7 0.848 | no |
| 43 | a1b:n7_2h_e3_y2030 |  | 0226e5e77299 | n7 1.5/3.0 @2030 | 104 | 653,355,839 | 851,292 | 654,207,131 | 347,670 | 24,983 | 15,353 | 3,762 | 12,064 | no | 0.0585 | 0.0576 | 15,214 | 72,553.46 | 653,283,285 | n7 1.090 | n7 0.735 | no |
| 44 | a1b:n7_2h_e3_y2035 |  | f90159726c7d | n7 1.5/3.0 @2035 | 110 | 653,582,342 | 713,674 | 654,296,016 | 436,555 | 16,054 | 6,424 | 2,233 | 24,506 | no | 0.0349 | 0.0342 | 18,502 | 200,726.54 | 653,381,616 | n7 1.024 | n7 0.850 | no |
| 45 | a1b:n7_2h_e4_y2030 |  | 6d0daf8d44ff | n7 2.0/4.0 @2030 | 100 | 653,173,092 | 1,135,056 | 654,308,148 | 448,687 | 25,304 | 15,674 | 3,785 | 25,388 | no | 0.0593 | 0.0580 | 26,363 | 95,581.28 | 653,077,510 | n7 1.093 | n7 0.732 | no |
| 46 | a1b:n7_2h_e4_y2035 |  | 930e7a1528dd | n7 2.0/4.0 @2035 | 107 | 653,515,395 | 951,565 | 654,466,960 | 607,499 | 16,253 | 6,623 | 2,030 | 1,058 | no | 0.0297 | 0.0311 | -21,477 | 268,723.95 | 653,246,671 | n7 1.014 | n7 0.852 | no |
| 47 | a1b:n7_2h_e5_y2030 |  | b1d7831dfc0d | n7 2.5/5.0 @2030 | 96 | 653,004,397 | 1,418,820 | 654,423,218 | 563,756 | 20,461 | 10,831 | 815 | 70,239 | non-increasing | 0.0116 | 0.0125 | 53,686 | 124,459.78 | 652,879,938 | n7 1.068 | n7 0.741 | no |
| 48 | a1b:n7_2h_e5_y2035 |  | 947de89b21eb | n7 2.5/5.0 @2035 | 116 | 653,413,139 | 1,189,456 | 654,602,596 | 743,134 | 10,461 | 831 | 583 | 3,457 | no | 0.0079 | 0.0089 | -4,427 | 337,745.62 | 653,075,394 | n7 0.999 | n7 0.854 | no |
| 49 | a1b:n7_4h_e1_y2030 |  | c408836ee588 | n7 0.25/1.0 @2030 | 119 | 653,689,107 | 236,168 | 653,925,275 | 65,814 | 19,811 | 10,181 | 2,433 | 75,116 | non-increasing | 0.0371 | 0.0372 | 40,250 | 22,604.72 | 653,666,503 | n7 1.167 | n7 0.719 | no |
| 50 | a1b:n7_4h_e1_y2035 |  | 7fca7670b9b7 | n7 0.25/1.0 @2035 | 119 | 653,776,245 | 197,990 | 653,974,234 | 114,773 | 14,293 | 4,663 | 968 | 32,224 | non-increasing | 0.0146 | 0.0148 | 22,645 | 66,749.72 | 653,709,495 | n7 1.031 | n7 0.850 | no |
| 51 | a1b:n7_4h_e2_y2030 |  | 76baf11ecc05 | n7 0.5/2.0 @2030 | 111 | 653,527,589 | 472,336 | 653,999,925 | 140,464 | 44,869 | 35,240 | 1,061 | 93,311 | no | 0.0165 | 0.0162 | 41,652 | 46,921.91 | 653,480,667 | n7 1.136 | n7 0.727 | no |
| 52 | a1b:n7_4h_e2_y2035 |  | ca1b7bac4ba5 | n7 0.5/2.0 @2035 | 107 | 653,677,238 | 395,979 | 654,073,217 | 213,756 | 32,629 | 22,999 | 656 | 86,427 | non-increasing | 0.0096 | 0.0100 | 46,292 | 136,015.72 | 653,541,222 | n7 0.982 | n7 0.856 | no |
| 53 | a1b:n7_4h_e3_y2030 |  | 3b77a5f6804b | n7 0.75/3.0 @2030 | 106 | 653,381,538 | 708,504 | 654,090,042 | 230,581 | 17,362 | 7,732 | 3,128 | 54,207 | non-increasing | 0.0473 | 0.0479 | 22,565 | 72,536.86 | 653,309,001 | n7 1.107 | n7 0.734 | no |
| 54 | a1b:n7_4h_e3_y2035 |  | 1435efacb664 | n7 0.75/3.0 @2035 | 101 | 653,610,370 | 593,969 | 654,204,339 | 344,878 | 14,996 | 5,366 | 77 | 19,311 | no | 0.0002 | 0.0012 | 20,984 | 205,631.10 | 653,404,739 | n7 0.961 | n7 0.859 | no |
| 55 | a1b:n7_4h_e4_y2030 |  | 0025f3453f36 | n7 1.0/4.0 @2030 | 102 | 653,221,519 | 944,672 | 654,166,191 | 306,729 | 45,099 | 35,469 | 4,026 | 44,349 | no | 0.0623 | 0.0616 | 25,393 | 99,374.61 | 653,122,144 | n7 1.081 | n7 0.741 | no |
| 56 | a1b:n7_4h_e4_y2035 |  | ab6edede03e3 | n7 1.0/4.0 @2035 | 91 | 653,510,196 | 791,958 | 654,302,154 | 442,693 | 24,007 | 14,377 | 2,205 | 64,926 | no | 0.0345 | 0.0338 | 50,512 | 280,836.97 | 653,229,359 | n7 0.897 | n7 0.868 | no |
| 57 | a1b:n7_4h_e5_y2030 |  | b1429b9d9a4a | n7 1.25/5.0 @2030 | 102 | 653,064,056 | 1,180,840 | 654,244,896 | 385,435 | 20,734 | 11,104 | 7,000 | 9,050 | no | 0.1078 | 0.1072 | 24,446 | 125,318.64 | 652,938,737 | n7 1.072 | n7 0.743 | no |
| 58 | a1b:n7_4h_e5_y2035 |  | ed8c36fd40c6 | n7 1.25/5.0 @2035 | 93 | 653,425,376 | 989,948 | 654,415,323 | 555,862 | 44,619 | 34,989 | 257 | 95,686 | no | 0.0051 | 0.0039 | 54,630 | 350,751.74 | 653,074,624 | n7 0.899 | n7 0.867 | no |
| 59 | a2:lattice_c_star_p1.0_e4.0 |  | a002e3ab10e2 | n5 1.0/4.0 + n7 1.0/4.0 + n9 1.0/4.0 @2025 | 109 | 650,893,053 | 3,815,484 | 654,708,537 | 849,076 | 17,238 | 7,608 | 732 | 22,394 | no | 0.0112 | 0.0112 | 21,682 | -0.00 | 650,893,053 | n5 1.126; n7 1.130; n9 1.128 | n5 0.638; n7 0.637; n9 0.637 | no |
| 60 | a2:presence_n5_n7 |  | 00b96fd5cdf5 | n5 0.25/1.0 + n7 0.25/1.0 @2025 | 118 | 653,363,181 | 635,914 | 653,999,095 | 139,633 | 19,224 | 9,594 | 548 | 29,854 | non-increasing | 0.0084 | 0.0084 | 19,855 | -0.00 | 653,363,181 | n5 1.142; n7 1.153 | n5 0.621; n7 0.618 | no |
| 61 | a2:presence_n5_n7_n9 |  | 6fbf40202169 | n5 0.25/1.0 + n7 0.25/1.0 + n9 0.25/1.0 @2025 | 109 | 653,095,878 | 953,871 | 654,049,749 | 190,288 | 29,202 | 19,572 | 1,945 | 99,889 | non-increasing | 0.0298 | 0.0298 | 51,457 | -0.00 | 653,095,878 | n5 1.140; n7 1.145; n9 1.142 | n5 0.624; n7 0.622; n9 0.623 | no |
| 62 | a2:presence_n5_n9 |  | f9689a4b1d27 | n5 0.25/1.0 + n9 0.25/1.0 @2025 | 119 | 653,352,697 | 635,914 | 653,988,611 | 129,150 | 15,437 | 5,807 | 304 | 38,411 | non-increasing | 0.0047 | 0.0047 | 28,783 | 0.00 | 653,352,697 | n5 1.142; n9 1.149 | n5 0.621; n9 0.620 | no |
| 63 | a2:presence_n7_n9 |  | 85d6d95a8ef8 | n7 0.25/1.0 + n9 0.25/1.0 @2025 | 115 | 653,374,249 | 635,914 | 654,010,163 | 150,702 | 30,209 | 20,579 | 3,029 | 17,771 | no | 0.0464 | 0.0464 | 10,273 | -0.00 | 653,374,249 | n7 1.144; n9 1.142 | n7 0.623; n9 0.623 | no |
| 64 | a3:res_n7_p0.5_e1.5_y2025 |  | 062178ab2f3a | n7 0.5/1.5 @2025 | 129 | 653,475,798 | 508,975 | 653,984,773 | 125,312 | 17,586 | 7,956 | 2,608 | 25,060 | no | 0.0399 | 0.0399 | 25,711 | -0.00 | 653,475,798 | n7 1.208 | n7 0.615 | no |
| 65 | a3:res_n7_p0.75_e1.5_y2025 |  | 4f135d7a623c | n7 0.75/1.5 @2025 | 141 | 653,464,682 | 573,055 | 654,037,736 | 178,275 | 14,699 | 5,069 | 549 | 32,206 | no | 0.0084 | 0.0084 | 27,948 | 0.00 | 653,464,682 | n7 1.306 | n7 0.603 | no |
| 66 | a3:res_n7_p0.75_e2.5_y2025 |  | 222e4f71cc13 | n7 0.75/2.5 @2025 | 119 | 653,232,459 | 826,932 | 654,059,392 | 199,930 | 18,190 | 8,560 | 314 | 34,668 | no | 0.0048 | 0.0048 | 20,564 | -0.00 | 653,232,459 | n7 1.173 | n7 0.628 | no |
| 67 | a3:res_n7_p1.25_e2.5_y2025 |  | fdfe88cd4342 | n7 1.25/2.5 @2025 | 130 | 653,194,765 | 955,091 | 654,149,855 | 290,394 | 17,160 | 7,530 | 3,504 | 56,438 | non-increasing | 0.0536 | 0.0536 | 41,423 | 0.00 | 653,194,765 | n7 1.260 | n7 0.615 | no |
| 68 | a3:res_n7_p1_e2.5_y2025 |  | 3e15cb08487b | n7 1.0/2.5 @2025 | 128 | 653,213,468 | 891,012 | 654,104,480 | 245,019 | 17,131 | 7,501 | 5,874 | 41,599 | no | 0.0899 | 0.0899 | 26,228 | 0.00 | 653,213,468 | n7 1.224 | n7 0.620 | no |

## T2 -- decision-relevant subset: F - F(x=0) < 150,000 EUR (22 rows)

Objective convention: Q(x) = certified_cost_gross_settlement_excluded = GROSS operational cost, settlement-EXCLUDED (the oracle cost convention); F(x) = I(x) + Q(x) with I(x) from the corrected cost file; terminal_salvage_value and net_operational_recourse = gross - salvage are REPORTED, EXCLUDED from F; the settlement remainder (T_TSO + sum T_DSO) is REPORTED, EXCLUDED from Q and F. Currency EUR.

Last column: terminal gross step < 2,000 EUR (Advisor test for excluding a size-dependent unfinished-descent bias).

| # | stage:label | dup | key | x (P MVA/E MWh) | cyc | Q gross | I(x) | F | F - F0 | res. bar | bar | term. step | win. descent | mono | r10 net | r10 gross | settl. rem. | salvage | net recourse | EFC/day max (node) | min SoH (node) | floor row | step<2000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | a0_c7:x0 |  | 8435c71859dd | x = 0 @2025 | 132 | 653,859,461 | 0 | 653,859,461 | 0 | 0 | 9,630 | 220 | 39,896 | no | 0.0034 | 0.0034 | 23,789 | 0.00 | 653,859,461 | - | - | no | yes |
| 2 | a0_c7:n5_p0.25_e0.5 |  | 33031c083b68 | n5 0.25/0.5 @2025 | 134 | 653,721,244 | 191,018 | 653,912,262 | 52,801 | 15,379 | 5,749 | 115 | 37,305 | non-increasing | 0.0018 | 0.0018 | 23,876 | 0.00 | 653,721,244 | n5 1.352 | n5 0.585 | no | yes |
| 3 | a0_c7:n7_p0.25_e1.0 | dup of a1a:n7_4h_e1 | db77e1549af8 | n7 0.25/1.0 @2025 | 125 | 653,597,654 | 317,957 | 653,915,611 | 56,150 | 20,970 | 11,340 | 1,209 | 62,882 | no | 0.0185 | 0.0185 | 50,304 | -0.00 | 653,597,654 | n7 1.145 | n7 0.622 | no | yes |
| 4 | a1a:n7_4h_e1 | dup of a0_c7:n7_p0.25_e1.0 | db77e1549af8 | n7 0.25/1.0 @2025 | 125 | 653,597,654 | 317,957 | 653,915,611 | 56,150 | 20,970 | 11,340 | 1,209 | 62,882 | no | 0.0185 | 0.0185 | 50,304 | -0.00 | 653,597,654 | n7 1.145 | n7 0.622 | no | yes |
| 5 | a0_c7:n9_p0.25_e1.0 | dup of a1a:n9_4h_e1 | a361ac932145 | n9 0.25/1.0 @2025 | 125 | 653,602,442 | 317,957 | 653,920,399 | 60,937 | 21,396 | 11,766 | 1,359 | 70,027 | non-increasing | 0.0208 | 0.0208 | 36,717 | 0.00 | 653,602,442 | n9 1.142 | n9 0.623 | no | yes |
| 6 | a1a:n9_4h_e1 | dup of a0_c7:n9_p0.25_e1.0 | a361ac932145 | n9 0.25/1.0 @2025 | 125 | 653,602,442 | 317,957 | 653,920,399 | 60,937 | 21,396 | 11,766 | 1,359 | 70,027 | non-increasing | 0.0208 | 0.0208 | 36,717 | 0.00 | 653,602,442 | n9 1.142 | n9 0.623 | no | yes |
| 7 | a1b:n7_4h_e1_y2030 |  | c408836ee588 | n7 0.25/1.0 @2030 | 119 | 653,689,107 | 236,168 | 653,925,275 | 65,814 | 19,811 | 10,181 | 2,433 | 75,116 | non-increasing | 0.0371 | 0.0372 | 40,250 | 22,604.72 | 653,666,503 | n7 1.167 | n7 0.719 | no | no |
| 8 | a0_c7:n7_p0.25_e0.5 |  | 56b9730bfb7b | n7 0.25/0.5 @2025 | 139 | 653,736,824 | 191,018 | 653,927,842 | 68,381 | 17,920 | 8,290 | 1,251 | 20,059 | no | 0.0191 | 0.0191 | 12,527 | 0.00 | 653,736,824 | n7 1.362 | n7 0.584 | no | yes |
| 9 | a0_c7:n5_p0.25_e1.0 | dup of a1a:n5_4h_e1 | 1821665996ec | n5 0.25/1.0 @2025 | 128 | 653,614,718 | 317,957 | 653,932,675 | 73,214 | 13,446 | 3,816 | 1,008 | 5,004 | no | 0.0154 | 0.0154 | 10,815 | -0.00 | 653,614,718 | n5 1.141 | n5 0.624 | no | yes |
| 10 | a1a:n5_4h_e1 | dup of a0_c7:n5_p0.25_e1.0 | 1821665996ec | n5 0.25/1.0 @2025 | 128 | 653,614,718 | 317,957 | 653,932,675 | 73,214 | 13,446 | 3,816 | 1,008 | 5,004 | no | 0.0154 | 0.0154 | 10,815 | -0.00 | 653,614,718 | n5 1.141 | n5 0.624 | no | yes |
| 11 | a0_c7:n9_p0.25_e0.5 |  | c50a0e021a23 | n9 0.25/0.5 @2025 | 140 | 653,755,659 | 191,018 | 653,946,678 | 87,216 | 28,682 | 19,052 | 2,540 | -67,183 | non-decreasing | 0.0389 | 0.0389 | -21,909 | 0.00 | 653,755,659 | n9 1.356 | n9 0.589 | no | no |
| 12 | a1b:n7_2h_e1_y2030 |  | e7c94efae91d | n7 0.5/1.0 @2030 | 131 | 653,669,414 | 283,764 | 653,953,178 | 93,717 | 18,300 | 8,670 | 247 | 44,578 | no | 0.0039 | 0.0038 | 49,145 | 21,378.13 | 653,648,036 | n7 1.208 | n7 0.707 | no | yes |
| 13 | a1a:n5_2h_e1 |  | 2a1302016f89 | n5 0.5/1.0 @2025 | 138 | 653,587,843 | 382,036 | 653,969,880 | 110,418 | 14,242 | 4,612 | 1,679 | 33,363 | non-increasing | 0.0257 | 0.0257 | 22,844 | 0.00 | 653,587,843 | n5 1.339 | n5 0.593 | no | yes |
| 14 | a1a:n9_2h_e1 |  | 75271798795b | n9 0.5/1.0 @2025 | 151 | 653,590,205 | 382,036 | 653,972,242 | 112,781 | 15,390 | 5,760 | 2,284 | 41,219 | non-increasing | 0.0349 | 0.0349 | 29,428 | 0.00 | 653,590,205 | n9 1.340 | n9 0.592 | no | no |
| 15 | a1a:n5_4h_e2 |  | 3eed1a1ec1a7 | n5 0.5/2.0 @2025 | 111 | 653,337,271 | 635,914 | 653,973,185 | 113,724 | 20,492 | 10,862 | 197 | 65,106 | no | 0.0030 | 0.0030 | 55,097 | -0.00 | 653,337,271 | n5 1.130 | n5 0.631 | no | yes |
| 16 | a1b:n7_4h_e1_y2035 |  | 7fca7670b9b7 | n7 0.25/1.0 @2035 | 119 | 653,776,245 | 197,990 | 653,974,234 | 114,773 | 14,293 | 4,663 | 968 | 32,224 | non-increasing | 0.0146 | 0.0148 | 22,645 | 66,749.72 | 653,709,495 | n7 1.031 | n7 0.850 | no | yes |
| 17 | a1a:n7_2h_e1 |  | 3993aebc641d | n7 0.5/1.0 @2025 | 138 | 653,598,723 | 382,036 | 653,980,759 | 121,298 | 15,583 | 5,953 | 873 | 40,707 | non-increasing | 0.0133 | 0.0133 | 24,095 | 0.00 | 653,598,723 | n7 1.317 | n7 0.600 | no | yes |
| 18 | a3:res_n7_p0.5_e1.5_y2025 |  | 062178ab2f3a | n7 0.5/1.5 @2025 | 129 | 653,475,798 | 508,975 | 653,984,773 | 125,312 | 17,586 | 7,956 | 2,608 | 25,060 | no | 0.0399 | 0.0399 | 25,711 | -0.00 | 653,475,798 | n7 1.208 | n7 0.615 | no | no |
| 19 | a2:presence_n5_n9 |  | f9689a4b1d27 | n5 0.25/1.0 + n9 0.25/1.0 @2025 | 119 | 653,352,697 | 635,914 | 653,988,611 | 129,150 | 15,437 | 5,807 | 304 | 38,411 | non-increasing | 0.0047 | 0.0047 | 28,783 | 0.00 | 653,352,697 | n5 1.142; n9 1.149 | n5 0.621; n9 0.620 | no | yes |
| 20 | a2:presence_n5_n7 |  | 00b96fd5cdf5 | n5 0.25/1.0 + n7 0.25/1.0 @2025 | 118 | 653,363,181 | 635,914 | 653,999,095 | 139,633 | 19,224 | 9,594 | 548 | 29,854 | non-increasing | 0.0084 | 0.0084 | 19,855 | -0.00 | 653,363,181 | n5 1.142; n7 1.153 | n5 0.621; n7 0.618 | no | yes |
| 21 | a1b:n7_4h_e2_y2030 |  | 76baf11ecc05 | n7 0.5/2.0 @2030 | 111 | 653,527,589 | 472,336 | 653,999,925 | 140,464 | 44,869 | 35,240 | 1,061 | 93,311 | no | 0.0165 | 0.0162 | 41,652 | 46,921.91 | 653,480,667 | n7 1.136 | n7 0.727 | no | yes |
| 22 | a1a:n7_4h_e2 |  | cc4eb5df3858 | n7 0.5/2.0 @2025 | 119 | 653,372,840 | 635,914 | 654,008,754 | 149,292 | 15,842 | 6,212 | 3,089 | -4,242 | no | 0.0473 | 0.0473 | 18,311 | 0.00 | 653,372,840 | n7 1.134 | n7 0.630 | no | no |

Unfinished-settling flag (terminal step >= 3,000 and monotone 10-step window): a1a:n9_4h_e2, a1b:n7_2h_e2_y2030, a1b:n7_4h_e3_y2030, a3:res_n7_p1.25_e2.5_y2025

## T3 -- break-even from the node-7 2025 surface

Objective convention: Q(x) = certified_cost_gross_settlement_excluded = GROSS operational cost, settlement-EXCLUDED (the oracle cost convention); F(x) = I(x) + Q(x) with I(x) from the corrected cost file; terminal_salvage_value and net_operational_recourse = gross - salvage are REPORTED, EXCLUDED from F; the settlement remainder (T_TSO + sum T_DSO) is REPORTED, EXCLUDED from Q and F. Currency EUR.

- **value**: value(x) = Q(0) - Q(x), Q = gross settlement-excluded cost, Q(0) = Q of the x = 0 evaluation (a0_c7:x0); positive = storage lowers operating cost
- **fit**: OLS (numpy.linalg.lstsq, rcond=None) of value on X = [1, E, P], one row per DISTINCT candidate key (the A0/A1a duplicate keys have bit-identical Q and enter once); x = 0 is NOT a row
- **standard_errors**: sqrt(diag(s^2 (X^T X)^-1)), s^2 = RSS / (n - 3)
- **residual_rms**: sqrt(RSS / n)
- **residual_max**: max |value_i - fitted_i|
- **unit_costs**: read from W2 investment_cost_results.json expected_unit_costs_per_case_year.{new,old}.2025.{energy_eur_per_mwh_discounted, power_eur_per_mva_discounted} (discount multiplier 1.0 at 2025)
- **i_breakeven_energy_cost_smallest_4h**: e* = (V_meas(0.25/1.0) - p_cost * 0.25) / 1.0
- **ii_breakeven_energy_cost_marginal_4h_MWh**: e* = b + c/4 - p_cost/4  (one extra MWh at 4 h brings 0.25 MVA; value b + c/4, cost e + p/4); se = sqrt(var b + var c/16 + cov(b,c)/2)
- **iii_value_multiplier**: m = I(x) / V_meas(x): the factor by which the measured value would have to grow for storage to pay (F(x) = F(0) when m V = I)
- **iv_old_file**: m_old = I_old(x) / V_meas(x), I_old from the W2 table I_old_eur field (checked against e_old E + p_old P); the break-even energy costs (i), (ii) do not depend on the energy unit cost and are compared with e_old
- **v_ratio_duration**: r(h) = (b + c/h) / (e_cost + p_cost/h) per MWh at duration h = E/P

n = 16 distinct candidates:

| identity | P | E | h | value | fitted | residual | bar | term. step |
|---|---|---|---|---|---|---|---|---|
| a0_c7:n7_p0.25_e0.5 (evals: a0_c7:n7_p0.25_e0.5) | 0.25 | 0.5 | 2.00 | 122,637 | 141,087 | -18,450 | 8,290 | 1,251 |
| a0_c7:n7_p0.25_e1.0 (evals: a0_c7:n7_p0.25_e1.0, a1a:n7_4h_e1) | 0.25 | 1.0 | 4.00 | 261,808 | 254,950 | 6,857 | 11,340 | 1,209 |
| a0_c7:lattice_plan_n7_p1.5_e3.0 (evals: a0_c7:lattice_plan_n7_p1.5_e3.0, a1a:n7_2h_e3) | 1.5 | 3.0 | 2.00 | 760,161 | 775,046 | -14,885 | 2,690 | 2,475 |
| a1a:n7_2h_e1 (evals: a1a:n7_2h_e1) | 0.5 | 1.0 | 2.00 | 260,739 | 267,879 | -7,140 | 5,953 | 873 |
| a1a:n7_2h_e2 (evals: a1a:n7_2h_e2) | 1.0 | 2.0 | 2.00 | 533,938 | 521,462 | 12,476 | 6,801 | 2,849 |
| a1a:n7_4h_e2 (evals: a1a:n7_4h_e2) | 0.5 | 2.0 | 4.00 | 486,622 | 495,605 | -8,984 | 6,212 | 3,089 |
| a1a:n7_4h_e3 (evals: a1a:n7_4h_e3) | 0.75 | 3.0 | 4.00 | 735,864 | 736,260 | -396 | 8,424 | 68 |
| a1a:n7_2h_e4 (evals: a1a:n7_2h_e4) | 2.0 | 4.0 | 2.00 | 1,037,782 | 1,028,630 | 9,153 | 10,000 | 7,837 |
| a1a:n7_4h_e4 (evals: a1a:n7_4h_e4) | 1.0 | 4.0 | 4.00 | 985,614 | 976,916 | 8,698 | 11,330 | 3,284 |
| a1a:n7_2h_e5 (evals: a1a:n7_2h_e5) | 2.5 | 5.0 | 2.00 | 1,269,998 | 1,282,213 | -12,215 | 4,721 | 3,426 |
| a1a:n7_4h_e5 (evals: a1a:n7_4h_e5) | 1.25 | 5.0 | 4.00 | 1,208,712 | 1,217,571 | -8,859 | 4,083 | 3,887 |
| a3:res_n7_p0.5_e1.5_y2025 (evals: a3:res_n7_p0.5_e1.5_y2025) | 0.5 | 1.5 | 3.00 | 383,663 | 381,742 | 1,921 | 7,956 | 2,608 |
| a3:res_n7_p0.75_e1.5_y2025 (evals: a3:res_n7_p0.75_e1.5_y2025) | 0.75 | 1.5 | 2.00 | 394,779 | 394,671 | 109 | 5,069 | 549 |
| a3:res_n7_p0.75_e2.5_y2025 (evals: a3:res_n7_p0.75_e2.5_y2025) | 0.75 | 2.5 | 3.33 | 627,002 | 622,397 | 4,605 | 8,560 | 314 |
| a3:res_n7_p1_e2.5_y2025 (evals: a3:res_n7_p1_e2.5_y2025) | 1.0 | 2.5 | 2.50 | 645,993 | 635,326 | 10,667 | 7,501 | 5,874 |
| a3:res_n7_p1.25_e2.5_y2025 (evals: a3:res_n7_p1.25_e2.5_y2025) | 1.25 | 2.5 | 2.00 | 664,697 | 648,254 | 16,442 | 7,530 | 3,504 |

- a = 14,295 +/- 6,194 EUR; b = 227,727 +/- 3,714 EUR/MWh; c = 51,714 +/- 8,307 EUR/MVA; cov(b,c) = -2.52111e+07; dof = 13
- residual rms = 10,285 EUR; residual max |.| = 18,450 EUR
- unit costs 2025 (from W2 table): energy new 253,877.68, power 256,317.32; energy old 203,102.14, power old 256,317.32
- (i) smallest 4 h unit a0_c7:n7_p0.25_e1.0: V = 261,808 (resolution 20,970); break-even energy cost e* = 197,728.17 EUR/MWh = 0.7788 x new, 0.9735 x old
- (ii) marginal 4 h MWh: value b + c/4 = 240,655.11; e* = 176,575.78 +/- 2,344.60 EUR/MWh = 0.6955 x new, 0.8694 x old
- (iii)/(iv) a0_c7:n7_p0.25_e1.0 (0.25/1.0): V meas 261,808 (fit 254,950); I new 317,957 -> m = 1.214 (fit 1.247); I old 267,181 -> m = 1.021 (fit 1.048)
- (iii)/(iv) a1a:n7_4h_e5 (1.25/5.0): V meas 1,208,712 (fit 1,217,571); I new 1,589,785 -> m = 1.315 (fit 1.306); I old 1,335,907 -> m = 1.105 (fit 1.097)

(v) EXTRAPOLATION beyond the evaluated 2-4 h range for h > 4 (the fit has no point with E/P > 4); linear-in-(E, P) marginal value assumed

| h | value/MWh | cost/MWh new | ratio new | cost/MWh old | ratio old | extrapolation |
|---|---|---|---|---|---|---|
| 2 | 253,583.62 | 382,036.34 | 0.6638 | 331,260.80 | 0.7655 | in range |
| 4 | 240,655.11 | 317,957.01 | 0.7569 | 267,181.47 | 0.9007 | in range |
| 6 | 236,345.61 | 296,597.23 | 0.7969 | 245,821.70 | 0.9615 | EXTRAPOLATION |
| 8 | 234,190.86 | 285,917.34 | 0.8191 | 235,141.81 | 0.9960 | EXTRAPOLATION |
| 10 | 232,898.01 | 279,509.41 | 0.8332 | 228,733.87 | 1.0182 | EXTRAPOLATION |

## T4 -- additivity

Objective convention: Q(x) = certified_cost_gross_settlement_excluded = GROSS operational cost, settlement-EXCLUDED (the oracle cost convention); F(x) = I(x) + Q(x) with I(x) from the corrected cost file; terminal_salvage_value and net_operational_recourse = gross - salvage are REPORTED, EXCLUDED from F; the settlement remainder (T_TSO + sum T_DSO) is REPORTED, EXCLUDED from Q and F. Currency EUR.

- **V**: V(x) = Q(0) - Q(x) (gross, settlement-excluded)
- **ratio**: V(combined) / sum_n V(single node n at the same per-node setting, 2025)
- **difference**: V(combined) - sum_n V(single_n)
- **resolution_bar**: bar(combined) + sum_n bar(single_n) + |k - 1| bar(x0), k = number of nodes (Q(0) enters the difference k - 1 times); bar = max |gross step| over the last 10 cycles
- **resolution_terminal**: the same with the terminal gross step in place of the bar
- **fit_residual_rms**: T3 node-7 2025 fit residual rms (a scale of lattice-surface noise)

| combined | V combined | singles | sum singles | ratio | difference | res. bar | res. terminal | |diff|/res.bar | |diff|/fit rms |
|---|---|---|---|---|---|---|---|---|---|
| a2:presence_n5_n7 | 496,281 | a0_c7:n5_p0.25_e1.0 244,743; a0_c7:n7_p0.25_e1.0 261,808 | 506,551 | 0.9797 | -10,270 | 34,380 | 2,984 | 0.299 | 0.999 |
| a2:presence_n5_n9 | 506,764 | a0_c7:n5_p0.25_e1.0 244,743; a0_c7:n9_p0.25_e1.0 257,020 | 501,763 | 1.0100 | 5,002 | 31,020 | 2,891 | 0.161 | 0.486 |
| a2:presence_n7_n9 | 485,213 | a0_c7:n7_p0.25_e1.0 261,808; a0_c7:n9_p0.25_e1.0 257,020 | 518,827 | 0.9352 | -33,615 | 53,316 | 5,816 | 0.630 | 3.268 |
| a2:presence_n5_n7_n9 | 763,583 | a0_c7:n5_p0.25_e1.0 244,743; a0_c7:n7_p0.25_e1.0 261,808; a0_c7:n9_p0.25_e1.0 257,020 | 763,570 | 1.0000 | 13 | 65,755 | 5,960 | 0.000 | 0.001 |
| a2:lattice_c_star_p1.0_e4.0 | 2,966,408 | a1a:n5_4h_e4 1,006,327; a1a:n7_4h_e4 985,614; a1a:n9_4h_e4 1,006,975 | 2,998,915 | 0.9892 | -32,508 | 58,158 | 6,701 | 0.559 | 3.161 |

fit residual rms (T3) = 10,285 EUR

## T5 -- paper-scale memory and timing

- **block_solve_wall_time**: stage-label interval: t(last sample carrying the label) - t(first sample carrying it), from rss_samples_cycle.jsonl (sampler period 0.5 s, so each interval UNDER-states the true solve time by up to one period); labels "init: DSO|TSO build + solve :: solve <network> <year> <season>"; the label running at a watchdog abort is not a completed solve and is excluded from times
- **footprint_after_solve**: footprint_self of the LAST sample carrying the solve label
- **per_solve_increment_all_consecutive**: footprint_after(solve j) - footprint_after(solve j-1) over consecutive solve labels in sample order, INCLUDING the label running at the abort and the steps that span an unlabelled network-build block
- **per_solve_increment_within_network**: the same, restricted to consecutive solves of the SAME network (no build in between) and to completed labels
- **units**: GiB = 2^30 bytes

### paper_cycle_snapoff_r1: status watchdog_abort (exit 97), abort at "init: DSO build + solve" (process-tree memory above 24.00 GiB (25769803776 bytes))
solve labels 40 (completed 40); DSO interval {'n': 40, 'mean': 18.37155, 'median': 17.65099999999996, 'min': 14.433999999999997, 'max': 25.451999999999998}; TSO interval None
- increment_all_consecutive: n 39, mean 0.3969, median 0.2513, first-10 0.2613, last-10 0.2230 GiB
- increment_within_network_completed: n 38, mean 0.2575, median 0.2484, first-10 0.2613, last-10 0.2230 GiB
- build block "init: DSO build + solve": 61.4 s, footprint +5.279 GiB
- build block "init: DSO build + solve": 65.2 s, footprint +5.353 GiB
- build block "init: DSO build + solve": 34.8 s, footprint +2.693 GiB (ABORTED inside this block; incomplete)

| solve label | interval s | samples | footprint after GiB | completed |
|---|---|---|---|---|
| init: DSO build + solve :: solve case33_1 2025 Spring | 17.664 | 35 | 5.841 | yes |
| init: DSO build + solve :: solve case33_1 2025 Summer | 14.959 | 30 | 5.993 | yes |
| init: DSO build + solve :: solve case33_1 2025 Autumn | 16.491 | 33 | 6.238 | yes |
| init: DSO build + solve :: solve case33_1 2025 Winter | 16.061 | 32 | 6.456 | yes |
| init: DSO build + solve :: solve case33_1 2028 Spring | 17.506 | 35 | 6.646 | yes |
| init: DSO build + solve :: solve case33_1 2028 Summer | 15.949 | 32 | 6.886 | yes |
| init: DSO build + solve :: solve case33_1 2028 Autumn | 18.844 | 27 | 7.093 | yes |
| init: DSO build + solve :: solve case33_1 2028 Winter | 15.961 | 32 | 7.547 | yes |
| init: DSO build + solve :: solve case33_1 2031 Spring | 16.562 | 33 | 7.824 | yes |
| init: DSO build + solve :: solve case33_1 2031 Summer | 21.624 | 43 | 8.132 | yes |
| init: DSO build + solve :: solve case33_1 2031 Autumn | 14.434 | 29 | 8.454 | yes |
| init: DSO build + solve :: solve case33_1 2031 Winter | 17.672 | 35 | 8.614 | yes |
| init: DSO build + solve :: solve case33_1 2034 Spring | 16.524 | 33 | 8.934 | yes |
| init: DSO build + solve :: solve case33_1 2034 Summer | 17.054 | 34 | 9.261 | yes |
| init: DSO build + solve :: solve case33_1 2034 Autumn | 16.581 | 33 | 9.585 | yes |
| init: DSO build + solve :: solve case33_1 2034 Winter | 17.059 | 34 | 9.850 | yes |
| init: DSO build + solve :: solve case33_1 2037 Spring | 18.631 | 37 | 10.127 | yes |
| init: DSO build + solve :: solve case33_1 2037 Summer | 17.086 | 34 | 10.490 | yes |
| init: DSO build + solve :: solve case33_1 2037 Autumn | 17.110 | 34 | 10.827 | yes |
| init: DSO build + solve :: solve case33_1 2037 Winter | 17.706 | 35 | 11.032 | yes |
| init: DSO build + solve :: solve case33_2 2025 Spring | 18.214 | 36 | 16.727 | yes |
| init: DSO build + solve :: solve case33_2 2025 Summer | 25.452 | 50 | 16.952 | yes |
| init: DSO build + solve :: solve case33_2 2025 Autumn | 17.638 | 35 | 17.190 | yes |
| init: DSO build + solve :: solve case33_2 2025 Winter | 20.246 | 40 | 17.473 | yes |
| init: DSO build + solve :: solve case33_2 2028 Spring | 20.827 | 41 | 17.690 | yes |
| init: DSO build + solve :: solve case33_2 2028 Summer | 20.708 | 41 | 18.017 | yes |
| init: DSO build + solve :: solve case33_2 2028 Autumn | 17.212 | 34 | 18.286 | yes |
| init: DSO build + solve :: solve case33_2 2028 Winter | 19.741 | 39 | 18.648 | yes |
| init: DSO build + solve :: solve case33_2 2031 Spring | 21.881 | 43 | 18.793 | yes |
| init: DSO build + solve :: solve case33_2 2031 Summer | 22.840 | 45 | 19.089 | yes |
| init: DSO build + solve :: solve case33_2 2031 Autumn | 17.145 | 34 | 19.285 | yes |
| init: DSO build + solve :: solve case33_2 2031 Winter | 18.164 | 36 | 19.479 | yes |
| init: DSO build + solve :: solve case33_2 2034 Spring | 21.235 | 42 | 19.731 | yes |
| init: DSO build + solve :: solve case33_2 2034 Summer | 20.780 | 41 | 19.928 | yes |
| init: DSO build + solve :: solve case33_2 2034 Autumn | 16.579 | 33 | 20.124 | yes |
| init: DSO build + solve :: solve case33_2 2034 Winter | 17.308 | 34 | 20.433 | yes |
| init: DSO build + solve :: solve case33_2 2037 Spring | 17.224 | 34 | 20.619 | yes |
| init: DSO build + solve :: solve case33_2 2037 Summer | 18.679 | 37 | 20.863 | yes |
| init: DSO build + solve :: solve case33_2 2037 Autumn | 21.330 | 42 | 21.053 | yes |
| init: DSO build + solve :: solve case33_2 2037 Winter | 20.181 | 40 | 21.319 | yes |

### paper_cycle_snapoff_r2: status watchdog_abort (exit 97), abort at "init: TSO build + solve :: solve case9 2034 Autumn" (swap used grew by 1147011072 bytes (above 1.00 GiB) within 60 s (thrashing guard))
solve labels 75 (completed 74); DSO interval {'n': 60, 'mean': 19.34093333333334, 'median': 18.67900000000003, 'min': 14.431999999999988, 'max': 30.690000000000055}; TSO interval {'n': 14, 'mean': 6.760214285714304, 'median': 6.219499999999925, 'min': 4.669000000000096, 'max': 10.823000000000093}
- increment_all_consecutive: n 74, mean 0.4060, median 0.2301, first-10 0.2201, last-10 0.0939 GiB
- increment_within_network_completed: n 70, mean 0.2214, median 0.2248, first-10 0.2201, last-10 0.0918 GiB
- build block "init: DSO build + solve": 60.6 s, footprint +5.280 GiB
- build block "init: DSO build + solve": 65.4 s, footprint +6.397 GiB
- build block "init: DSO build + solve": 95.6 s, footprint +5.443 GiB
- build block "init: TSO build + solve": 59.7 s, footprint +2.006 GiB
- Planner-figure reproduction variants: {"A_mean_all_consecutive_74_steps_gib": 0.406, "A_last10_mean_gib": 0.0939, "B_(fp_after_last_label - fp_end_first_DSO_build_block)/75_gib": 0.404, "C_(fp_after_last_label - fp_after_first_solve)/74_gib": 0.406, "D_mean_completed_only_73_steps_gib": 0.4106, "D_last10_mean_completed_only_gib": 0.0918}

| solve label | interval s | samples | footprint after GiB | completed |
|---|---|---|---|---|
| init: DSO build + solve :: solve case33_1 2025 Spring | 17.135 | 34 | 5.799 | yes |
| init: DSO build + solve :: solve case33_1 2025 Summer | 15.037 | 30 | 5.995 | yes |
| init: DSO build + solve :: solve case33_1 2025 Autumn | 16.104 | 32 | 6.232 | yes |
| init: DSO build + solve :: solve case33_1 2025 Winter | 15.883 | 31 | 6.422 | yes |
| init: DSO build + solve :: solve case33_1 2028 Spring | 17.633 | 35 | 6.676 | yes |
| init: DSO build + solve :: solve case33_1 2028 Summer | 15.579 | 31 | 6.870 | yes |
| init: DSO build + solve :: solve case33_1 2028 Autumn | 18.891 | 28 | 7.120 | yes |
| init: DSO build + solve :: solve case33_1 2028 Winter | 15.239 | 30 | 7.292 | yes |
| init: DSO build + solve :: solve case33_1 2031 Spring | 16.553 | 33 | 7.519 | yes |
| init: DSO build + solve :: solve case33_1 2031 Summer | 21.628 | 43 | 7.741 | yes |
| init: DSO build + solve :: solve case33_1 2031 Autumn | 14.432 | 29 | 8.000 | yes |
| init: DSO build + solve :: solve case33_1 2031 Winter | 18.095 | 36 | 8.237 | yes |
| init: DSO build + solve :: solve case33_1 2034 Spring | 16.670 | 33 | 8.453 | yes |
| init: DSO build + solve :: solve case33_1 2034 Summer | 16.604 | 33 | 8.672 | yes |
| init: DSO build + solve :: solve case33_1 2034 Autumn | 17.077 | 34 | 8.890 | yes |
| init: DSO build + solve :: solve case33_1 2034 Winter | 17.059 | 34 | 9.088 | yes |
| init: DSO build + solve :: solve case33_1 2037 Spring | 19.114 | 38 | 9.323 | yes |
| init: DSO build + solve :: solve case33_1 2037 Summer | 16.599 | 33 | 9.490 | yes |
| init: DSO build + solve :: solve case33_1 2037 Autumn | 17.574 | 35 | 9.760 | yes |
| init: DSO build + solve :: solve case33_1 2037 Winter | 18.113 | 36 | 9.956 | yes |
| init: DSO build + solve :: solve case33_2 2025 Spring | 18.001 | 36 | 16.643 | yes |
| init: DSO build + solve :: solve case33_2 2025 Summer | 26.187 | 52 | 16.857 | yes |
| init: DSO build + solve :: solve case33_2 2025 Autumn | 18.061 | 36 | 17.168 | yes |
| init: DSO build + solve :: solve case33_2 2025 Winter | 20.192 | 40 | 17.364 | yes |
| init: DSO build + solve :: solve case33_2 2028 Spring | 21.589 | 43 | 17.772 | yes |
| init: DSO build + solve :: solve case33_2 2028 Summer | 20.604 | 41 | 17.928 | yes |
| init: DSO build + solve :: solve case33_2 2028 Autumn | 17.598 | 35 | 18.220 | yes |
| init: DSO build + solve :: solve case33_2 2028 Winter | 19.694 | 39 | 18.513 | yes |
| init: DSO build + solve :: solve case33_2 2031 Spring | 22.177 | 44 | 18.660 | yes |
| init: DSO build + solve :: solve case33_2 2031 Summer | 23.131 | 46 | 18.925 | yes |
| init: DSO build + solve :: solve case33_2 2031 Autumn | 17.540 | 35 | 19.296 | yes |
| init: DSO build + solve :: solve case33_2 2031 Winter | 18.685 | 37 | 19.637 | yes |
| init: DSO build + solve :: solve case33_2 2034 Spring | 21.822 | 43 | 19.917 | yes |
| init: DSO build + solve :: solve case33_2 2034 Summer | 21.315 | 42 | 20.179 | yes |
| init: DSO build + solve :: solve case33_2 2034 Autumn | 16.611 | 33 | 20.369 | yes |
| init: DSO build + solve :: solve case33_2 2034 Winter | 17.200 | 34 | 20.672 | yes |
| init: DSO build + solve :: solve case33_2 2037 Spring | 17.644 | 35 | 20.959 | yes |
| init: DSO build + solve :: solve case33_2 2037 Summer | 19.236 | 38 | 21.306 | yes |
| init: DSO build + solve :: solve case33_2 2037 Autumn | 21.741 | 43 | 21.550 | yes |
| init: DSO build + solve :: solve case33_2 2037 Winter | 20.234 | 40 | 21.729 | yes |
| init: DSO build + solve :: solve case33_3 2025 Spring | 15.497 | 31 | 27.440 | yes |
| init: DSO build + solve :: solve case33_3 2025 Summer | 26.837 | 53 | 27.790 | yes |
| init: DSO build + solve :: solve case33_3 2025 Autumn | 20.777 | 41 | 28.066 | yes |
| init: DSO build + solve :: solve case33_3 2025 Winter | 17.680 | 35 | 28.342 | yes |
| init: DSO build + solve :: solve case33_3 2028 Spring | 30.690 | 60 | 28.553 | yes |
| init: DSO build + solve :: solve case33_3 2028 Summer | 19.737 | 39 | 28.884 | yes |
| init: DSO build + solve :: solve case33_3 2028 Autumn | 21.789 | 43 | 29.117 | yes |
| init: DSO build + solve :: solve case33_3 2028 Winter | 16.753 | 33 | 29.328 | yes |
| init: DSO build + solve :: solve case33_3 2031 Spring | 22.281 | 44 | 29.661 | yes |
| init: DSO build + solve :: solve case33_3 2031 Summer | 21.439 | 42 | 29.881 | yes |
| init: DSO build + solve :: solve case33_3 2031 Autumn | 20.753 | 41 | 30.290 | yes |
| init: DSO build + solve :: solve case33_3 2031 Winter | 25.124 | 49 | 30.574 | yes |
| init: DSO build + solve :: solve case33_3 2034 Spring | 21.837 | 43 | 30.809 | yes |
| init: DSO build + solve :: solve case33_3 2034 Summer | 22.338 | 44 | 31.122 | yes |
| init: DSO build + solve :: solve case33_3 2034 Autumn | 16.629 | 33 | 31.390 | yes |
| init: DSO build + solve :: solve case33_3 2034 Winter | 17.698 | 35 | 31.552 | yes |
| init: DSO build + solve :: solve case33_3 2037 Spring | 24.788 | 49 | 31.926 | yes |
| init: DSO build + solve :: solve case33_3 2037 Summer | 18.673 | 37 | 32.241 | yes |
| init: DSO build + solve :: solve case33_3 2037 Autumn | 19.733 | 39 | 32.233 | yes |
| init: DSO build + solve :: solve case33_3 2037 Winter | 19.122 | 38 | 32.514 | yes |
| init: TSO build + solve :: solve case9 2025 Spring | 6.721 | 14 | 34.586 | yes |
| init: TSO build + solve :: solve case9 2025 Summer | 5.176 | 11 | 34.710 | yes |
| init: TSO build + solve :: solve case9 2025 Autumn | 4.669 | 10 | 34.824 | yes |
| init: TSO build + solve :: solve case9 2025 Winter | 6.224 | 13 | 34.852 | yes |
| init: TSO build + solve :: solve case9 2028 Spring | 5.200 | 11 | 34.905 | yes |
| init: TSO build + solve :: solve case9 2028 Summer | 6.733 | 14 | 34.976 | yes |
| init: TSO build + solve :: solve case9 2028 Autumn | 6.197 | 13 | 35.043 | yes |
| init: TSO build + solve :: solve case9 2028 Winter | 6.215 | 13 | 35.159 | yes |
| init: TSO build + solve :: solve case9 2031 Spring | 7.215 | 15 | 35.303 | yes |
| init: TSO build + solve :: solve case9 2031 Summer | 8.788 | 18 | 35.435 | yes |
| init: TSO build + solve :: solve case9 2031 Autumn | 6.201 | 13 | 35.516 | yes |
| init: TSO build + solve :: solve case9 2031 Winter | 8.269 | 17 | 35.596 | yes |
| init: TSO build + solve :: solve case9 2034 Spring | 10.823 | 22 | 35.683 | yes |
| init: TSO build + solve :: solve case9 2034 Summer | 6.212 | 13 | 35.770 | yes |
| init: TSO build + solve :: solve case9 2034 Autumn | 1.542 | 4 | 35.845 | no |

### srp1_cycle_snapoff_r1: status complete (exit 0), abort at "None" (None)
paper block intervals (s): {"DSO": {"n": 100, "mean": 18.953180000000003, "median": 18.07800000000003, "source": "r1 + r2 completed init DSO solve labels"}, "TSO": {"n": 14, "mean": 6.760214285714304, "median": 6.219499999999925, "source": "r2 completed init TSO solve labels (r1 aborted before the TSO)"}}
- SRP1 initialization: 37 solve labels observed; footprint 0.3872 -> 1.0626 GiB, total +0.6753 GiB; per consecutive step mean 0.0188; within-network steps only (no build block in between): 33 steps, sum +0.3187 GiB. SRP1 solves are shorter than the 0.5 s sampler period, so labels are aliased: only 37 of the 48 network solves of the pass carry a sample; the first-to-last total is exact for the span it covers, the per-label counts are not per-solve counts
- SRP1 cycle 1: 41 solve labels observed; footprint 1.1084 -> 1.3297 GiB, total +0.2213 GiB; per consecutive step mean 0.0055; within-network steps only (no build block in between): 37 steps, sum +0.2092 GiB. SRP1 solves are shorter than the 0.5 s sampler period, so labels are aliased: only 41 of the 48 network solves of the pass carry a sample; the first-to-last total is exact for the span it covers, the per-label counts are not per-solve counts
- SRP1 init pass spans the build blocks ['init: DSO build + solve', 'init: DSO build + solve', 'init: DSO build + solve', 'init: TSO build + solve', 'init: ESSO build + solve']

Projection: T_cycle = n_DSO_blocks * mean(DSO interval) + n_TSO_blocks * mean(TSO interval), n_DSO_blocks = 60, n_TSO_blocks = 20 (paper build block_counts); ESSO solves and ADMM bookkeeping NOT included (not measured at paper scale) -> a LOWER bound under the assumption that cycle solves cost what cold init solves cost. T_eval = T_build + (K + 1) * T_cycle; +1 = the initialization round; T_build = sum of the unlabelled "init: <agent> build + solve" block intervals of r2 (3 DSO + 1 TSO network builds). K = 114.5 (median cycles_run over the 64 distinct Phase A candidates at SRP1 (min 91, max 151); ASSUMES paper scale needs the same cycle count -- unverified).
- T_cycle = 1272.4 s; T_build = 281.3 s; T_eval = 147243 s = 40.90 h (K min/max: 32.59 / 53.80 h)
- SRP1 reference per-cycle wall: {"definition": "run_admm_arm_s / cycles_run per distinct Phase A candidate", "mean": 32.23124515607435, "median": 32.51598701337575}
- neither paper run completed initialization (r1 aborted at the 24 GiB process-tree watchdog in the DSO init, r2 at the swap-growth guard in the TSO init); the projection is a wall-time figure only and says nothing about feasibility in memory

