# APIA 2.0 — Deneysel Sonuç Raporu

**Sistem:** Sabit ESN rezervuar (D=100, seyreklik 0.1, alpha=0.3, rho~0.97) +
B-spline oznitelik haritasi (K=10, D*K+1=1001 oznitelik) + cevrimici RLS okuma
+ N-slot baglamsal cift-hizli bellek. Backprop yok; tum ogrenme yerel/cevrimici.

---

## 1. Okuma motoru karakterizasyonu (Mackey-Glass, 10 seed)

| Yontem | NMSE @5000 adim | Karmasiklik | Bellek |
|---|---|---|---|
| Ridge (batch) | ~9e-6 (referans) | O(F^3) batch | F^2 |
| Full RLS lam=1 | ~1e-3 | O(F^2)/adim | F^2 |
| Block-diag RLS lam=1 | ~2e-2 | O(D*K^2)/adim | D*K^2 (~x100 az) |

- Block-RLS, Python-vektorize (~53us/adim), Full RLS'e gore ~75x hizli.
- lam=0.98'de Full RLS kovaryans windup ile ıraksiyor; trace-bounding (P *= bound/trace)
  her iki motorda da windup'i kesiyor. `init_var=1e2` blok icin optimal (1e6 kotu).

## 2. Sinüs -> Kare -> Sinüs protokolu (10 seed, 1500 adim/faz, lam=0.98)

| Yontem | F1-end | F2-end | F3-ONSET | F3-end |
|---|---|---|---|---|
| FullRLS+bound | 3.8e-11 | 0.108 | 8.17 +/- 4.0 | 6.2e-3 |
| BlockRLS+bound | 1.6e-4 | 0.545 | 0.459 +/- 0.22 | 1.0e-2 |
| BlockRLS+naive | 1.6e-4 | 0.543 | 0.456 | 1.0e-2 |
| MultiHead | 1.6e-4 | 0.545 | 0.424 +/- 0.16 | 9.3e-3 |
| **BlockRLS+CTX** | 1.9e-4 | 0.443 | **0.104 +/- 0.02** | 3.7e-2 |

**Sonuc (nedensel-m2 mekanizmasi, son hali): x4.40 ortalama F3-onset
iyilesmesi; 10/10 seed iyilesti (0.067-0.128). En kotu durum
0.913 -> 0.081. Naive konsolidasyon kazanci sifir.
Wilcoxon signed-rank (block>ctx, tek-yonlu): W=55, p=9.8e-4;
paired t: t=4.85, p=9.1e-4.**
Pencere duyarliligi: w=100/250/500 -> x4.41 / x4.40 / x4.66.
(Earlier global-var runs: 0.095, x4.84 — neredeyse birebir ayni.)

Rakip merdiveni (F3-onset): MultiHead 0.424 -> +alloc 0.377 (eksik
tahsis kanali) -> +alloc+soft 0.291 (normalize gate de yarar ama
yeterli degil; tek basina soft=0.323). Kalan ~x2.8 rakibin ungated
kanali olmadigindan gate'ten gelemez -> cift-oranli transfer
mimarisine atfedilir (merdivenli dekompozisyon, faktoriyel degil).

ANA PROTOKOL VALF KONTROLU (headline confound, 10 seed):
R = 0.939 (ctx konsolidasyon olaylarinda medyan ||W_fast||, 180 olay).
|| Konfig | F3-onset | F3-end |
|---|---|---|
|| block | 0.459 | 0.010 |
|| block+clamp | 0.222 | 0.405 |
|| ctx | 0.104 | 0.037 |
- Klemp tek basina x2.07 (W=53, p=0.003); bellek klempin otesinde
  x2.13 (W=55, p=0.001) -> x4.40 ~= x2.07 drenaj x x2.13 bellek.
- LOSSY vs LOSSLESS: klemp F3-end'i ~40x bozar (0.405 vs 0.010);
  transfer bilgiyi korur -> "neden norm cap degil"in olculen cevabi.

NEDENSELLIK DUZELTMESI (kritik): signal_var artik akisin tamamindan
hesaplanan GLOBAL istatistik degil; mekanizma icindeki tum esikler
(err_gate, tau) online EMA tahmini m2 = E[y^2] kullaniyor
(alpha_m2=0.02). Bos-slot tabani zaten E[y^2] idi (M4 fix ayni
yerden). Sifir-ortalamali sinus/kare akisinda E[y^2]=var oldugundan
sayilar korundu; MG/NARMA (mu!=0) icin mu^2 terimi esigi yukseltir —
NARMA testinde desen 5/5 -> 3/5, kazanc x1.10 -> x1.05 (olculen bedel).

Ablation katmanlari (10 seed, F3-onset NMSE):
| Varyant | NMSE | Not |
|---|---|---|
| block | 0.459+/-0.22 | baseline |
| naive | 0.456 | kosulsuz konsolidasyon |
| MultiHead (Bereska) | 0.424+/-0.16 | RAKIP: 9/10 seed'de baseline'a coker |
| MultiHead+alloc | 0.377+/-0.13 | kanal izolasyonu: tetik yardimci, mimari baskin |
| CTX v0 (self-mask bug) | 0.455 | W_fast'in slot hatasina katilmasi |
| CTX v1 (slow-err) | 0.216+/-0.07 | sadece self-masking duzeltmesi |
| CTX v2 (+protect, norm) | 0.187+/-0.06 | causal esik altinda ORTALAMA kazanc var; 2/10 seed geriler |
| CTX v3 (+count) | 0.104+/-0.02 | final — 10/10 seed'de v2'yi yener (p<1e-3) |

DOYGUNLUK MINI-ABLATION (sine->square->triangle->sine, 10 seed, F4-onset):
| Politika | F4-onset | eviction | refusal |
|---|---|---|---|
| block | 0.069+/-0.024 | - | - |
| S=2 overwrite | 0.016+/-0.002 | 11.2 | 0 |
| S=2 refuse | 0.073+/-0.031 | 0 | 14.5 |
| S=3 overwrite | 0.023+/-0.012 | 11.8 | 0 |
| S=3 refuse | 0.071+/-0.028 | 0 | 13.4 |

**Sasirtici bulgu:** refuse memoryless'a coker, overwrite x3-4 korur.
Mekanizma: refuse konsolidasyonu durdurur -> gamma-decay hic ateslenmez
-> W_fast yeni rejimi TEK BASINA ogrenip norm ~36'ya siser (overwrite:
~0.3). Donus'te gate dogru slotu secer ama devasa bayat W_fast katkisini
bogar. Overwrite'da count-argmin en AZ konsolideyi siler (slot0'in 13
yazimi eviction-proof) + eviction additive'dir (dilute, erase degil).

VALF CONFOUND TESTI (tam faktoriyel, R=3.18 norm-clamp):
| Konfig | F4-onset | norm@sinir |
|---|---|---|
| block | 0.069 | 34.7 |
| block+clamp | 0.051 | 3.2 |
| refuse | 0.073 | 35.8 |
| refuse+clamp | 0.019 | 3.2 |
| overwrite | 0.016 | 0.31 |
Korelasyon: log||W_fast||@sinir vs onset: rho=0.815, p=1.5e-10.
MIDAHALE KARARI: refuse+clamp hatirlamayi TAM geri getirir (0.019~0.016)
-> sebep BUYUKLUK (drenaj), depolama degil. Ama block+clamp sadece
0.051'e iner -> sakli baglam klempten ote ~2.7x katki verir; bellek
malzemeyi, drenaj erisimi saglar. Naive (ana protokolde drenaj atesler
ama 0.456) drenajin TEK BASINA yetersiz oldugunu kanitlar.
UC KOSUL BIRLIKTE GEREKLI: >=2 ayrilabilir slot + dogru gate + duzenli
drenaj. Not: 4-rejim akisi ucuz-rejenerasyon (block onset 0.069, end'ler
[0,.34,.18,.003]) — mutlak kazanc valfi olcer, bellegi degil.
Refuse secilirse W_fast norm-cap zorunlu.

**MultiHead neden cohuyor (onemli rakip bulgusu):** pasif korumanin TEK
ayrisma kanali tahmin gate'i; bos head tabani E[y^2]~1, sinus-head'in
kare-hatasi ~0.7 (kare=sign(sin) kismi korelasyonu) -> argmin asla
slot degistirmez -> mekanizma devreye girmeden tek-head'e doner.
Tek calisan seed (6): head0 hatasi >1 -> gecis oldu -> 0.913->0.546.
Bizim ikinci kanalimiz (write-protection: E_min>tau*m2 -> NEW_CTX)
gate kapali kalirken bile baglamlari ayirir — ampirik fark kaniti.

## 3. Ablation / mekanizma bulgulari (her biri ayri ders)

| # | Bulgu | Kanit |
|---|---|---|
| 1 | Kosulsuz konsolidasyon unutmayi W_slow'a tasir (x1.01) | DualMemory F3=0.379 |
| 2 | Slot hatasina W_fast dahil edilirse self-masking: W_fast aktif slotun hatasini telafi eder, gate kilitlenir | Tum konsolidasyonlar slot0'a |
| 3 | Zehirli slot (yanlis rejimle kontamine) argmin'i yaniltir: hatasi bos slottan da kotu olabilir | E_1=4.19 > var(y) |
| 4 | "En bos slot" icin ||W_slow|| normu hatali vekil: rejimlerin agirlik normlari kiyaslanamaz (kare ~4x sinus). Causal esik altinda norm da ortalama kazanc verir (0.216->0.187) AMA yanlis secimde kontaminasyonu KORUR -> 2/10 seed geriler (en kotu seed 0: 0.147->0.176, +%20; seed 5: +%18) | v2: 8 seed iyi, 2 seed kotu |
| 5 | cons_count sayaci dogru doluluk olcutu: hem dogru hedef hem dusuk varyans; 10/10 seed'de v2'yi yener | v3 0.104+/-0.02 vs v2 0.187+/-0.06 |
| 6 | PASIF koruma (Bereska) tek kanaldir: gate-floor asilmazsa mekanizma hic devreye girmez; write-triggered alloc kanali kismen kurtarir | MH 0.424->+alloc 0.377 (seed 1/6/8) |
| 7 | Konsolidasyon = W_fast basinc valfi: yazim reddedilirse W_fast siser (norm ~30), dogru slot bogulur | refuse 0.073 vs overwrite 0.016 |

**Duzeltmelerin matematigi:**
- Slot eslesme hatasi W_fast'i disarida birakir: E_i = ||y - beta*phi^T W_slow_i||^2
- Konsolidasyon kapisi: e_fast <= err_gate (gecis penceresinde yazma yok)
- Hedef secim: E_min < tau*m2 ise REINFORCE, degilse argmin(cons_count) NEW_CTX
  (doygunlukta: 'refuse' atla | 'overwrite' argmin c_i eviction)
- Transfer: W_slow[k] += (1-gamma)/beta * W_fast ; W_fast *= gamma
  (emin gate'te w_k~1 iken sureklilik makine epsiloni duzeyinde: 3.4e-15)

## 4. Farkli dinamiklerde genelleme (5 seed)

| Test | Fazlar | Block | MultiHead | CTX F3-onset | Kazanc | Slot deseni |
|---|---|---|---|---|---|---|
| A | MG17->MG30->MG17 | 1.40e-2 | 1.40e-2 | 1.38e-2 | x1.01 | [0,0,0] x5 |
| B | NARMA10->MG17->NARMA10 | 0.108 | 0.108 | 0.103 | x1.05 | [0,1,0] x3 |

**Anahtar bulgu: CTX kazanci unutma maliyetiyle orantili.**
- Rejimler ayrismazsa (MG17~MG30): slot hatalari farklilasamaz, mekanizma zararsiz
  bicimde pasiflesir. Korunacak sey yok -> kazanc yok.
- Yeniden ogrenme ucuzsa (NARMA-10: baseline onset ~ end): slot atama
  ([0,1,0] 3/5) ama kazanilacak marj kucuk -> x1.05. (m2'nin mu^2
  terimi ayrismayi zayiflatir: oracle-var altinda 5/5 idi — nedenselligin
  olculen bedeli.)
- Unutma pahaliysa (sinus<->kare): x4.40.

Bu, "mekanizma her zaman iyilestirir"den daha guclu bir iddia: **zararsiz ve
kosullu fayda saglar**; en kotu tek vaka gozlemi %14 maliyet payiyla sinirli.

## 5. Bilinen sinirlamalar (makalede acikca raporlanacak)

1. EMA gecikmesi kor noktasi: rejim gecisinin ilk ~10 adiminda E hala dusuk
   gorunur -> 1-2 kontamine REINFORCE yazimi kacabilir (seed0: t=1500,1750).
2. Soft gate gecis maliyeti: bootstrap/esit-slot durumunda w_k<1 yazimlari
   sinirli tahmin sIçramasi uretir (gozlenen maks 1.49).
3. F3-end maliyeti: CTX 3.7e-2 vs block 1.0e-2 (3.6x; y[-1] wrap bug fix sonrasi) — gamma=0.1 sönümleme son
   yakinsamayi hafif yavaslatir.
4. Onset'in kalan ~0.09'u = gate gecis gecikmesi + W_fast interferansi
   (future work: emin slotta W_fast katkisinin dinamik kisilmasi).
5. Tek mimari/tek parametre seti: D=100,K=10,tau=0.5,eta=2 — parametre
   taramasi yapilmadi.

## 6. Tekrarlanabilirlik

- `python benchmark_block_rls.py` : okuma motoru karakterizasyonu (10 seed)
- `python continual_protocol.py` : sinus->kare->sinus (10 seed, 4 yontem)
- `python different_tasks.py` : MG-rejim ve NARMA-MG genelleme testleri (5 seed)
- Tum RNG `np.random.default_rng(seed)` ile yerel; seed=0..9/0..4.

## 7. Native-LM kolu: 6-arm x 3-seed causal grid (apia-v3-causal, Kaggle)

`native_lm_v3_causal.json` — tinyshakespeare char-LM, A->B->A->B->A
donus akisi (5 faz), frozen char-embedding ozellikleri uzerinde APIA.
Gate-ordering fix SONRASI ilk tam matris (onceden hardgate/oracle/
forced/detect tek seed idi). Birim: bits-per-char (dusuk = iyi).

| Arm | final BPC | return-onset (A2,B2,A3) | vs block |
|---|---|---|---|
| apia_block | 5.658 | 5.776 +/- 0.004 | — |
| apia_ctxmem (soft) | 5.660 | 5.773 +/- 0.014 | -0.003 +/- 0.014 |
| apia_ctxmem_forced | 5.658 | 5.776 +/- 0.004 | +0.000 (cons=0) |
| apia_ctxmem_detect | 5.665 | 5.784 +/- 0.014 | +0.008 +/- 0.014 |
| apia_ctxmem_hardgate | **5.632** | **5.653 +/- 0.107** | **-0.123 +/- 0.108** |
| apia_ctxmem_oracle | 5.634 | 5.674 +/- 0.074 | -0.102 +/- 0.075 |

Okuma:
- Soft gate (paper-1 mekanizmasi) native-LM ozelliklerinde kazanc
  saglamiyor (-0.003) — reservoir regresyonundaki x4.40 burada yok.
- HARD gate ise gercek kazanc veriyor: return-onset'te -0.12 BPC,
  final'de -0.026 BPC; ustelik oracle'i bile yakaliyor/asiyor
  (gate kalitesi darbogaz degil — slot icerigi yeterli).
- Kazanc seed-bagimli: seed 0/1'de -0.16/-0.20, seed 2'de ~0.
- forced kolu kontrol olarak calisti: cons=0, block ile birebir ayni
  (tetik sarti hic saglanmadi — forced-write kosulu e_fast esigine
  bagli kaldi).
- detect (degisim-noktasi tetikli) hafif kotu: +0.008.
- Sonuc, paper-2'nin "bellek faydasi ozellik-uzayina bagli"
  anlatisiyla tutarli: siniflandirma ozelliklerinde (GPT-2/OPT) soft
  gate faydasiz, char-LM'de ise sadece hard atama kazanc veriyor —
  mekanizma transfer ediyor ama fayda gate tasarimina duyarli.
