# Makale Taslagi — paper_outline.md

**Calisma basligi:**
"Cost-Aware Contextual Memory for Online Continual Learning in
Reservoir-Based Time Series Prediction"

Alternatifler:
- "Slot-Based Contextual Memory Enables Forgetting Resistance in Online
  Reservoir Computing"
- "Write-Protected Contextual Consolidation for Backpropagation-Free
  Continual Learning"

---

## Ozet (Abstract) — taslak

Online surekli ogrenen sistemler degisen ortamlara uyum saglarken eski
bilgiyi koruma gerilimiyle karsi karsiyadir. Bu calisma, rezervuar tabanli
zaman serisi tahmini icin dusuk maliyetli baglamsal bir bellek mekanizmasi
sunmaktadir. Sistem; seyrek sabit Echo State Network (D=100) ve B-spline
oznitelik haritasi (D*K+1=1001 oznitelik) ile dinamik temsil uretir,
Block-Diagonal RLS ile cevrimici guncellenir (~75x hiz, ~100x bellek
avantaji, iz-sinirli P-bounding ile lambda<1'de kararli) ve N-slot
baglamsal cift-hizli bellek ile rejim degisimlerini izler. Bes asamali
mekanizma analizi, kosulsuz konsolidasyonun unutmayi cozmedigini
(yavas bellege tasidigini), slot eslesme hatasinin hizli bellek
tarafindan maskelendigini (self-masking) ve doluluk icin agirlik
normunun hatali bir vekil oldugunu gostermektedir. Onerilen
context-aware write-protection ve konsolidasyon sayaci ile sinus->kare->
sinus rejim protokolunde donus-onset NMSE 0.459'dan 0.096'e dusmektedir
(x4.76, 10 seed, 10/10 pozitif). Mekanizma NARMA-10 ve Mackey-Glass
gorevlerinde baglamsal ayrisma ile genellesmekte; kazancin unutma
maliyetiyle orantili oldugu, ayrisamayan rejimlerde ise zararsiz bicimde
pasiflestigi gosterilmektedir. Tum ogrenme yerel ve cevrimicidir
(backprop yok); ~53us/adim islem ve O(D*K^2) bellek ile edge deployment
icin uygundur.

Anahtar kelimeler: continual learning, reservoir computing, echo state
network, contextual memory, catastrophic forgetting, online learning,
edge AI

---

## Katki Listesi

1. **Dusuk maliyetli cevrimici okuma motoru:** B-spline oznitelikli
   Block-Diagonal RLS + trace-tabanli P-bounding; Tam RLS'e gore
   ~75x hiz, ~100x kovaryans bellegi avantaji ve lambda<1'de kararlilik.

2. **Baglamsal surekli ogrenme mekanizmasi:** N-slot yavas bellek,
   W_fast'ten arindirilmis slot eslesme hatasi, EMA slot secimi,
   analitik transfer kurali (sureklilik korumali) ve cons_count
   write-protection.

3. **Mekanizma kesif sureci (ablation hikayesi):** Kosulsuz
   konsolidasyon -> self-masking -> slot poisoning -> norm hatasi ->
   cons_count: her asamasi olculmus 5 katmanli deneysel analiz.

4. **Kosullu-fayda teoremi ampirik kaniti:** Kazancin unutma
   maliyetiyle orantili oldugu; ayrisamayan rejimlerde mekanizmanin
   zararsiz pasiflestigi gosterimi (x4.76 / x1.05 / x1.00).

---

## Bolum Yapisi

### 1. Introduction (~1.5 sayfa)
- Surekli ogrenme problemi ve katastrofik unutma
- Backprop gerektirmeyen cevrimici ogrenmenin motivasyonu (edge/enerji)
- Rezervuar hesaplama + RLS okumanin dogal uyumu
- Bosluk: varolan RC calismalari rejim-donusu hatirlamayi
  adreslemiyor; replay/EWC agir / backprop bagimli
- Katki ozeti (4 madde)

### 2. Related Work (~1.5 sayfa)
- Continual learning: EWC, SI, replay, regularization tabanli
- Reservoir computing: ESN, ridge/RLS readout, FORCE, ESP literaturu
- Mixture-of-experts / context gating benzerlikleri ve farklar
- Edge/duşuk-guclu cevrimici ogrenme (SNN/HDC kiyasina kisa deginme)

### 3. Method (~3 sayfa)

**3.1 Sistem genel bakis** — Sekil 1 (mimari diyagram)
- u_t -> seyrek ESN (leaky alpha=0.3, tanh, rho(W)~0.97)
- x_t -> B-spline oznitelik phi_t (D*K+1)
- W_fast: BlockDiagRLS (lam=0.98, init_var=1e2, trace-bound)
- N_slot=3 yavas bellek + soft gate + cons_count

**3.2 Rezervuar ve oznitelik haritasi**
- ESP yeterli kosulu: (1-alpha)+alpha*||W||_2<1 ile olcekli olusturma
- Spline: per-bilesen K=10 kubik baz, D*K+1=1001

**3.3 Block-Diagonal RLS**
- Ortak payda s = lam + sum_j phi_j'P_j phi_j (blok-diyagonal yaklasim)
- Maliyet: O(D*K^2) hesap/bellek vs Full RLS O(F^2)
- Trace-bounding: P *= bound/trace(P) — kovaryans windup cozumu

**3.4 Baglamsal bellek**
- Slot hatasi: E_i = ||y - beta*phi'W_slow_i||^2 (W_fast DISARIDA —
  self-masking onlemi, Kesif #2)
- Soft gate w = softmax(-eta*E), eta=2
- Konsolidasyon kapilari: aralik (250 adim) + e_fast <= err_gate
- Hedef secim: E_min < tau*m2 (tau=0.5) -> REINFORCE;
  m2 = E[y^2] icin online EMA (alpha=0.02); global/future istatistik YOK.
  degilse argmin(cons_count) -> NEW_CTX (Kesif #4, #5)
- Transfer: W_slow[k] += (1-gamma)/beta * W_fast ; W_fast *= gamma
  (w_k~1'de sureklilik ~3e-15; Kesif matematigi)
- P enflasyonu: inflate_P(p_inflate=1.5) — plastisite restorasyonu

### 4. Experiments (~3 sayfa)

**4.1 Kurulum**
- Gorevler: sinus->kare->sinus (10 seed), MG17->MG30->MG17 (5),
  NARMA10->MG17->NARMA10 (5); Mackey-Glass RLS karakterizasyonu (10)
- Metrikler: faz-penceresi NMSE (onset=ilk 250, end=son 250), ms/adim,
  bellek, slot desenleri, konsolidasyon gunlukleri
- Yontemler: Ridge, FullRLS+bound, BlockRLS+bound, +naive, +CTX

**4.2 RLS karakterizasyonu** — Tablo 1, Sekil 2

**4.3 Ana protokol** — Tablo 2, Sekil 3, Sekil 4

**4.4 Ablation** — Tablo 3 (5 katman)

**4.5 Genelleme** — Tablo 4 (Test A/B)

**4.6 Maliyet profili** — Tablo 5

### 5. Discussion (~1.5 sayfa)
- 5 kesif katmaninin bilimsel anlami (her biri ayri ders)
- Kosullu-fayda yorumu: mekanizma "ne zaman yardim etmeyecegini" biliyor
- Sinirlamalar: EMA kor noktasi (~10 adimlik kontaminasyon penceresi),
  soft-gate bootstrap sicramasi (maks ~1.5), F3-end maliyeti
  (3.5e-2 vs 1.6e-2), tek parametre seti (tau/eta/n_slot taramasi yok)
- Future work: emin slotta W_fast kisitlamasi (B secenegi), dinamik
  slot acma, daha fazla gorev ailesi, SNN/HDC karsilastirmasi,
  fotnik/neuromorfik donanim haritalama

### 6. Conclusion (~0.5 sayfa)
- x4.76 hatirlatma kazanimi + zararsiz pasiflesme
- Yerel kurallarla backprop'suz surekli ogrenme mumkun ve olculmus
- Maliyet profili edge'e uygun

---

## Sekil ve Tablo Plani (gercek olcumlerle)

**Sekil 1:** Sistem mimarisi (reservoir->spline->RLS->slotlu bellek).
**Sekil 2:** P-bounding etkisi: lambda=0.98'de FullRLS NMSE patlamasi
             (eski kosum ~0.54->inf) vs sinirli kararlilik.
**Sekil 3:** Sinus->kare->sinus hata trajektoru: block vs ctx
             (kayan pencere NMSE, faz sinirlari isaretli).
**Sekil 4:** Slot gate agirligi isi haritasi (seed0): faz1 slot0,
             faz2 slot1/2 dagilimi, faz3 slot0'a donus.
**Sekil 5:** Ablation bar grafigi: 5 varyant F3-onset (mean+/-std).

**Tablo 1 — RLS karakterizasyonu (MG, 10 seed):**
| Yontem | NMSE@5000 | us/adim | Kovaryans bellegi |
| FullRLS lam=1 | ~1e-3 | ~3960 | F^2 ~8MB |
| BlockRLS lam=1 | ~2e-2 | ~53 | D*K^2 ~78KB |
| BlockRLS lam=0.98+bound | kararli | ~53 | ayni |
| FullRLS lam=0.98+bound | kararli ama kirilgan | ~3960 | ayni |

**Tablo 2 — Ana protokol (10 seed, NMSE):**
| Yontem | F1-end | F2-end | F3-onset | F3-end |
| FullRLS+bound | 3.8e-11 | 0.108 | 8.17+/-4.0 | 6.2e-3 |
| BlockRLS+bound | 1.6e-4 | 0.545 | 0.459+/-0.20 | 1.6e-2 |
| +naive | 1.6e-4 | 0.543 | 0.456 | 1.6e-2 |
| +CTX | 2.4e-4 | 0.458 | **0.096+/-0.02** | 3.5e-2 |

**Tablo 3 — Ablation katmanlari (10 seed, F3-onset):**
| Varyant | NMSE | Duzeltilen/bulunan mekanizma hatasi |
| +naive | 0.456 | — (unutma W_slow'a tasindi) |
| +CTX v0 | 0.446 | orijinal bug: slot hatasi W_fast icerir |
| +CTX v1 | 0.207 | self-masking duzeltmesi: saf W_slow hatasi |
| +CTX v2 | 0.218 | write-protection: norm vekili varyans buyutur |
| +CTX v3 (final) | 0.096 | cons_count: dogru hedef + dusuk varyans |

Not: v2 ortalamasi v1'e esit ama seed-bazinda bolunmus (6 iyi, 2 kotu);
norm heuristiigi yanlis secimde kontaminasyonu korur -> cons_count
hem hedef dogrulugunu hem varyansi duzeltir (std 0.09->0.02).

**Tablo 4 — Genelleme (5 seed):**
| Gorev | Block | CTX | Kazanc | Desen |
| MG17->MG30->MG17 | 1.40e-2 | 1.40e-2 | x1.00 | [0,0,0] |
| NARMA->MG->NARMA | 0.108 | 0.098 | x1.05 | [0,1,0] x5 |
| sinus->kare->sinus | 0.459 | 0.096 | x4.76 | [0,*,0] baskin |

**Tablo 5 — Maliyet:**
| Bilesen | Hesap/adim | Bellek |
| ESN | O(nnz)~2000 MAC | 1000 seyrek |
| Spline | O(D*K)=1000 | — |
| BlockRLS | O(D*K^2)~10K | 78KB |
| CTX | O(N*F)~3K | 24KB (float64) |
| Toplam | ~53us | ~110KB (~0.1MB) |

---

## Hedef Venue

**Birincil:** IEEE SMC 2026 (6-8 sayfa, continual learning + edge AI
uyumu, konferans sonrasi dergi genisletmesi mumkun).
**Alternatif:** Neural Networks (Elsevier, IF~7.8) — tam ablation +
genelleme anlatisi dergi formatina da uygundur.

## Zaman cizelgesi
- Hf1: sekil uretimi (Fig3/4/5 icin veri zaten protokol ciktisinda)
- Hf2-3: Method+Experiments
- Hf4: Intro/Related/Discussion
- Hf5: gozden gecirme + basvuru

## Acik kalan opsiyonel isler (basvuru oncesi degerlendirilebilir)
- Fig3/4/5 uretim scripti (matplotlib, mevcut metriklerden)
- Wilcoxon paired test (block vs ctx, 10 seed) — p-degeri ile guclendirme
- Kod deposu duzenleme + MIT lisansi + tek-paragrafik README
