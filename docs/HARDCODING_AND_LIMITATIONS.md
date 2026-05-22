# Hardcodări, Euristici și Limitări în Nexus Cortex

> **Scop**: Acest document declară toate valorile implicite de operare, pragurile euristice, datele de seed (inițializare) și limitările structurale cunoscute din codebase-ul **Nexus Cortex**. Nimic nu este ascuns — acesta este inventarul onest și complet al sistemului.

---

## 1. Date de Inițializare / Seed (Configurabile prin Config)

Toate datele de inițializare sunt acumocate în câmpurile structurii `Config` (vezi [config.go](file:///D:/Nexus%20cortex/cortex/config.go)) și pot fi suprascrise complet prin intermediul unui fișier JSON de configurare sau prin construcție programatică.

### Topici de Inițializare Automată (`AutoSeedTopics`)
* **Valoare Implicită**: 32 de topici acoperind domenii precum știință, matematică, tehnologie, istorie, geografie, logică și cultura română. Acestea au rolul de a stimula curiozitatea modulului `AutonomousLearner`. Ele nu reprezintă date de antrenament pre-existente, ci interogări de căutare pentru Wikipedia/HuggingFace.

### Seturi de Date de Inițializare (`AutoSeedDatasets`)
* **Valoare Implicită**: `tatsu-lab/alpaca`, `gsm8k`, `hellaswag`. Acestea reprezintă seturi publice cunoscute de instrucțiuni și raționamente utilizate pentru procesul inițial de învățare. Ele pot fi modificate sau golite complet din configurație.

### Limbi de Căutare (`AutoSearchLangs`)
* **Valoare Implicită**: `["en", "ro"]`. Determină limbile în care sunt interogate API-urile Wikipedia.

---

## 2. Date de Evaluare (Benchmark)

Fișierul [data/evals/comprehensive.jsonl](file:///D:/Nexus%20cortex/data/evals/comprehensive.jsonl) conține aproximativ 30 de cazuri de test structurate în 4 categorii majore: *recall*, *generalization*, *wikipedia* și *reasoning*. Acesta reprezintă un set redus de evaluare destinat exclusiv iterațiilor rapide în dezvoltare, nu emiterii de pretenții de capabilitate generală. Răspunsurile așteptate sunt șiruri fixe de caractere, iar evaluarea se face prin scoring parțial (Jaccard).

De asemenea, în [data/evals/hidden_benchmark.jsonl](file:///D:/Nexus%20cortex/data/evals/hidden_benchmark.jsonl) a fost integrat un set robust de exact 100 de cazuri de test de înaltă fidelitate pentru a împiedica supra-ajustarea (overfitting) pe setul de testare vizibil.

> [!WARNING]
> Acest set de evaluare este prea mic pentru a susține afirmații privind inteligența artificială generală (AGI). El testează exclusiv capabilitățile de bază de recuperare a informației (retrieval) și de raționament simbolic rigid.

---

## 3. Motorul de Raționament Determinist (`reasoning.go`)

Motorul de raționament utilizează reguli stricte și potriviri de tip pattern-matching **simbolic/bazat pe reguli**:
* **Aritmetică**: parser de expresii bazat pe expresii regulate (regex).
* **Secvențe**: detecția progresilor aritmetice și geometrice simple.
* **Logică**: parsarea silogismelor simple ("All X are Y. Z is X. Is Z Y?").
* **Sortare**: parsarea și sortarea listelor de valori numerice.

> [!IMPORTANT]
> **Acesta NU este un raționament neural emergent.** Este un mecanism determinist simbolic (așa-numitul "regex shim") care oferă răspunsuri corecte pentru interogări matematice și logice pre-structurate. El nu trebuie prezentat ca dovadă a unei capabilități de gândire emergente a rețelei neuronale spiking.

---

## 4. Valori Implicite de Operare

| Setare | Valoare Implicită | Câmp Config | Rol Funcțional |
| :--- | :--- | :--- | :--- |
| **Port Server Dashboard** | `8080` | `Config.WebPort` | Portul de ascultare al serverului HTTP Dashboard. |
| **Adresă Bind Web** | `127.0.0.1` | `Config.WebBindAddr` | Adresa IP pe care se bindează serverul. |
| **Seed RNG** | `42` | `Config.Seed` | Seed-ul utilizat pentru generarea pseudo-aleatoare. |
| **Dimensiune SDR** | `10000` | `Config.SDRSize` | Numărul total de biți ai vectorului sparse distribuite. |
| **Biți Activi SDR** | `50` | `Config.ActiveCount` | Numărul exact de biți setați pe 1 în SDR (sparsity de 0.5%). |
| **Capacitate Max Memories** | `10000` | `Config.MaxMemories` | Limita maximă de memorii stocate în Hippocampus. |
| **Interval Învățare Auto** | `30s` | `Config.AutoLearnInterval` | Frecvența cu care rulează bucla autonomă. |
| **Timeout HTTP Web** | `10s` | `Config.WebLearnerTimeoutSecs` | Limita de timp pentru interogările externe efectuate. |
| **Rată Limită HTTP** | `2s` | `Config.WebLearnerRateLimitMs` | Pauza obligatorie între request-urile către Wikipedia. |
| **Limită Corp Request** | `5 MB` | `Config.WebLearnerBodyLimitMB` | Dimensiunea maximă a răspunsurilor HTTP descărcate. |
| **User-Agent HTTP** | `NexusCortex/1.0 (autonomous learner)` | `Config.WebLearnerUserAgent` | Identificatorul HTTP trimis în headerele de request. |
| **Timeout WebGPU** | `5s` | `Config.WebGPUTimeoutSecs` | Timpul maxim de așteptare pentru alocările WebGPU. |

Toate aceste câmpuri pot fi modificate fără recompilare prin editarea fișierului de configurare JSON central al sistemului.

---

## 5. Permisiuni de Fișiere și Securitate

* **Fișiere de Stare și Modele**: Toate fișierele de persistență (cum ar fi `.nxt1`, `.nxbrain`, `.json`) utilizează permisiunea octală strictă `0600` (citire și scriere permisă exclusiv proprietarului procesului).
* **Directoare**: Toate directoarele create în timpul execuției (inclusiv folderul de date `./data/cortex/`) utilizează permisiunea strictă `0700` (acces total exclusiv proprietarului).
* **Autentificare Dashboard**: Serverul dashboard generează automat la startup un token criptografic unic de 32 de caractere hexazecimale prin `crypto/rand` (dacă nu este specificat explicit prin `-token`). Acest token este obligatoriu în toate apelurile POST modificatoare și în cele de citire a telemetriei (`/api/stats`).
* **Protecție CSRF / Same-Origin**: Serverul verifică strict potrivirea headerelor `Origin` sau `Referer` (folosind net/url parser) cu portul local de rulare pentru a preveni atacurile de tip Cross-Site Request Forgery.
* **Protecție SSRF**: Modulul `WebLearner` validează schemele adreselor interogate (exclusiv `"https"` securizat) și restricționează domeniile prin intermediul unei whitelist-ări severe (`wikipedia.org`, `huggingface.co`), inclusiv pe fiecare hop de redirect.

---

## 6. End-point-uri API Externe Accesate

| Endpoint de Bază | Scop | Câmp Config Asociat |
| :--- | :--- | :--- |
| `*.wikipedia.org` | Căutare și extragere articole enciclopedice | `Config.WebLearnerWikiBaseURL` |
| `huggingface.co/api/datasets` | Căutare seturi de date de antrenament | `Config.WebLearnerHFSearchURL` |
| `datasets-server.huggingface.co/rows` | Descărcare rânduri cu instrucțiuni brute | `Config.WebLearnerHFRowsURL` |

Sistemul nu utilizează chei de API hardcodate; toate interogările sunt publice și respectă politicile de rate-limiting definite în `Config`.

---

## 7. Scriptul de Descărcare Python (`scripts/download_hf_corpus.py`)

* **Securitate**: Utilizează parametrul `trust_remote_code=False` în toate apelurile bibliotecii `transformers/datasets` pentru a preveni execuția de cod malițios.
* **Versiuni Wikipedia Pinned**: Dump-urile Wikipedia sunt blocate pe versiuni istorice specifice (`"20231101.ro"` și `"20231101.en"`).
* **Plafonări și Eliminări**: Lungimea maximă a frazelor este limitată la 50 de cuvinte, iar fragmentele sub 3 cuvinte sunt respinse automat pentru a preveni zgomotul semantic.

---

## 8. Defecte Logice și Limitări Structurale Identificate în Audit

Următoarele aspecte reprezintă limitări structurale severe ale implementării curente, descoperite în timpul auditului static complet din 22 Mai 2026:

### 1. Defect Critic în Generalizarea Conceptelor ([semantic_memory.go:L150-180](file:///D:/Nexus%20cortex/cortex/semantic_memory.go#L150-L180))
Consolidarea conceptelor tinere aplică uniunea dintre prototipul existent și intersecția cu noul input. Matematic, această operație reprezintă o identitate:
$$\text{Prototype} \cup (\text{Prototype} \cap \text{Input}) \equiv \text{Prototype}$$
Din acest motiv, conceptele tinere nu învață niciodată elemente noi din episoade ulterioare, micșorându-se monoton la fiecare consolidare prin somn, până la pierderea completă a biților activi (devenind SDR-uri vide).

### 2. Defect Critic în Calculul Stabilității Prefrontale ([prefrontal.go:L120-145](file:///D:/Nexus%20cortex/cortex/prefrontal.go#L120-L145))
Funcția `measureStability` include neuronii inactivi (tăcuți) în calculul coeficientului de similaritate. Într-o rețea sparse activă la ~5%, componenta tăcută de 95% domină total scorul, ducând la o valoare de încredere/stabilitate blocată artificial în jurul a ~95%, anulând eficacitatea escaladării în lanțul decizional.

### 3. Defect Critic de Dublă Incrementare în Encoder ([encoder.go:L180-205](file:///D:/Nexus%20cortex/cortex/encoder.go#L180-L205))
Funcția `EncodeSentence` incrementează manual variabila `ActiveCount` a SDR-ului de propoziție, deși apelul prealabil `sdr.Set(bit)` realizase deja această incrementare la nivel intern. Acest lucru corupe valoarea densității raportate, fiind dublă față de realitate, afectând calculele de similaritate.

### 4. Limitarea "Shared-Weights" în Fractal Cortex ([fractal_cortex.go:L180-210](file:///D:/Nexus%20cortex/cortex/fractal_cortex.go#L180-L210))
Neurogeneza adaugă layere noi prin copierea directă a memoriei fizice a ponderilor părinte. Fără o perturbație stocastică de rupere a simetriei, noul bloc aplică exact aceleași transformări liniare/ternare, reprezentând o creștere iluzorie fără parametri independenți reali.

### 5. Căutare Liniară $O(N)$ în Hippocampus ([hippocampus.go:L85-110](file:///D:/Nexus%20cortex/cortex/hippocampus.go#L85-L110))
Retragerea informației episodică parcurge secvențial toate înregistrările, limitând masiv scalabilitatea latenței în timp real pe măsură ce baza de cunoștințe crește.

---

## 9. Ce NU Este Hardcodat în Nexus Cortex

* **Fără Secrete sau Token-uri de API**: Codebase-ul nu conține chei private, parole sau token-uri de autentificare expuse în cod.
* **Fără Căi Absolute de Sistem**: Nu există dependențe de structuri absolute de directoare de pe mașina de producție; toate rutele de stocare implicite sunt relative (`./data/cortex`).
* **Ponderi Ne-Hardcodate**: Nu există coeficienți neurali statici definiți în cod; toate ponderile sunt fie generate stochastic la inițializare, fie ajustate dinamic prin reguli de plasticitate.
