# Pose-based behavioural feature pipeline (prototipo)

Prototipo di pipeline per estrarre marker comportamentali quantitativi da
video di interazione, tramite pose estimation multi-persona (Ultralytics
YOLO-pose) e feature engineering su serie temporali di keypoint. Sviluppato
come progetto dimostrativo in preparazione allo stage sul progetto CHUV di
neurosviluppo infantile (analisi comportamentale video-based).

## Perché questo approccio

La letteratura su pose estimation applicata allo studio del neurosviluppo
infantile mostra due filoni principali, entrambi rilevanti per questo
prototipo:

- **General Movements Assessment (GMA) automatizzato**: tracking markerless
  di neonati/lattanti per la diagnosi precoce di paralisi cerebrale
  infantile (Tsuji, Nakashima, Hayashi et al., 2020, *Markerless
  Measurement and Evaluation of General Movements in Infants*, Scientific
  Reports 10:1422, doi:10.1038/s41598-020-57580-z; letteratura successiva
  che confronta backbone di pose estimation come OpenPose, MediaPipe e
  HRNet su video clinici di lattanti, es. *Marker-Less Video Analysis of
  Infant Movements for Early Identification of Neurodevelopmental
  Disorders*, Diagnostics, 2025, doi:10.3390/diagnostics15020136).
- **Screening dello spettro autistico via pose estimation 2D**: angoli
  articolari e dinamiche di movimento estratti da video RGB standard come
  marker oggettivi e quantificabili di sintomatologia ASD, con accuratezza
  di classificazione dell'80.9% (F1 0.818) in un modello addestrato su
  video di interazione clinica (Kojovic, Natraj, Mohanty, Maillart &
  Schaer, 2021, *Using 2D video-based pose estimation for automated
  prediction of autism spectrum disorders in young children*, Scientific
  Reports 11:15069, Università di Ginevra — studio particolarmente
  rilevante essendo condotto in Svizzera romanda; revisione sistematica in
  de Belen, Bednarz, Sowmya & Del Favero, 2020, *Computer vision in autism
  spectrum disorder research: a systematic review of published studies
  from 2009 to 2019*, Translational Psychiatry 10:333).

Questi lavori motivano le scelte progettuali della pipeline: multi-persona
(bambino + caregiver, non solo un soggetto isolato), feature su angoli
articolari e dinamica del movimento, e un indice esplicito di periodicità
per movimenti potenzialmente stereotipati.

**Importante**: le feature qui implementate sono un punto di partenza
esplorativo e tecnico, non marker diagnostici validati. Qualsiasi utilizzo
clinico richiede validazione su dati annotati, supervisione di personale
qualificato e approvazione del comitato etico competente.

## Struttura del progetto

```
pose_behavior_pipeline/
├── src/
│   ├── geometry.py        # utility geometriche condivise (angolo tra tre punti)
│   ├── keypoints.py       # schema COCO-17, costanti condivise (inclusi gli edge dello scheletro)
│   ├── pose_estimation.py # wrapper Ultralytics YOLO-pose + tracking
│   ├── features.py        # angoli, velocità, simmetria, ripetitività, sincronia
│   ├── anonymize.py       # blur del volto basato sui keypoint della testa
│   ├── viz.py             # overlay scheletro/mani/metriche/FPS (condiviso batch e live)
│   ├── gaze_head.py       # head-pose + proxy attenzione condivisa (MediaPipe FaceLandmarker)
│   ├── hands.py           # mani a livello di dita (MediaPipe HandLandmarker)
│   ├── pipeline.py        # orchestrazione batch (video registrato) + CLI
│   ├── reid.py            # re-identificazione real-time via firma antropometrica (uscita/rientro)
│   ├── chuv_features.py   # feature engineering del repository CHUV, replicato in tempo reale
│   └── live_demo.py       # riconoscimento IN TEMPO REALE (Canon R8 / webcam) + CLI
└── demo/
    ├── synth_data.py          # generatori di keypoint sintetici (bambino, caregiver)
    ├── synthetic_demo.py      # verifica batch di features.py su dati sintetici
    ├── live_render_check.py  # verifica la logica live (finestra rolling + overlay) senza camera/YOLO
    ├── gaze_math_check.py     # verifica head-pose/joint attention senza camera/MediaPipe
    ├── hand_math_check.py     # verifica mani/dita senza camera/MediaPipe
    ├── face_math_check.py     # verifica bocca/occhi/blink senza camera/MediaPipe
    ├── reid_check.py          # verifica re-identificazione (uscita/rientro + estraneo) senza camera/YOLO
    ├── reid_color_check.py    # verifica il segnale colore maglia/pantaloni della re-identificazione
    ├── chuv_features_check.py # verifica angoli/distanze/COM/derivate temporali di chuv_features.py
    └── demo_outputs/          # CSV, grafici e video generati dai demo
```

## Setup sul MacBook Pro M1

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Ultralytics userà automaticamente il backend Metal (`device="mps"`) su
Apple Silicon; con `yolov8n-pose.pt` ci si aspettano tipicamente 30-60+ FPS
a risoluzioni moderate (vedi benchmark citati più sotto).

## Esecuzione

**Demo su dati sintetici** (nessuna dipendenza da Ultralytics, gira ovunque):

```bash
cd demo
python synthetic_demo.py
```

**Pipeline completa su un video registrato**:

```bash
cd src
python pipeline.py --source /percorso/video.mp4 --fps 30 --out features.csv
```

## Riconoscimento in tempo reale dalla Canon EOS R8

`live_demo.py` è il punto di ingresso dedicato al riconoscimento live: legge
da una sorgente in streaming, disegna lo scheletro e le metriche
comportamentali direttamente sul video (overlay), e salva un log CSV a fine
sessione.

**Collegamento della R8 al Mac**, due opzioni:

1. USB + [EOS Webcam Utility](https://www.usa.canon.com/digital-cameras/eos-webcam-utility)
   (supporta macOS Sonoma/Sequoia/Tahoe): la R8 compare come webcam
   standard, semplice ma con risoluzione/latenza limitate dall'utility.
2. Uscita HDMI pulita della R8 + capture card USB-C (es. Elgato Cam Link):
   qualità e latenza migliori, anch'essa esposta come webcam standard.

In entrambi i casi la sorgente è accessibile via OpenCV con l'indice
webcam corretto (verificabile con
`python -c "import cv2;[print(i, cv2.VideoCapture(i).isOpened()) for i in range(4)]"`,
o controllando l'ordine in Impostazioni di Sistema > Privacy e Sicurezza >
Fotocamera).

```bash
cd src
python live_demo.py --source 0 --fps 30 --model yolov8n-pose.pt --device mps \
    --window-seconds 3 --blur-faces --out live_session.csv
```

Premi `q` nella finestra video per interrompere la sessione; le feature
accumulate (finestra scorrevole di `--window-seconds` secondi) vengono
salvate in `--out`.

**Budget di tempo per frame**: `demo/live_render_check.py` misura, su dati
sintetici e SENZA il modello YOLO, il tempo speso da feature extraction +
disegno dell'overlay: ~3.7 ms/frame in media nell'ambiente di sviluppo
usato per questo prototipo. A un target di 30 FPS (33.3 ms/frame) restano
quindi ~29.6 ms/frame di budget per l'inferenza YOLO-pose, coerente con i
benchmark pubblici di YOLOv8n-pose su Apple Silicon con backend MPS
(tipicamente 15-25 ms/frame per `yolov8n-pose` a risoluzioni moderate).
Il collo di bottiglia reale sarà quindi l'inferenza del modello, non la
logica di feature extraction/overlay.

**Pipeline batch su un video registrato** (caso d'uso principale per
l'analisi clinica, non vincolato al framerate):

```bash
cd src
python pipeline.py --source /percorso/video.mp4 --fps 30 --out features.csv
```

## Head-pose e attenzione condivisa (`gaze_head.py`)

Estensione opzionale che aggiunge, oltre allo scheletro COCO-17: head-pose
(yaw/pitch/roll) per persona, tramite MediaPipe Tasks FaceLandmarker, e un
proxy 2D semplificato di "attenzione condivisa" (quanto la testa di una
persona è orientata verso l'altra) — marker esplicitamente citato nella
letteratura ASD raccolta per questo progetto (coordinazione dello sguardo,
frequenza dei movimenti della testa).

**Setup** (una tantum, richiede connessione internet per scaricare il
modello):

```bash
pip install mediapipe
curl -L -o src/face_landmarker.task \
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
```

**Uso**:

```bash
cd src
python live_demo.py --source 0 --fps 30 --device mps --with-gaze \
    --face-model face_landmarker.task
```

Con `--with-gaze` attivo, l'overlay mostra una freccia dalla testa nella
direzione dello yaw stimato, e — quando sono tracciate esattamente due
persone — un punteggio "guarda ID X: 0.xx" che indica quanto la testa di
una persona è orientata verso l'altra.

**Limite metodologico da tenere presente**: è un proxy 2D con una singola
camera non calibrata, non un vero gaze-tracking 3D (che richiederebbe
calibrazione della camera, stima di profondità e idealmente landmark
dell'iride ad alta risoluzione). L'euristica assume che le due persone
siano a distanza comparabile dalla camera. È adatto per un prototipo/demo,
ma va validato empiricamente (es. confrontando lo yaw stimato con
rotazioni note della testa) prima di qualunque uso interpretativo — la
convenzione di segno di yaw/pitch/roll di MediaPipe stessa andrebbe
verificata sul proprio setup, non essendo testabile in un ambiente senza
fotocamera.

La sola logica matematica (decomposizione della matrice di rotazione,
euristica di attenzione condivisa) è verificata senza camera in
`demo/gaze_math_check.py`.

## Mani a livello di dita (`hands.py`)

Estensione opzionale che affianca al polso (unico punto disponibile in
COCO-17) i 21 landmark per mano di MediaPipe Tasks HandLandmarker: angolo
di flessione di ciascun dito, indice di apertura/chiusura della mano
(pugno vs mano aperta), e uno score di ripetitività calcolato sulla punta
dell'indice invece che sul solo polso — più sensibile a stereotipie fini
(es. sfregamento/movimento delle dita con polso relativamente fermo, non
distinguibile dal solo scheletro COCO-17).

**Setup** (una tantum):

```bash
pip install mediapipe
curl -L -o src/hand_landmarker.task \
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
```

**Uso** (combinabile con `--with-gaze`):

```bash
cd src
python live_demo.py --source 0 --fps 30 --device mps --with-hands \
    --hand-model hand_landmarker.task
```

Le mani rilevate da MediaPipe vengono associate al polso YOLO più vicino
(non all'etichetta "Left/Right" di MediaPipe, che dipende dal punto di
vista della persona e può risultare invertita se il feed è specchiato —
vedi nota nel docstring di `hands.py`). L'overlay disegna lo scheletro
della mano (21 punti) e mostra apertura/ripetitività per ciascuna mano
rilevata.

La logica matematica (angoli di flessione, indice di apertura, matching
mani-polsi) è verificata senza camera in `demo/hand_math_check.py`, con
due mani sintetiche (aperta e a pugno chiuso).

**Nota di performance**: con `--with-gaze` e `--with-hands` entrambi
attivi, ogni frame viene processato da tre modelli (YOLO-pose +
FaceLandmarker + HandLandmarker in sequenza) — su M1 è ragionevole
aspettarsi un calo di framerate rispetto alla sola pose estimation; se il
video risulta poco fluido, valuta di abilitarne uno alla volta o di usare
`yolov8n-pose.pt` (il più leggero) come modello di base.

## Bocca, blink, sopracciglia, postura e auto-contatto

Segnali aggiuntivi tutti a costo marginale rispetto a quanto già in
pipeline (riusano landmark/keypoint già estratti, nessun modello nuovo):

- **Apertura della bocca** (`gaze_head.mouth_aspect_ratio`): rapporto
  verticale/orizzontale sui landmark della bocca già forniti da
  FaceLandmarker (prima usati solo per l'head-pose). Proxy grezzo di
  vocalizzazione o di movimenti ripetitivi della bocca (mouthing).
  Attivo con `--with-gaze`, nessun flag aggiuntivo.
- **Ripetitività della bocca** (`features.repetitive_motion_score`
  applicato alla serie temporale del MAR): stessa logica FFT già usata per
  polso/dita, qui applicata all'apertura della bocca nel tempo. Proxy di
  mouthing/vocalizzazione ripetuta — vedi nota sulla lingua più sotto sul
  perché questa è la scelta più informativa disponibile, non un
  ripiego arbitrario. Attivo con `--with-gaze`.
- **Scuotimento (yaw) e annuimento (pitch) della testa**
  (`features.repetitive_motion_score` applicato ai segnali grezzi di yaw e
  pitch stimati da FaceLandmarker): stessa logica FFT già vista per
  polso/dita/bocca, ma qui applicata direttamente all'angolo (non alla sua
  velocità) perché yaw/pitch sono già segnali con segno intorno a uno zero
  naturale — questo evita l'artefatto di raddoppio della frequenza
  documentato più sotto per lo score su polso/dita (che usa la velocità
  perché la posizione del polso deriva insieme al corpo). Proxy di
  scuotimento ripetitivo della testa (rilevante come possibile stereotipia)
  o di annuimento (gesti comunicativi/rocking verticale). Attivo con
  `--with-gaze`.
- **Blink rate** (`gaze_head.eye_aspect_ratio` / `mean_eye_aspect_ratio`):
  Eye Aspect Ratio sui landmark degli occhi, con conteggio dei blink nella
  finestra scorrevole (soglia configurabile via `--blink-ear-threshold`,
  default 0.2 — valore di partenza tipico in letteratura per l'EAR
  classico, da validare sul proprio setup). Attivo con `--with-gaze`.
- **Sollevamento sopracciglia** (`gaze_head.eyebrow_raise_ratio` /
  `mean_eyebrow_raise`): distanza verticale tra sopracciglio e palpebra
  superiore, normalizzata sulla distanza interoculare (scala del volto).
  Valori alti = sopracciglio sollevato (es. sorpresa), valori bassi/negativi
  = sopracciglio abbassato/aggrottato (es. concentrazione, disappunto). Solo
  un indicatore geometrico grezzo, non un riconoscitore di emozioni: non
  distingue *perché* il sopracciglio si muove. Attivo con `--with-gaze`.
- **Escursione verticale** (`features.vertical_excursion`): quanto il
  centro-bacino sale/scende nella finestra scorrevole, normalizzato sulla
  lunghezza del busto — proxy di transizioni posturali (seduto/in
  piedi/accovacciato), non una stima di statura (richiederebbe calibrazione
  della camera, non fatta qui). Sempre attivo, nessuna dipendenza da
  MediaPipe.
- **Activity ratio** (`features.activity_ratio`): frazione di frame con
  energia di movimento sopra `--activity-threshold` (default 40.0, da
  calibrare — l'unità dipende dalla risoluzione video e dalla distanza
  dalla camera). Proxy di quanto tempo la persona passa in movimento vs.
  relativamente ferma. Sempre attivo.
- **Self-touch** (`features.self_touch_score`): quanto un polso è vicino
  alla testa, normalizzato sulla lunghezza del busto; `--self-touch-threshold`
  (default 0.5) definisce quando contare un frame come "auto-contatto".
  Proxy di frequenza di comportamenti mano-al-volto (autoregolazione,
  possibili comportamenti autolesivi) — non ne distingue il motivo. Sempre
  attivo, usa solo i keypoint YOLO già tracciati.

```bash
cd src
python live_demo.py --source 0 --fps 30 --device mps --with-gaze \
    --activity-threshold 40 --self-touch-threshold 0.5 --blink-ear-threshold 0.2
```

**Overlay visivo**: con `--with-gaze` attivo, oltre ai numeri nel riquadro
metriche, `viz.draw_face_signals` disegna sul volto un piccolo quadrilatero
sulla bocca (si apre/chiude seguendo il MAR), il contorno di ciascun occhio
(si "chiude" durante il blink) e una polilinea sopra ciascun sopracciglio —
prima di questa aggiunta questi segnali comparivano solo come testo, senza
alcun segno disegnato sul volto (a differenza di testa e mani).

**Perché non c'è il tracking della lingua**: MediaPipe FaceLandmarker
modella solo la *superficie* del volto (pelle, contorno labbra, palpebre,
sopracciglia — 478 punti), non strutture intraorali. La lingua non ha
landmark dedicati e non è visibile/segmentabile in modo affidabile con
questo modello (servirebbe un modello specializzato di intraoral tracking,
non disponibile negli strumenti usati qui, e comunque fragile per
illuminazione/angolo). La ripetitività della bocca (sopra) è il proxy più
vicino disponibile: se il bambino fa mouthing ripetuto o vocalizza in modo
stereotipato, il MAR oscilla periodicamente e lo score FFT lo cattura,
anche senza vedere la lingua direttamente.

La logica matematica di tutti i segnali di questa sezione (mouth/eye aspect
ratio, conteggio blink, sollevamento sopracciglia, escursione verticale,
activity ratio, self-touch score) è verificata senza camera in
`demo/face_math_check.py` (bocca/occhi/blink/sopracciglia) — le funzioni di
postura in `features.py` sono testate direttamente negli esempi del modulo
e riusano la stessa infrastruttura di `synthetic_demo.py`.

**Nota su tutte le soglie**: `activity-threshold`, `self-touch-threshold` e
`blink-ear-threshold` sono punti di partenza ragionevoli, non valori
validati clinicamente — vanno calibrati osservando i valori reali prodotti
dal proprio setup (risoluzione, distanza dalla camera, illuminazione) prima
di trarre conclusioni comportamentali.

## Più persone nell'inquadratura e persona target

YOLO+ByteTrack assegna già di default un `track_id` distinto a ciascuna
persona rilevata (es. bambino + caregiver), e li tratta separatamente per
tutta la pipeline: scheletro, angoli, energia, postura, self-touch vengono
calcolati e salvati per ognuno, con righe CSV separate per `track_id`. La
prima volta che una persona compare nel video, il suo ID viene stampato a
console (`Nuova persona rilevata: ID N`), utile per capire rapidamente chi
è chi senza dover riaprire il CSV a posteriori.

**Distinzione visiva a colpo d'occhio**: ogni `track_id` ha un colore
stabile e diverso (`viz.get_track_color`, ciclico su una palette di 8
colori), usato in modo coerente per lo scheletro della persona, il bordo
del suo riquadro metriche ed un'etichetta grande "ID N" disegnata sopra la
testa (`viz.draw_person_label`) — prima tutte le persone avevano lo stesso
scheletro verde ed era difficile distinguerle rapidamente guardando lo
schermo. Se è impostato `--target-track-id`, l'etichetta della persona
selezionata mostra anche "★ TARGET" per capire subito chi viene monitorato
per i segnali di volto/mani.

**Riquadro metriche in posizione fissa**: il riquadro di testo con i
numeri non segue più la testa della persona (come nelle prime versioni,
dove si spostava insieme a lei e poteva risultare invadente/coprire la
scena) — ora tutti i riquadri sono impilati in un punto fisso in alto a
sinistra dello schermo, uno sotto l'altro, ordinati nell'ordine in cui
`frame_result.people` li restituisce. Il collegamento visivo con la
persona giusta resta comunque chiaro grazie al bordo colorato (stesso
colore dello scheletro/etichetta) e alla prima riga "ID N" di ciascun
riquadro.

**Limite da conoscere**: `track_id` persiste finché ByteTrack riesce a
seguire la persona nel tempo, ma se qualcuno esce completamente
dall'inquadratura e rientra dopo un po' (o cambia aspetto, es. vestiti),
può ricevere un nuovo ID diverso da quello originale — ByteTrack fa
matching solo su movimento (Kalman) + IoU, nessun segnale d'aspetto. Per
sessioni brevi in una stanza (il caso d'uso tipico qui) è raramente un
problema, ma va tenuto presente per sessioni lunghe con ingressi/uscite
frequenti. Vedi sezione successiva per un prototipo che attenua (non
elimina) questo limite.

## Re-identificazione dopo uscita/rientro (`reid.py`)

Estensione opzionale, disattivata di default, che affronta direttamente il
limite appena descritto: quando una persona esce completamente
dall'inquadratura e rientra con un nuovo `track_id` (per ByteTrack è
indistinguibile da uno sconosciuto), `reid.py` prova a riconoscerla e a
ripristinare il suo `person_id` originale, usando una **firma
antropometrica** — i rapporti tra segmenti corporei (larghezza spalle/anche,
avambraccio, coscia, ecc.), tutti normalizzati sulla lunghezza del busto
(`features.torso_length`, già usata altrove in questa pipeline). Essendo un
rapporto tra lunghezze, la firma è invariante alla distanza dalla camera e,
soprattutto, **al vestiario** — a differenza di un matching basato su
colore/aspetto dei vestiti, un cambio d'abito fra un'uscita e un rientro
non la altera.

Stessa strategia prototipata anche per la pipeline batch CHUV
(`reid_signature.py` in Video-Annotation-System, schema BODY-25), ma
validata prima qui: quella pipeline richiede SAM3 su GPU CUDA (non
disponibile su MacBook M1), mentre questo modulo gira in tempo reale su
dati non protetti, permettendo iterazione rapida prima di riproporre
l'approccio nel repository CHUV.

**Come funziona, in breve**: ogni nuovo `track_id` riceve subito un
`person_id` provvisorio (nessun ritardo nell'overlay). In parallelo si
accumula un piccolo buffer di firme; una volta raccolti frame a sufficienza
(`--reid-min-signature-frames`, default 15), la firma mediana viene
confrontata con le persone scomparse di recente (`--reid-max-lost-seconds`,
default 30s). Sotto una soglia di distanza (`--reid-max-signature-dist`,
default 0.12) il `person_id` viene ripristinato — solo per i frame
**successivi** al match, senza riscrivere quanto già salvato/disegnato con
l'id provvisorio. Ogni merge viene stampato a console e registrato in un
log interno (`ReIdentifier.merge_log`) per trasparenza.

**Uso**:

```bash
cd src
python live_demo.py --source 0 --fps 30 --device mps --with-reid \
    --out live_session.csv
```

Combinabile con `--with-gaze`/`--with-hands`/`--target-track-id` senza
modifiche: il resto della pipeline continua a trattare l'id come una chiave
generica, che sia il `track_id` grezzo di ByteTrack o il `person_id`
stabile di `reid.py`.

**Segnale opzionale: colore maglia/pantaloni.** La sola firma
antropometrica può essere troppo debole quando i keypoint sono rumorosi
(occlusioni parziali, persona ai bordi dell'inquadratura proprio durante
l'uscita/rientro) — in pratica un rientro vero può non superare la soglia.
Con `--with-reid` attivo, `live_demo.py` passa automaticamente anche il
frame corrente a `reid.py`, che campiona il colore medio (tonalità +
saturazione, non luminosità — più robusto a cambi di esposizione) di
maglia e pantaloni dal poligono spalle/anche/ginocchia. Il colore **non
sostituisce né alza mai la soglia** sulle proporzioni: se i vestiti sono
diversi (vero cambio d'abito) il match si decide come prima, solo sul
corpo; se i vestiti sembrano gli stessi, la distanza tra le proporzioni
viene "scontata" (peso configurabile con `--reid-color-weight`, default
0.5), rendendo più facile recuperare un rientro vero anche con proporzioni
un po' rumorose — senza mai compromettere l'invarianza al vestiario che
era l'obiettivo originale del prototipo.

```bash
cd src
python live_demo.py --source 0 --fps 30 --device mps --with-reid \
    --reid-color-weight 0.5 --out live_session.csv
```

**Limiti onesti**:

- Serve un numero minimo di frame con buona confidenza sui giunti chiave
  per costruire una firma affidabile; un passaggio molto breve
  nell'inquadratura non produce mai una firma e resta un `person_id` a sé.
- Due persone di corporatura simile (ed eventualmente di colore di vestiti
  simile) possono generare un falso positivo di merge — le soglie di
  default sono prudenti ma **non sono state validate su dati reali**, vanno
  calibrate osservando le distanze effettive nel proprio setup.
- In tempo reale il merge è automatico: non c'è modo di chiedere conferma a
  un essere umano nell'istante in cui avviene (a differenza del prototipo
  batch per CHUV, pensato per revisione umana) — per questo resta sempre
  tracciato nel log invece di sparire silenziosamente.
- La logica è verificata su scenari sintetici (uscita/rientro con
  proporzioni pulite e con proporzioni deliberatamente rumorose, persona
  estranea, cambio vestiti) in `demo/reid_check.py` e
  `demo/reid_color_check.py`, non su video reali — le soglie di default
  vanno ricalibrate osservando il comportamento reale sulla propria camera.

**Limitare i segnali di volto/mani a una sola persona fissa** (es. non
calcolare blink/bocca/sopracciglia/scuotimento-testa/dita per il
caregiver, solo per il bambino): usa `--target-track-id`. Prima prova
senza il flag per scoprire quale ID corrisponde a chi (dalla stampa a
console o dall'etichetta "ID N" sull'overlay), poi rilancia:

```bash
cd src
python live_demo.py --source 0 --fps 30 --device mps \
    --with-gaze --with-hands --target-track-id 1 --out live_session.csv
```

Con `--target-track-id` impostato, scheletro/postura/energia restano
calcolati e salvati per **tutte** le persone rilevate (utili per contesto e
per feature diadiche come `windowed_synchrony`), ma i segnali che
richiedono FaceLandmarker/HandLandmarker (blink, bocca, sopracciglia,
scuotimento/annuimento testa, apertura/ripetitività dita) vengono
calcolati **solo** per il `track_id` scelto — sia nel CSV (colonne a NaN
per le altre persone) sia nell'overlay video (nessun disegno su volto/mani
per chi non è il target). Nota: FaceLandmarker/HandLandmarker processano
comunque l'intero frame e rilevano tutti i volti/mani presenti (il costo
del modello non cambia con `--target-track-id`) — il filtro scarta i
risultati delle persone non target subito dopo, risparmiando solo il
calcolo delle feature derivate (buffer, FFT) per loro, non l'inferenza del
modello stesso.

## Feature engineering del repository CHUV, in tempo reale (`chuv_features.py`)

Estensione opzionale, disattivata di default, pensata per un obiettivo
preciso: testare la STRATEGIA di feature engineering del repository CHUV
(Video-Annotation-System) — gli stessi angoli, distanze, indici di
simmetria, centro di massa e derivate temporali definiti in
`src/models/train.py` di quel repository — senza bisogno di GPU CUDA né di
video clinici protetti, usando questa pipeline leggera (YOLO + ByteTrack)
come banco di prova. Non è un adattatore che collega le due pipeline: è la
stessa logica di calcolo, riscritta per lavorare frame-per-frame su
COCO-17 invece che offline su un CSV BODY-25 già completo.

**Cosa calcola**: per ogni persona tracciata, ad ogni frame — coordinate
normalizzate rispetto al bacino (stessa convenzione di `normalize_keypoints`
nel repository CHUV: `(x - mid_hip_x) / lunghezza_busto`), 9 angoli
articolari (gomiti, spalle — con vertice al collo, diversa dalla
definizione già presente in `features.py` che usa il vertice-anca —,
ginocchia, anche, tronco), 8 distanze normalizzate (occhio-occhio,
naso-collo, polso-anca, polso-naso, naso-caviglie, anca-caviglie), 6
differenze di simmetria sinistra/destra, centro di massa e "spread"
corporeo, più velocità/accelerazione di 8 punti chiave (calcolate
frame-per-frame in tempo reale, a differenza dell'originale che le calcola
raggruppate per etichetta di annotazione — vedi il docstring del modulo
per il perché). Tutte le colonne finiscono nel CSV con prefisso `chuv_`.

**Uso**:

```bash
cd src
python live_demo.py --source 0 --fps 30 --device mps --with-chuv-features \
    --out live_session.csv
```

Combinabile con `--with-gaze`/`--with-hands`/`--with-reid`/
`--target-track-id` senza conflitti.

**Cosa NON fa, deliberatamente**:

- Non replica 5 delle 56 colonne del set di feature finale del repository
  CHUV (punta-piede/tallone/sfondo): derivano dallo schema BODY-25
  (OpenPose), che COCO-17 (YOLO-pose) non include.
- Non carica il modello già addestrato (`model_xgboost.joblib`): le sue
  classi sono codici di annotazione clinica specifici (formato WAKEE) che
  richiedono un'etichettatura umana secondo un protocollo che qui non
  esiste, e il file che tradurrebbe le predizioni numeriche del modello in
  etichette leggibili non è nemmeno salvato dal repository originale.
  Questo modulo si ferma volutamente al feature engineering — i numeri, non
  le predizioni. Un classificatore proprio (eventualmente multimodale,
  incorporando audio/sottotitoli oltre alla posa) è un lavoro separato, non
  ancora iniziato.
- Non è validato su dati reali: `demo/chuv_features_check.py` verifica solo
  che le formule producano i numeri geometricamente attesi su scheletri
  sintetici (angolo retto → 90°, braccio disteso → 180°, traslazione rigida
  → velocità nulla nelle feature normalizzate, ecc.), non che le feature
  siano utili o comparabili in valore assoluto a quelle del repository
  CHUV calcolate su un video BODY-25 reale.

## Feature implementate (`features.py`)

- **Angoli articolari** (gomiti, ginocchia, spalle) per frame — proxy di
  postura e controllo motorio.
- **Velocità/energia di movimento** per keypoint e a livello di persona.
- **Indice di simmetria sinistra/destra**, `|v_L - v_R| / (v_L + v_R)`.
- **Score di movimento ripetitivo**: rapporto tra la potenza spettrale del
  picco dominante in banda 0.5-8 Hz e la potenza totale del segnale di
  velocità (FFT), come proxy tecnico di periodicità/stereotipia. La banda
  di frequenza è un punto di partenza, da validare su dati reali.
- **Prossimità bambino-caregiver**: distanza euclidea tra i centri-bacino.
- **Sincronia motoria**: correlazione di Pearson finestrata tra l'energia
  di movimento delle due persone tracciate.

Nota tecnica sulla frequenza: poiché la velocità è calcolata come *norma*
del vettore spostamento, un movimento sinusoidale a frequenza *f* produce
un segnale di velocità con contenuto dominante a *2f* (effetto di
raddrizzamento). Il demo su dati sintetici lo rende visibile: un'oscillazione
generata a 3 Hz viene rilevata a 6 Hz nello score di ripetitività — comportamento
atteso, documentato qui per trasparenza metodologica.

## Considerazioni etiche e di privacy

Trattandosi di video di minori in contesto clinico, qualunque pipeline
reale dovrà integrare, oltre all'approvazione del comitato etico:

- anonimizzazione (blur del volto, vedi `anonymize.py`) applicata il prima
  possibile nella pipeline, idealmente prima di qualunque salvataggio su
  disco non cifrato;
- conformità alla legge svizzera sulla protezione dei dati (LPD) e, se
  applicabile, al GDPR;
- separazione tra dati grezzi (video) e dati derivati (keypoint/feature),
  con policy di retention distinte.

## Estensioni possibili

- Sostituire/affiancare YOLO-pose con MediaPipe Holistic per maggiore
  densità di landmark (mani, volto) su singola persona.
- Classificatore temporale (es. GRU/LSTM o feature-based con scikit-learn)
  su finestre di feature per la classificazione di pattern comportamentali,
  una volta disponibili dati annotati.
- Esportazione in formato compatibile con strumenti di analisi del gruppo
  di ricerca (es. notebook Jupyter con pandas per l'esplorazione statistica).
