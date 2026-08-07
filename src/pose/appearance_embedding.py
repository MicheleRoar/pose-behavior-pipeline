"""
appearance_embedding.py
=========================
Segnale di aspetto basato su un vero embedding di deep re-identification
(OSNet, via `torchreid`), invece delle euristiche colore/forma/posizione
gia' presenti in `reid.py`/`seg_reid.py`. Nasce dalla richiesta esplicita
di aggiungere OSNet e "l'idea" di StrongSORT alla pipeline, con priorita'
assoluta: gli id non devono cambiare, e le persone devono restare in
memoria per essere ri-associate facilmente al rientro.

Perche' "OSNet + l'idea di StrongSORT" e non "sostituire tutto con
StrongSORT" (scelta di scoping, vedi anche la memoria di progetto)
------------------------------------------------------------------------
StrongSORT (Du et al. 2022) e' DeepSORT + tre aggiunte principali: (1) un
embedding di aspetto vero (tipicamente OSNet) al posto delle feature IoU-
only di SORT, (2) un "feature bank" per traccia aggiornato con una media
mobile esponenziale (EMA) invece di tenere solo l'ultimo embedding visto,
(3) un filtro di Kalman "NSA" (rumore di processo scalato sulla confidenza
della detection) piu' compensazione del moto di camera (ECC) -- questi
ultimi pensati per una camera CHE SI MUOVE (es. inseguimento drone/veicolo),
non il caso qui (camera fissa, contesto clinico).

Riscrivere da zero l'intero tracker (Kalman NSA + ECC + matching cascade)
butterebbe via `identity_manager.py` gia' costruito e testato (matching
ungherese batch, policy "uncertain" invece di fusione silenziosa,
`session_mode`, causalita', il tetto `max_people` per sessioni chiuse) --
tutta logica su misura per il contesto clinico che un tracker generico non
conosce. Le due idee di StrongSORT che contano davvero per la richiesta
("non cambiare id" + "restare in memoria per il rientro") sono invece
esattamente (1) e (2): un embedding di aspetto piu' forte delle euristiche
attuali, e una memoria che si CONSOLIDA nel tempo invece di basarsi su un
solo frame. Questo modulo fornisce (1) (`OSNetEmbedder`); (2) e' applicata
qui sotto (`ema_update`) e usata da `reid.py`/`seg_reid.py` per aggiornare
`self.embedding` ad ogni frame in cui la persona e' visibile, non solo al
momento della perdita.

Dipendenza pesante e opzionale
--------------------------------
`torchreid` (e quindi `torch`) NON sono elencati in requirements.txt come
dipendenza normale -- stesso trattamento di SAM 3.1/SAM2, vedi li'.
L'import e' quindi ritardato dentro `OSNetEmbedder.__init__`: senza
torchreid installato, il resto della pipeline (incluso il resto del
Re-ID euristico) continua a funzionare normalmente, semplicemente senza
questo segnale aggiuntivo. Il primo utilizzo di un `model_name` noto (es.
'osnet_x1_0') senza `model_path` fa scaricare a torchreid i pesi
pre-addestrati dal suo model zoo -- richiede quindi una connessione
internet al primo avvio, non solo l'installazione del pacchetto.

Formato dei crop
-----------------
`torchreid.utils.FeatureExtractor` accetta array numpy (H, W, C) e li
converte internamente con `torchvision.transforms.ToPILImage()`, che per
un array 3 canali produce un'immagine in modalita' 'RGB' -- quindi il
crop va preparato in RGB, non nel BGR nativo di OpenCV (vedi `_crop_person`,
`[:, :, ::-1]`). Se viene passato anche il poligono maschera (non solo il
bbox), lo sfondo dentro il bbox ma fuori dalla sagoma viene azzerato prima
di passare il crop al modello: l'embedding si concentra sulla persona, non
sul contesto (sfondo, altre persone parzialmente dentro lo stesso bbox) --
un miglioramento rispetto a un crop-bbox grezzo, discusso nella
consultazione architetturale iniziale.
"""

from __future__ import annotations

import numpy as np

DEFAULT_MODEL_NAME = "osnet_x1_0"

# Sotto queste dimensioni (pixel) un crop e' troppo piccolo/troppo
# schiacciato per fidarsi dell'embedding risultante -- stesso principio di
# "nessun segnale inventato" di compute_color_signature/_mask_hue_histogram:
# meglio None che un embedding rumoroso spacciato per affidabile.
MIN_CROP_W = 24
MIN_CROP_H = 48

# Frazione minima di pixel-maschera dentro il bbox perche' valga la pena
# azzerare lo sfondo (sotto questa soglia il poligono e' probabilmente
# troppo degenere/rumoroso per un mascheramento affidabile -- si usa
# comunque il crop-bbox grezzo, non si scarta il frame).
_MIN_MASK_FILL = 0.15


def _resolve_feature_extractor():
    """Risolve la classe `FeatureExtractor`, gestendo DUE layout diversi del
    pacchetto 'torchreid' che si puo' finire per installare con
    `pip install torchreid`:

    - il progetto originale (github.com/KaiyangZhou/deep-person-reid,
      installabile con `pip install git+...` o `pip install -e .` da un
      clone) espone `torchreid/utils/` come un vero sottopacchetto --
      `from torchreid.utils import FeatureExtractor` funziona.
    - la distribuzione PyPI "torchreid-pip" di terzi (quella che installa
      di default il semplice `pip install torchreid`, verificato: e' un
      repackaging non ufficiale) nasconde tutto sotto `torchreid.reid.*` e
      si limita a rebindare `utils` come ATTRIBUTO del pacchetto top-level
      dentro il proprio `torchreid/__init__.py` (`from torchreid.reid import
      ..., utils`) -- non un vero sottomodulo. Con questo layout, `from
      torchreid.utils import FeatureExtractor` fallisce con
      `ModuleNotFoundError: No module named 'torchreid.utils'` anche se il
      pacchetto e' installato correttamente (causa reale di un bug
      segnalato dall'utente: "torchreid" importabile da solo, ma questo
      import specifico no).

    Importare `torchreid` e poi accedere per ATTRIBUTO
    (`torchreid.utils.FeatureExtractor`) funziona in entrambi i casi, quindi
    e' quello che usiamo qui invece di un `from torchreid.utils import ...`
    diretto."""
    import torchreid
    return torchreid.utils.FeatureExtractor


class OSNetEmbedder:
    """Wrapper minimale su `torchreid.utils.FeatureExtractor` per un singolo
    modello OSNet. Uso in reid.py/seg_reid.py (un solo punto di wiring):

        embedder = OSNetEmbedder(device="cpu")  # o "cuda"/"mps" se disponibile
        ...
        vec = embedder.embed(frame_bgr, bbox_xyxy, poly=poly)  # None se crop scadente
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, model_path: str | None = None,
                 device: str = "cpu"):
        try:
            FeatureExtractor = _resolve_feature_extractor()
        except ImportError as exc:
            raise ImportError(
                "L'embedding di aspetto OSNet richiede 'torch' e 'torchreid', "
                "non installati per default (dipendenza pesante, opzionale -- "
                "vedi requirements.txt). Installare con: "
                "pip install torch torchreid"
            ) from exc
        except AttributeError as exc:
            raise ImportError(
                "'torchreid' risulta installato ma non espone "
                "'torchreid.utils.FeatureExtractor' -- probabile che "
                "'pip install torchreid' abbia installato la distribuzione "
                "PyPI 'torchreid-pip' di terzi (repackaging con un layout "
                "diverso, non il progetto originale deep-person-reid) o "
                "un'installazione incompleta. Verificare con: "
                "python -c \"import torchreid; print(torchreid.utils.FeatureExtractor)\""
            ) from exc

        kwargs = dict(model_name=model_name, device=device, verbose=False)
        if model_path:
            kwargs["model_path"] = model_path
        # pretrained=True quando model_path e' assente: torchreid scarica i
        # pesi dal proprio model zoo al primo utilizzo di un model_name noto
        # (richiede internet la prima volta, poi restano in cache locale).
        self._extractor = FeatureExtractor(**kwargs)
        self.model_name = model_name
        self.device = device

    def embed(self, frame_bgr: np.ndarray, bbox_xyxy: np.ndarray,
               poly: np.ndarray | None = None) -> np.ndarray | None:
        """Embedding L2-normalizzato (np.ndarray 1D) della persona in
        `bbox_xyxy` (e opzionalmente mascherata da `poly`) dentro
        `frame_bgr`, oppure `None` se il crop e' troppo piccolo/degenere per
        fidarsi."""
        crop_rgb = _crop_person(frame_bgr, bbox_xyxy, poly)
        if crop_rgb is None:
            return None
        features = self._extractor([crop_rgb])  # torch tensor (1, D), no_grad interno
        vec = features[0].detach().cpu().numpy().astype(np.float64)
        norm = np.linalg.norm(vec)
        if norm < 1e-9 or not np.isfinite(norm):
            return None
        return vec / norm


def _crop_person(frame_bgr: np.ndarray, bbox_xyxy: np.ndarray,
                  poly: np.ndarray | None) -> np.ndarray | None:
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox_xyxy]
    x1, x2 = int(np.clip(x1, 0, w)), int(np.clip(x2, 0, w))
    y1, y2 = int(np.clip(y1, 0, h)), int(np.clip(y2, 0, h))
    if x2 - x1 < MIN_CROP_W or y2 - y1 < MIN_CROP_H:
        return None

    crop = frame_bgr[y1:y2, x1:x2].copy()
    if poly is not None and poly.shape[0] >= 3:
        import cv2
        shifted = poly.copy().astype(np.float64)
        shifted[:, 0] -= x1
        shifted[:, 1] -= y1
        mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        cv2.fillPoly(mask, [np.round(shifted).astype(np.int32)], 255)
        fill = cv2.countNonZero(mask) / float(mask.size)
        if fill >= _MIN_MASK_FILL:
            crop[mask == 0] = 0  # azzera lo sfondo, l'embedding si concentra sulla persona

    return crop[:, :, ::-1]  # BGR (OpenCV) -> RGB (atteso da torchreid)


def embedding_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    """Similarita' 0..1 tra due embedding L2-normalizzati, via cosine
    similarity riscalata da [-1, 1] a [0, 1] (stessa convenzione 0..1 degli
    altri segnali del modulo, non la convenzione [-1, 1] nativa del coseno).
    `None` se manca uno dei due embedding."""
    if a is None or b is None:
        return None
    cos = float(np.dot(a, b))
    return float(np.clip((cos + 1.0) / 2.0, 0.0, 1.0))


def torchreid_available() -> bool:
    """True se `torchreid.utils.FeatureExtractor` e' effettivamente
    raggiungibile in questo ambiente (stessa risoluzione di
    `_resolve_feature_extractor()` usata da `OSNetEmbedder`, non solo un
    `import torchreid` nudo -- un `import torchreid` puo' riuscire mentre
    `torchreid.utils.FeatureExtractor` no, vedi il docstring di
    `_resolve_feature_extractor` per il perche' concreto, verificato su un
    caso reale). Non istanzia nessun modello (non scarica pesi, non tocca
    la GPU) -- usato solo per gating lato GUI (checkbox disabilitata con un
    motivo, stesso schema di `cudaAvailable`/le opzioni SAM 3.1/SAM2 in
    app.js), MAI per decidere silenziosamente di saltare l'embedding: se
    l'utente lo richiede esplicitamente ma la dipendenza manca,
    `OSNetEmbedder.__init__` solleva comunque il suo `ImportError` con le
    istruzioni di installazione, non fallisce in silenzio."""
    try:
        _resolve_feature_extractor()
    except (ImportError, AttributeError):
        return False
    except Exception as exc:
        # torchreid e' un pacchetto fermo dal 2021: capita che l'import
        # fallisca con qualcos'altro (non ImportError) su un ambiente con
        # numpy/torch piu' recenti -- es. `np.float`/`np.int` rimossi in
        # numpy>=1.24, referenziati da alcune versioni di torchreid/yacs, che
        # sollevano AttributeError durante l'import, non ImportError. Se non
        # lo intercettassimo qui, l'eccezione risalirebbe fino a
        # `Api.detect_device()` (nessun try/except li', vedi webui/api.py) e
        # farebbe fallire ANCHE il rilevamento cuda/mps/cpu insieme ad esso
        # (app.js gestisce l'intera chiamata con un solo try/catch) -- un
        # sintomo confuso ("SAM 3.1/SAM2 e l'embedding sono entrambi
        # disabilitati, anche se torch e' installato") per una causa non
        # ovvia. Meglio segnalare qui SOLO l'embedding come non disponibile,
        # stampando il motivo reale sul terminale (non visibile in UI, ma
        # utile per diagnosticare) invece di un errore silenzioso o di un
        # crash che disabilita anche il rilevamento device.
        print(f"[appearance_embedding] torchreid installato ma l'import fallisce "
              f"({type(exc).__name__}: {exc}) -- embedding OSNet non disponibile. "
              f"Probabile incompatibilita' di versione (torchreid e' un pacchetto "
              f"non piu' mantenuto attivamente, spesso in conflitto con numpy/torch "
              f"recenti).")
        return False
    return True


def ema_update(prev: np.ndarray | None, new: np.ndarray | None,
                alpha: float = 0.9) -> np.ndarray | None:
    """Media mobile esponenziale su un embedding "in memoria" -- l'idea di
    StrongSORT (2) citata nel docstring del modulo: invece di ricalcolare
    l'embedding da un solo frame (rumoroso: motion blur, posa, occlusione
    parziale), lo si affina nel tempo, cosi' la firma memorizzata per una
    persona diventa via via piu' stabile piu' a lungo resta visibile --
    esattamente il comportamento richiesto ("restare in memoria per
    associarli facilmente al rientro"). `alpha` alto (default 0.9, come
    tipico in letteratura StrongSORT/DeepSORT) da' molto peso alla storia,
    poco al frame corrente: un singolo frame anomalo non fa "saltare"
    l'embedding memorizzato. Ri-normalizzato a norma 1 dopo la media (la
    media di due vettori unitari non e' in generale unitaria)."""
    if new is None:
        return prev
    if prev is None:
        return new
    updated = alpha * prev + (1.0 - alpha) * new
    norm = np.linalg.norm(updated)
    if norm < 1e-9:
        return prev
    return updated / norm
