
# Overview
Contiene il progetto che permette di analizzare i movimenti delle parti del corpo dei soggetti presenti alle registrazioni 360 delle sessioni di musicoterapia.

# Requisiti
1. Installare librerie da `requirements.txt`
2. Aggiungere in `data/modules` il modello [PointRend model](https://github.com/ayoolaolafenwa/PixelLib/releases/download/0.2.0/pointrend_resnet50.pkl)

# Utilizzo

## Divisione in clip

Individua i momenti di canto in un video e genera una clip per ogni canzone; inoltre viene generato un file .xml che tiene traccia della posizione delle clip nel video originale.

Utilizzo:

```bash
python main.py estrai_canzoni input_video output_dir
```
#### `--input_video`
Descrizione: è il perscorso del video da suddividere in clip.

#### `--output_dir`
Descrizione: è il perscorso della cartella nella quale verrà salvato l'ouput




## Normalizzazione dei video

Ruota la vista dei video 360 in modo da allontanare le persone dal bordo verticale del frame e riduce l'altezza delle immagini tagliando la parte superiore e quella inferiore del video. Viene fatta come operazione preliminare sulle clip prima di essere analizzate.

Utilizzo:

```bash
python main.py norm_video input_video output_video -p padding
```
#### `--input_video`
Descrizione: è il perscorso del video da suddividere in clip.

#### `--output_video`
Descrizione: è la posizione che avrà il nuovo video dopo l'elaborazione

#### `--padding`
Descrizione: è il padding applicato durante il calcolo dei due punti di taglio orizzontale del frame.



## Analisi dei video

Genera un file .json contentente i dati sui movimenti e sulle posizioni delle parti del corpo per le persone presentinei video. L'output può essere visualizzato usando le funzioni presenti nel notebook.

Utilizzo:

```bash
python main.py analizza input_video output -fef face_encodings_file -sF segmentation_freq -fdF face_detection_freq -ofF optical_flow_freq -mw max_width
```
#### `--input_video`
Descrizione: è il perscorso del video da suddividere in clip.

#### `--output`
Descrizione: è la posizione della cartella che conterrà l'output.

#### `--face_encodings_file`
Descrizione: è il percorso del file contenente le codifiche dei volti da confrontare durante la fase di riconosciento facciale. Se non specificato, al termine dell'elaborazione viene chiesto di inserire un'etichetta per ogni persona individuata e viene creato un file json con le codifiche calcolate.

#### `--segmentation_freq`
Descrizione: è la frequenza con la quale viene applicata la segmentazione, minore è la frequenza e minore sarà il tempo di elaborazione.

#### `--face_detection_freq`
Descrizione: è la frequenza con la quale viene applicata la ricerca e il riconoscimento dei volti, minore è la frequenza e minore sarà il tempo di elaborazione.

#### `--optical_flow_freq`
Descrizione: è la frequenza con la quale viene applicato il calcolo del flusso ottico, minore è la frequenza e minore sarà il tempo di elaborazione.

#### `--max_width`
Descrizione: è la dimensione orizzontale massima in pixel che può avere un frame per poter essere elaborato, il video viene ridimensionato se la sua dimensione orizzontale è maggiore. Minore è questa dimensione e minore sarà il tempo di elaborazione.



## Visualizzazione dati
1. Includere `notebook.ipynb` nel proprio account google Drive
2. Nella stessa posizione creare la cartella `data`. Questa cartella deve conterere una cartella per ogni seduta, all'interno di ogni cartella contenuta in `data` si devono trovare i file .json generati.
3. Aprire il notebook, specificare la sua posizione del notebook e il nome della cartella contentente i file .json da visualizzare
