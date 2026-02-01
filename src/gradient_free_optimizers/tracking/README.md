# SearchTracker Module - Development Status

## Implemented (Core)

### Data Model
- [x] `EvaluationRecord` - einzelne Evaluation mit Parameters, Score, Timing
- [x] `ExperimentMetadata` - Experiment-Name, Search Space, Tags

### Storage
- [x] `StorageBackend` Protocol - Interface fuer alle Backends
- [x] `SQLiteBackend` - Lokale SQLite-Datenbank (Default)

### SearchTracker
- [x] `@tracker.track` Decorator - erfasst alle Evaluationen
- [x] Automatische Erkennung von GFO-Style (`objective(params_dict)`) und Python-Style (`objective(x=1, y=2)`)
- [x] Multi-Run Support (`start_run()`, `end_run()`)
- [x] DataFrame Export (`to_dataframe()`)
- [x] Persistenz (automatisch in SQLite)
- [x] `load()` - Experiment aus Datei laden
- [x] `summary()` - Text-Zusammenfassung

### Plotting (via `tracker.plot.*`)
- [x] `convergence()` - Score ueber Iterationen
- [x] `convergence_by_run()` - Vergleich mehrerer Runs
- [x] `search_space()` - 2D Scatter der evaluierten Punkte
- [x] `search_space_3d()` - 3D Scatter
- [x] `parameter_importance()` - Balkendiagramm (Correlation-basiert)
- [x] `score_distribution()` - Histogramm der Scores
- [x] `evaluation_time()` - Evaluationszeit ueber Iterationen
- [x] `parallel_coordinates()` - Parallele Koordinaten fuer alle Parameter


## TODO - Storage Backends

### CSV Backend
- [ ] `CSVBackend` - Export/Import als CSV
- [ ] Separate Metadata-Datei (JSON) fuer Search Space etc.
- [ ] Streaming-Support (append ohne komplettes Neuschreiben)

### JSON Lines Backend
- [ ] `JSONLBackend` - Ein JSON-Objekt pro Zeile
- [ ] Human-readable, gut fuer Debugging

### Parquet Backend (Optional Dependency: pyarrow)
- [ ] `ParquetBackend` - Effizient fuer grosse Datenmengen
- [ ] Columnar Storage, gut fuer Analytics


## TODO - Erweiterte Plots

### Parameter Analysis
- [ ] `partial_dependence()` - Effekt einzelner Parameter auf Score
- [ ] `interaction_heatmap()` - Interaktionen zwischen Parametern
- [ ] fANOVA-basierte Importance (Optional Dependency: fanova)

### Advanced Visualizations
- [ ] `pairplot()` - Pairwise Scatter Matrix (Optional Dependency: seaborn)
- [ ] `trajectory_animation()` - Animierter Suchpfad
- [ ] `improvement_rate()` - Wann werden Verbesserungen seltener?


## TODO - Integrations

### MLflow Integration
- [ ] `tracker.enable_mlflow()` - Real-time Sync zu MLflow
- [ ] `tracker.export_to_mlflow()` - Nachtraeglicher Export
- [ ] Mapping: Records -> MLflow Metrics/Params
- [ ] Artifact Upload (DataFrame als CSV)

### Weights & Biases Integration
- [ ] `tracker.enable_wandb()` - Real-time Sync
- [ ] `tracker.export_to_wandb()` - Nachtraeglicher Export
- [ ] Nutzung von wandb.log() fuer Metrics

### Optuna Export
- [ ] `tracker.to_optuna_study()` - Konvertierung zu Optuna Study
- [ ] Ermoeglicht Nutzung von Optuna's Visualisierungen


## TODO - Features

### Reporting
- [ ] `tracker.report()` - HTML Report generieren
- [ ] Automatische Zusammenfassung aller wichtigen Plots
- [ ] Export als standalone HTML-Datei

### Query API
- [ ] `tracker.query(score_gt=0.5, run_id="...")` - Filter Records
- [ ] SQL-aehnliche Syntax fuer komplexe Queries

### Checkpointing
- [ ] Automatisches Speichern waehrend langer Runs
- [ ] Crash Recovery - Fortsetzen nach Absturz

### Concurrent Access
- [ ] Thread-safe Writes
- [ ] Multiple Processes schreiben in gleiche DB


## Design Decisions

### Warum SQLite als Default?
- Keine externe Dependency (Teil der Python stdlib)
- Unterstuetzt concurrent reads
- Gute Performance fuer typische Experiment-Groessen
- Einzelne Datei - einfach zu teilen/backupen

### Warum Decorator statt Callback?
- Minimalinvasiv - nur eine Zeile aendern
- Funktioniert mit JEDEM Optimizer (auch externe Libraries)
- Kein Eingriff in Optimizer-Code noetig
- Natuerlicher "Choke Point" - alle Evaluationen muessen durch

### Warum eigener Tracker statt MLflow/wandb?
- GFO-spezifische Visualisierungen (Parameter Importance fuer Optimization)
- Keine Cloud-Dependency
- Leichtgewichtig
- Optionale Integration zu den grossen Playern


## Usage Example

```python
from gradient_free_optimizers import HillClimbingOptimizer, SearchTracker

tracker = SearchTracker("experiment.db")

@tracker.track
def objective(params):
    return -(params["x"]**2 + params["y"]**2)

opt = HillClimbingOptimizer({"x": (-5, 5), "y": (-5, 5)})
opt.search(objective, n_iter=100)

# Analyse
print(tracker.summary())
tracker.plot.convergence()
tracker.plot.parameter_importance()

# Spaeter laden
tracker = SearchTracker.load("experiment.db")
```
