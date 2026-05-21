import sys
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import json

# Asegurar que el paquete `src` sea importable cuando se ejecuta
# el script directamente (python gui/app.py). Añadimos el root del
# proyecto al inicio de sys.path.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

from src import data_loader, feature_engineering, preprocessing, model_supervised, kmeans_modeling


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Predicción Copa Mundial — GUI")
        self.geometry("900x600")
        self.configure(bg="#f0f4f8")
        self.create_styles()
        self.create_widgets()

        self.df = None
        self.features_df = None
        self.profile_df = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.preprocessor = None

    def create_widgets(self):
        frm = ttk.Frame(self, style="Content.TFrame")
        frm.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        # Header
        header = ttk.Frame(self, style="Header.TFrame")
        header.pack(fill=tk.X, side=tk.TOP, padx=12, pady=(0,8))
        header_lbl = ttk.Label(header, text="Predicción Copa Mundial — ML", style="Header.TLabel")
        header_lbl.pack(side=tk.LEFT, padx=8, pady=8)
        subtitle = ttk.Label(header, text="Interfaz (Fases: Preproc · RF · KMeans)", style="SubHeader.TLabel")
        subtitle.pack(side=tk.LEFT, padx=12)

        # Left controls (lateral)
        left = tk.Frame(frm, bg="#dff3f1", bd=0)
        left.grid(row=0, column=0, sticky="nsw", padx=(0,12), pady=4)
        left.config(width=300)

        left_title = ttk.Label(left, text="Controles", style="PanelTitle.TLabel")
        left_title.pack(anchor="w", padx=10, pady=(8,4))

        ttk.Label(left, text="Dataset:", style="FieldLabel.TLabel").pack(anchor="w", padx=10)
        self.dataset_path = tk.StringVar()
        e = ttk.Entry(left, textvariable=self.dataset_path, width=36)
        e.pack(padx=10)
        ttk.Button(left, text="Cargar desde data/ (default)", style="Primary.TButton", command=self.load_default).pack(fill=tk.X, padx=10, pady=6)
        ttk.Button(left, text="Seleccionar CSV...", style="Secondary.TButton", command=self.select_file).pack(fill=tk.X, padx=10)

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10, padx=10)
        ttk.Button(left, text="Ejecutar Preprocesamiento", style="Primary.TButton", command=self.run_preprocessing).pack(fill=tk.X, padx=10, pady=4)
        ttk.Button(left, text="Entrenar RandomForest (baseline)", style="Accent.TButton", command=self.run_train_rf).pack(fill=tk.X, padx=10)
        ttk.Button(left, text="Entrenar K-Means", style="Accent.TButton", command=self.run_kmeans).pack(fill=tk.X, padx=10, pady=6)

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10, padx=10)
        ttk.Button(left, text="Exportar artefactos", style="Secondary.TButton", command=self.export_artifacts).pack(fill=tk.X, padx=10, pady=(0,12))

        # Right: log / plots
        right = ttk.Frame(frm, style="Content.TFrame")
        right.grid(row=0, column=1, sticky="nsew")
        frm.columnconfigure(1, weight=1)
        frm.rowconfigure(0, weight=1)

        # Notebook: pestañas para Log y Plots
        nb = ttk.Notebook(right)
        nb.pack(fill=tk.BOTH, expand=True)

        log_frame = ttk.Frame(nb)
        plot_frame = ttk.Frame(nb)
        nb.add(log_frame, text="Registro / Log")
        nb.add(plot_frame, text="Visualizaciones")

        self.log = tk.Text(log_frame, wrap="word", bg="#ffffff", fg="#1f2937", bd=1)
        self.log.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # Área de gráficos (placeholder)
        self.fig, self.ax = plt.subplots(figsize=(4,3))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Progress bar
        self.progress = ttk.Progressbar(self, mode="indeterminate", style="Accent.Horizontal.TProgressbar")
        self.progress.pack(fill=tk.X, side=tk.BOTTOM, padx=12, pady=(0,12))

    def create_styles(self):
        style = ttk.Style()
        try:
            style.theme_use('clam')
        except Exception:
            pass

        # Palette
        primary = '#2b7a78'   # teal
        accent = '#f6c85f'    # warm yellow
        secondary = '#6b7280' # gray
        bg = '#f0f4f8'

        style.configure('Header.TFrame', background=primary)
        style.configure('Header.TLabel', background=primary, foreground='#ffffff', font=('Segoe UI', 14, 'bold'))
        style.configure('SubHeader.TLabel', background=primary, foreground='#e6f6f5', font=('Segoe UI', 9))

        style.configure('Content.TFrame', background=bg)
        style.configure('PanelTitle.TLabel', background='#dff3f1', font=('Segoe UI', 11, 'bold'))
        style.configure('FieldLabel.TLabel', background='#dff3f1', font=('Segoe UI', 9))

        style.configure('Primary.TButton', background=primary, foreground='#ffffff', font=('Segoe UI', 9, 'bold'))
        style.map('Primary.TButton', background=[('active', '#246a66'), ('pressed', '#1e5b58')])

        style.configure('Accent.TButton', background=accent, foreground='#000000', font=('Segoe UI', 9, 'bold'))
        style.map('Accent.TButton', background=[('active', '#e0b74a'), ('pressed', '#c99f3d')])

        style.configure('Secondary.TButton', background='#ffffff', foreground=secondary, font=('Segoe UI', 9))
        style.map('Secondary.TButton', background=[('active', '#f3f4f6')])

        style.configure('Field.TEntry', padding=4)

        style.configure('Accent.Horizontal.TProgressbar', troughcolor='#e6f6e9', background=primary)

    def append_log(self, text: str):
        self.log.insert(tk.END, text + "\n")
        self.log.see(tk.END)

    def load_default(self):
        try:
            self.append_log("Cargando dataset por defecto desde data/...")
            self.df = data_loader.load_matches()
            self.dataset_path.set("data/matches_1930_2022.csv")
            self.append_log(f"Dataset cargado: {self.df.shape[0]} filas × {self.df.shape[1]} columnas")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def select_file(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files","*.csv" )])
        if path:
            self.dataset_path.set(path)
            try:
                import pandas as pd
                self.df = pd.read_csv(path)
                self.append_log(f"Dataset cargado desde: {path}")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def run_in_thread(self, fn):
        def wrapped():
            try:
                self.progress.start(10)
                fn()
            finally:
                self.progress.stop()

        t = threading.Thread(target=wrapped, daemon=True)
        t.start()

    def run_preprocessing(self):
        if self.df is None:
            messagebox.showinfo("Info", "Cargue primero el dataset.")
            return
        self.run_in_thread(self._preprocessing)

    def _preprocessing(self):
        try:
            self.append_log("Iniciando limpieza y ensamblaje de features...")
            df_clean = preprocessing.clean_dataset(self.df)
            feats = feature_engineering.assemble_features(df_clean)
            self.features_df = feats
            self.append_log(f"Features generadas: {feats.shape[0]} filas × {feats.shape[1]} columnas")

            # Split temporal y construcción de preprocessor
            df_train, df_test = preprocessing.temporal_train_test_split(feats, test_tournaments=2)
            pre = preprocessing.build_rf_preprocessor(df_train)
            X_train, y_train, le = preprocessing.prepare_Xy_rf(df_train)
            X_test, y_test, _ = preprocessing.prepare_Xy_rf(df_test)

            # Guardar en memoria
            self.preprocessor = pre
            self.X_train = X_train
            self.y_train = y_train
            self.X_test = X_test
            self.y_test = y_test

            preprocessing.save_preprocessing_metadata(le, df_train, df_test)
            preprocessing.save_label_encoder(le)

            # K-Means profile
            profile = feature_engineering.build_kmeans_profile(feats)
            self.profile_df = profile
            import joblib
            joblib.dump((X_train, y_train, X_test, y_test), "models/Xy_phase2.pkl")
            joblib.dump(pre, "models/rf_preprocessor.pkl")
            profile.to_csv("outputs/kmeans_profile_phase2.csv", index=False)

            self.append_log("Preprocesamiento completado y artefactos guardados.")
        except Exception as e:
            self.append_log(f"Error en preprocesamiento: {e}")

    def run_train_rf(self):
        if self.preprocessor is None or self.X_train is None:
            messagebox.showinfo("Info", "Ejecute primero el preprocesamiento.")
            return
        self.run_in_thread(self._train_rf)

    def _train_rf(self):
        try:
            self.append_log("Entrenando RandomForest baseline...")
            res = model_supervised.train_baseline(self.X_train, self.y_train, self.X_test, self.y_test, self.preprocessor)
            import joblib
            joblib.dump(res["pipeline"], "models/random_forest_pipeline.pkl")
            # Guardar métricas y artefactos mínimos
            fi = model_supervised.get_feature_importance(res["pipeline"]) if res.get("pipeline") is not None else None
            model_supervised.save_phase3_artifacts(res["pipeline"], res, res, {"summary": {}}, fi, paths=None)
            self.append_log("Entrenamiento RF completado. Artefactos guardados en models/.")
        except Exception as e:
            self.append_log(f"Error en entrenamiento RF: {e}")

    def run_kmeans(self):
        if self.profile_df is None:
            messagebox.showinfo("Info", "Ejecute primero el preprocesamiento.")
            return
        self.run_in_thread(self._run_kmeans)

    def _run_kmeans(self):
        try:
            self.append_log("Preparando datos para K-Means...")
            X_k, teams = preprocessing.prepare_X_kmeans(self.profile_df)
            scaler, X_sc = kmeans_modeling.fit_scaler(X_k)
            df_metrics = kmeans_modeling.evaluate_k_range(X_sc)
            k_opt = kmeans_modeling.select_optimal_k(df_metrics)
            km = kmeans_modeling.train_kmeans(X_sc, k_opt)
            df_clustered = kmeans_modeling.assign_clusters(self.profile_df, teams, km.labels_, km, X_sc, list(X_k.columns))
            centroids_df, labels_map = kmeans_modeling.describe_clusters(df_clustered, list(X_k.columns), km, scaler)
            kmeans_modeling.save_phase4_artifacts(km, kmeans_modeling.build_final_kmeans_pipeline(scaler, km), df_clustered, centroids_df, labels_map, df_metrics, k_opt, float(df_metrics[df_metrics['k']==k_opt]['silhouette']))
            self.append_log("K-Means completado y artefactos guardados en models/ y outputs/phase4/.")
        except Exception as e:
            self.append_log(f"Error en K-Means: {e}")

    def export_artifacts(self):
        self.append_log("Los artefactos principales se guardan automáticamente en la carpeta models/ y outputs/.\nRevise esos directorios para los archivos generados.")


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
