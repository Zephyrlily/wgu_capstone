import main as phish
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import accuracy_score, roc_auc_score

DEFAULT_MODEL = os.path.join("models", "phish_detector.joblib")
DEFAULT_EVAL = os.path.join("data", "eval.csv")
DEFAULT_TRAIN = os.path.join("data", "train.csv")

def compute_all(bundle, urls, labels, threshold=0.55):
    probabilities = phish.predict_probability(bundle, urls)
    predictions = [int(p >= threshold) for p in probabilities]

    # Count per label
    tp = sum(1 for y, pred in zip(labels, predictions) if y == 1 and pred == 1)
    tn = sum(1 for y, pred in zip(labels, predictions) if y == 0 and pred == 0)
    fp = sum(1 for y, pred in zip(labels, predictions) if y == 0 and pred == 1)
    fn = sum(1 for y, pred in zip(labels, predictions) if y == 1 and pred == 0)

    # Get accuracy and auc
    accuracy = accuracy_score(labels, predictions)
    auc = roc_auc_score(labels, probabilities)
    evaluation_thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    accuracy_by_threshold = {}
    for thresh in evaluation_thresholds:
        threshold_predictions = [int(p >= thresh) for p in probabilities]
        accuracy_by_threshold[f"{thresh:.3f}"] = accuracy_score(labels, threshold_predictions)

    return {
        "counts": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
        "accuracy": float(accuracy),
        "auc": float(auc) if auc == auc else None,
        "probabilities": probabilities,
        "predictions": predictions,
        "threshold": float(threshold),
        "accuracy_by_threshold": accuracy_by_threshold,
    }

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Phish Detector")
        self.geometry("1000x1000")

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=10, pady=10)

        # Tabs
        self.train_tab = ttk.Frame(nb)
        self.pred_tab = ttk.Frame(nb)
        self.eval_tab = ttk.Frame(nb)
        nb.add(self.train_tab, text="Train")
        nb.add(self.pred_tab, text="Predict")
        nb.add(self.eval_tab, text="Evaluate")
        self._build_train_tab()
        self._build_eval_tab()
        self._build_pred_tab()

    def _build_train_tab(self):
        frame = self.train_tab

        ttk.Label(frame, text="Training CSV:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.train_csv_path = tk.StringVar(value=DEFAULT_TRAIN)
        ttk.Entry(frame, textvariable=self.train_csv_path, width=50).grid(row=0, column=1, sticky="we", padx=5, pady=5)
        ttk.Button(frame, text="Browse", command=self._select_train_csv).grid(row=0, column=2, padx=5, pady=5)

        ttk.Label(frame, text="Output path:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.train_model_path = tk.StringVar(value=DEFAULT_MODEL)
        ttk.Entry(frame, textvariable=self.train_model_path, width=50).grid(row=1, column=1, sticky="we", padx=5, pady=5)
        ttk.Button(frame, text="Browse", command=self._select_train_output).grid(row=1, column=2, padx=5, pady=5)

        ttk.Button(frame, text="Train", command=self._train).grid(row=2, column=2, padx=5, pady=5, sticky="w")

        self.train_log = ScrolledText(frame, height=10)
        self.train_log.grid(row=3, column=0, columnspan=3, sticky="nsew", padx=5, pady=5)

        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(3, weight=1)

    def _select_train_csv(self):
        path = filedialog.askopenfilename(title="Choose training CSV", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            self.train_csv_path.set(path)

    def _select_train_output(self):
        path = filedialog.asksaveasfilename(title="Save joblib to...", defaultextension=".joblib", filetypes=[("Joblib", "*.joblib"), ("All files", "*.*")])
        if path:
            self.train_model_path.set(path)

    def _train(self):
        csv_path = self.train_csv_path.get().strip() or DEFAULT_TRAIN
        model_path = self.train_model_path.get().strip() or DEFAULT_MODEL
        if not csv_path:
            messagebox.showerror("Error", "Please choose a proper training CSV.")
            return
        try:
            urls, labels = phish.read_csv_data(csv_path)
            bundle = phish.train_model(urls, labels)
            phish.save_bundle(bundle, model_path)
            self.train_log.insert("end", f"Model saved to: {model_path}\n")
            self.train_log.see("end")
            messagebox.showinfo("Training complete", f"Saved model to:\n{model_path}")
        except Exception as e:
            messagebox.showerror("Training error occurred", str(e))

    def _build_pred_tab(self):
        frame = self.pred_tab

        ttk.Label(frame, text="Model path:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.predict_model_path = tk.StringVar(value=DEFAULT_MODEL)
        ttk.Entry(frame, textvariable=self.predict_model_path, width=50).grid(row=0, column=1, sticky="we", padx=5, pady=5)
        ttk.Button(frame, text="Browse", command=self._select_pred_model).grid(row=0, column=2, padx=5, pady=5)

        ttk.Label(frame, text="Threshold:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.prediction_threshold = tk.DoubleVar(value=0.55)
        ttk.Spinbox(frame, from_=0.0, to=1.0, increment=0.01, textvariable=self.prediction_threshold, width=5).grid(row=1, column=1, sticky="w", padx=5, pady=5)

        ttk.Label(frame, text="One URL:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.predict_url = tk.StringVar()
        ttk.Entry(frame, textvariable=self.predict_url, width=50).grid(row=2, column=1, sticky="we", padx=5, pady=5)

        ttk.Label(frame, text="Multi URL CSV:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.predict_csv = tk.StringVar(value=DEFAULT_EVAL)
        ttk.Entry(frame, textvariable=self.predict_csv, width=50).grid(row=3, column=1, sticky="we", padx=5, pady=5)
        ttk.Button(frame, text="Browse", command=self._select_pred_csv).grid(row=3, column=2, padx=5, pady=5)

        ttk.Button(frame, text="Predict", command=self._predict).grid(row=4, column=2, padx=5, pady=5, sticky="e")

        self.tree = ttk.Treeview(frame, columns=("url", "score", "prediction"), show="headings", height=15)
        self.tree.heading("url", text="URL")
        self.tree.heading("score", text="Score")
        self.tree.heading("prediction", text="Prediction (benign = 0, phish = 1)")
        self.tree.column("url", width=650, anchor="w")
        self.tree.column("score", width=75, anchor="center")
        self.tree.column("prediction", width=125, anchor="center")
        self.tree.grid(row=5, column=0, columnspan=3, sticky="nsew", padx=5, pady=5)

        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(5, weight=1)

    def _select_pred_model(self):
        path = filedialog.askopenfilename(title="Choose model", filetypes=[("Joblib", "*.joblib"), ("All files", "*.*")])
        if path:
            self.predict_model_path.set(path)

    def _select_pred_csv(self):
        path = filedialog.askopenfilename(title="Choose predict CSV", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            self.predict_csv.set(path)

    def _predict(self):
        model_path = self.predict_model_path.get().strip() or DEFAULT_MODEL
        threshold = float(self.prediction_threshold.get())

        url = self.predict_url.get().strip()
        csv_path = self.predict_csv.get().strip()

        if not url and not csv_path:
            messagebox.showerror("Error", "Incorrect data entered. Please input data to predict.")
            return

        try:
            bundle = phish.load_bundle(model_path)
            if url:
                urls = [url]
            else:
                urls = phish.read_csv_urls(csv_path)

            probabilities = phish.predict_probability(bundle, urls)
            predictions = [int(p >= threshold) for p in probabilities]

            for row in self.tree.get_children():
                self.tree.delete(row)
            for u, p, pred in zip(urls, probabilities, predictions):
                self.tree.insert("", "end", values=(u, f"{p:.3f}", pred))

        except Exception as e:
            messagebox.showerror("Predicting error occurred", str(e))

    def _build_eval_tab(self):
        frame = self.eval_tab

        ttk.Label(frame, text="Evaluate CSV:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.evaluate_csv_path = tk.StringVar(value=DEFAULT_EVAL)
        ttk.Entry(frame, textvariable=self.evaluate_csv_path, width=50).grid(row=0, column=1, sticky="we", padx=5, pady=5)
        ttk.Button(frame, text="Browse", command=self._select_eval_csv).grid(row=0, column=2, padx=5, pady=5)

        ttk.Label(frame, text="Model path:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.evaluate_model_path = tk.StringVar(value=DEFAULT_MODEL)
        ttk.Entry(frame, textvariable=self.evaluate_model_path, width=50).grid(row=1, column=1, sticky="we", padx=5, pady=5)
        ttk.Button(frame, text="Browse", command=self._select_eval_model).grid(row=1, column=2, padx=5, pady=5)

        ttk.Label(frame, text="Threshold:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.evaluate_threshold = tk.DoubleVar(value=0.55)
        ttk.Spinbox(frame, from_=0.0, to=1.0, increment=0.01, textvariable=self.evaluate_threshold, width=5).grid(row=2, column=1, sticky="w", padx=5, pady=5)

        ttk.Button(frame, text="Evaluate", command=self._evaluate).grid(row=2, column=2, padx=5, pady=5, sticky="e")

        self.evaluate_summary = ttk.Label(frame, text="Accuracy: ????     AUC: ????")
        self.evaluate_summary.grid(row=3, column=0, columnspan=3, sticky="w", padx=5, pady=5)

        charts = ttk.Frame(frame)
        charts.grid(row=4, column=0, columnspan=3, sticky="nsew", padx=5, pady=5)
        charts.columnconfigure(0, weight=1)
        charts.columnconfigure(1, weight=1)
        charts.rowconfigure(0, weight=1)
        charts.rowconfigure(1, weight=1)

        self.pie_chart = Figure(figsize=(5, 3), dpi=100)
        self.pie_ax = self.pie_chart.add_subplot(111)
        self.pie_chart_canvas = FigureCanvasTkAgg(self.pie_chart, master=charts)
        self.pie_chart_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.bar_chart = Figure(figsize=(5, 3), dpi=100)
        self.bar_ax = self.bar_chart.add_subplot(111)
        self.bars_canvas = FigureCanvasTkAgg(self.bar_chart, master=charts)
        self.bars_canvas.get_tk_widget().grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        self.score_distribution = Figure(figsize=(10, 4), dpi=100)
        self.distribution_ax = self.score_distribution.add_subplot(111)
        self.distribution_canvas = FigureCanvasTkAgg(self.score_distribution, master=charts)
        self.distribution_canvas.get_tk_widget().grid(row=1, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)

        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(4, weight=1)

    def _select_eval_csv(self):
        path = filedialog.askopenfilename(title="Choose evaluation CSV", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            self.evaluate_csv_path.set(path)

    def _select_eval_model(self):
        path = filedialog.askopenfilename(title="Choose model", filetypes=[("Joblib", "*.joblib"), ("All files", "*.*")])
        if path:
            self.evaluate_model_path.set(path)

    def _evaluate(self):
        csv_path = self.evaluate_csv_path.get().strip() or DEFAULT_EVAL
        model_path = self.evaluate_model_path.get().strip() or DEFAULT_MODEL
        thresholds = float(self.evaluate_threshold.get())
        if not csv_path:
            messagebox.showerror("Error", "Please choose a proper evaluating CSV.")
            return
        try:
            bundle = phish.load_bundle(model_path)
            urls, labels = phish.read_csv_data(csv_path)
            results = compute_all(bundle, urls, labels, thresholds)

            auc = f"{results['auc']:.3f}"
            self.evaluate_summary.config(text=f"Accuracy: {results['accuracy']:.3f}    AUC: {auc}    Threshold: {results['threshold']}")

            # Pie Chart
            self.pie_ax.clear()
            sizes = [results["counts"]["TP"], results["counts"]["FP"], results["counts"]["TN"], results["counts"]["FN"]]
            labels_names = ["True Positive", "False Positive", "True Negative", "False Negative"]
            self.pie_ax.pie(sizes, labels=labels_names, autopct="%1.1f%%")
            self.pie_ax.axis("equal")
            self.pie_ax.set_title("Accuracy Summary")
            self.pie_chart_canvas.draw()

            # Bar Chart
            self.bar_ax.clear()
            thresholds = [float(k) for k in sorted(results["accuracy_by_threshold"].keys(), key=lambda x: float(x))]
            accuracies_by_thresholds = [results["accuracy_by_threshold"][f"{t:.3f}"] for t in thresholds]
            self.bar_ax.bar([str(t) for t in thresholds], accuracies_by_thresholds)
            self.bar_ax.set_ylim(0, 1)
            self.bar_ax.set_xlabel("Threshold")
            self.bar_ax.set_ylabel("Accuracy")
            self.bar_ax.set_title("Accuracy by Threshold")
            self.bars_canvas.draw()

            # Score Distribution
            self.distribution_ax.clear()
            scores = results["probabilities"]
            negative = [score for score, label in zip(scores, labels) if label == 0]
            positive = [score for score, label in zip(scores, labels) if label == 1]
            self.distribution_ax.hist([negative, positive], bins=20, stacked=False, label=["Negative", "Positive"])
            self.distribution_ax.axvline(results["threshold"], linestyle="--")
            self.distribution_ax.set_xlabel("Score")
            self.distribution_ax.set_ylabel("Count")
            self.distribution_ax.set_title("Score Distribution")
            self.distribution_ax.legend()
            self.distribution_canvas.draw()

        except Exception as e:
            messagebox.showerror("Evaluating error occurred", str(e))


if __name__ == "__main__":
    App().mainloop()
