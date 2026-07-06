# =============================================================
#  SU KALİTESİ SINIFLANDIRMA PROJESİ
#  Makine Öğrenmesi - Logistic Regression, Random Forest, XGBoost
# =============================================================

import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, font as tkfont
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                              ConfusionMatrixDisplay, roc_auc_score, roc_curve)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# RENK PALETİ
# ─────────────────────────────────────────────
BG          = '#0F1117'   # ana arka plan
SIDEBAR_BG  = '#1A1D27'   # kenar çubuğu
CARD_BG     = '#1E2130'   # kart arka planı
ACCENT      = '#4F8EF7'   # mavi vurgu
ACCENT2     = '#A78BFA'   # mor vurgu
SUCCESS     = '#34D399'   # yeşil
DANGER      = '#F87171'   # kırmızı
TEXT_PRI    = '#F1F5F9'   # birincil metin
TEXT_SEC    = '#94A3B8'   # ikincil metin
BORDER      = '#2D3147'   # kenarlık
SEL_BG      = '#252A3D'   # seçili arka plan

MODEL_COLORS = [ACCENT, SUCCESS, ACCENT2]

# ─────────────────────────────────────────────
# 1. VERİ & MODEL
# ─────────────────────────────────────────────
print("Veri yükleniyor...")
df = pd.read_csv('waterQuality1.csv')
df = df[df['is_safe'] != '#NUM!'].copy()
df['is_safe'] = df['is_safe'].astype(int)
for col in df.columns:
    if col != 'is_safe':
        df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna()

X = df.drop('is_safe', axis=1)
y = df['is_safe']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scale_pw = len(y[y==0]) / len(y[y==1])
models_def = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    'Random Forest':       RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'),
    'XGBoost':             XGBClassifier(n_estimators=200, random_state=42, eval_metric='logloss', scale_pos_weight=scale_pw)
}

results = {}
for name, model in models_def.items():
    print(f"  {name} eğitiliyor...")
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    results[name] = {
        'model':    model,
        'y_pred':   y_pred,
        'y_proba':  y_proba,
        'accuracy': accuracy_score(y_test, y_pred),
        'auc':      roc_auc_score(y_test, y_proba)
    }
    print(f"    Accuracy: {results[name]['accuracy']:.4f}  AUC: {results[name]['auc']:.4f}")

print("Modeller hazır. Arayüz açılıyor...\n")

# ─────────────────────────────────────────────
# 2. MATPLOTLIB DARK THEME
# ─────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':  BG,
    'axes.facecolor':    CARD_BG,
    'axes.edgecolor':    BORDER,
    'axes.labelcolor':   TEXT_SEC,
    'axes.titlecolor':   TEXT_PRI,
    'text.color':        TEXT_PRI,
    'xtick.color':       TEXT_SEC,
    'ytick.color':       TEXT_SEC,
    'grid.color':        BORDER,
    'grid.linewidth':    0.6,
    'figure.dpi':        100,
    'font.size':         10,
    'axes.titlesize':    13,
    'axes.titleweight':  'bold',
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

# ─────────────────────────────────────────────
# 3. GRAFİK FONKSİYONLARI
# ─────────────────────────────────────────────
NAMES = list(results.keys())

def make_fig(w=11, h=6.5):
    fig = plt.Figure(figsize=(w, h), facecolor=BG)
    return fig


def fig_overview():
    fig = make_fig()
    gs  = gridspec.GridSpec(1, 2, figure=fig, left=0.08, right=0.97,
                             top=0.88, bottom=0.14, wspace=0.35)
    accs = [r['accuracy'] for r in results.values()]
    aucs = [r['auc']      for r in results.values()]

    for i, (vals, title, ylabel) in enumerate([
            (accs, 'Accuracy Karşılaştırması', 'Accuracy'),
            (aucs, 'ROC-AUC Karşılaştırması',  'AUC Skoru')]):
        ax = fig.add_subplot(gs[i])
        bars = ax.bar(NAMES, vals, color=MODEL_COLORS,
                      edgecolor=BG, linewidth=2, width=0.5)
        ax.set_ylim(0.4, 1.08)
        ax.set_title(title)
        ax.set_ylabel(ylabel, color=TEXT_SEC)
        ax.grid(axis='y', linestyle='--')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.012,
                    f'{v:.4f}', ha='center', va='bottom',
                    fontweight='bold', fontsize=11, color=TEXT_PRI)
        ax.tick_params(axis='x', labelsize=9)
    return fig


def fig_confusion():
    fig = make_fig(w=13, h=5.5)
    gs  = gridspec.GridSpec(1, 3, figure=fig, left=0.04, right=0.98,
                             top=0.88, bottom=0.12, wspace=0.35)
    # Sabit renkler: TN=koyu lacivert, FP/FN=orta mavi, TP=parlak mavi
    cell_colors = ['#1E3A5F', '#2563EB', '#2563EB', '#60A5FA']   # TN FP FN TP
    text_colors = ['#93C5FD',  '#FFFFFF',  '#FFFFFF', '#0F172A']
    for i, (name, r) in enumerate(results.items()):
        ax = fig.add_subplot(gs[i])
        cm = confusion_matrix(y_test, r['y_pred'])
        # Her hücreyi ayrı ayrı çiz
        for row in range(2):
            for col in range(2):
                idx = row * 2 + col
                ax.add_patch(plt.Rectangle(
                    (col - 0.5, row - 0.5), 1, 1,
                    facecolor=cell_colors[idx], edgecolor='#0F1117', linewidth=3))
                ax.text(col, row, f'{cm[row, col]:,}',
                        ha='center', va='center',
                        fontsize=20, fontweight='bold',
                        color=text_colors[idx])
        ax.set_xlim(-0.5, 1.5); ax.set_ylim(-0.5, 1.5)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(['Güvensiz', 'Güvenli'], color=TEXT_SEC, fontsize=10)
        ax.set_yticklabels(['Güvensiz', 'Güvenli'], color=TEXT_SEC, fontsize=10)
        ax.set_xlabel('Tahmin', color=TEXT_SEC)
        ax.set_ylabel('Gerçek', color=TEXT_SEC)
        ax.set_title(f'{name}\nAcc {r["accuracy"]:.4f}  |  AUC {r["auc"]:.4f}',
                     color=TEXT_PRI)
        # Etiket ekle (TN FP FN TP)
        for row2, col2, lbl in [(0,0,'TN'),(0,1,'FP'),(1,0,'FN'),(1,1,'TP')]:
            ax.text(col2, row2 - 0.28, lbl,
                    ha='center', va='center',
                    fontsize=9, color="#000000")
    return fig


def fig_roc():
    fig = make_fig()
    ax  = fig.add_axes([0.10, 0.12, 0.84, 0.75])
    ax.set_facecolor(CARD_BG)
    for (name, r), color in zip(results.items(), MODEL_COLORS):
        fpr, tpr, _ = roc_curve(y_test, r['y_proba'])
        ax.plot(fpr, tpr, color=color, lw=2.5,
                label=f'{name}  (AUC = {r["auc"]:.4f})')
    ax.plot([0,1],[0,1], color=BORDER, lw=1.5, linestyle='--', label='Rastgele (0.50)')
    ax.fill_between([0,1],[0,1], alpha=0.04, color=BORDER)
    ax.set_xlabel('Yanlış Pozitif Oranı (FPR)')
    ax.set_ylabel('Doğru Pozitif Oranı (TPR)')
    ax.set_title('ROC Eğrisi — Model Karşılaştırması')
    leg = ax.legend(fontsize=11, loc='lower right',
                    facecolor=SIDEBAR_BG, edgecolor=BORDER, labelcolor=TEXT_PRI)
    ax.grid(linestyle='--')
    return fig


def fig_feature():
    fig = make_fig(h=7)
    ax  = fig.add_axes([0.18, 0.06, 0.76, 0.84])
    ax.set_facecolor(CARD_BG)
    rf  = results['Random Forest']['model']
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
    colors_fi = [ACCENT2 if v >= imp.quantile(0.75) else ACCENT for v in imp.values]
    bars = ax.barh(imp.index, imp.values, color=colors_fi,
                   edgecolor=BG, linewidth=1.2, height=0.6)
    for bar, v in zip(bars, imp.values):
        ax.text(v + 0.001, bar.get_y() + bar.get_height()/2,
                f'{v:.4f}', va='center', fontsize=9, color=TEXT_SEC)
    ax.set_xlabel('Önem Skoru')
    ax.set_title('Feature Importance — Random Forest  (mor = en önemli)')
    ax.grid(axis='x', linestyle='--')
    return fig


def fig_heatmap():
    fig = make_fig(w=12, h=7.5)
    ax  = fig.add_axes([0.10, 0.10, 0.85, 0.80])
    ax.set_facecolor(CARD_BG)
    corr = X.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Koyu temada net görünen özel renk paleti:
    # Negatif korelasyon → turuncu/kırmızı, Sıfır → koyu gri, Pozitif → mavi/mor
    cmap = sns.diverging_palette(260, 20, s=85, l=45, as_cmap=True)

    hm = sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap=cmap,
                ax=ax, linewidths=0.5, linecolor='#0F1117',
                center=0, vmin=-1, vmax=1,
                annot_kws={'size': 7.5, 'weight': 'bold', 'color': '#F1F5F9'},
                cbar_kws={'shrink': 0.75, 'pad': 0.02})

    # Colorbar yazılarını aydınlat
    cbar = hm.collections[0].colorbar
    cbar.ax.yaxis.set_tick_params(color=TEXT_SEC)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_SEC, fontsize=8)

    ax.set_title('Özellik Korelasyon Isı Haritası', color=TEXT_PRI)
    ax.tick_params(axis='x', colors=TEXT_PRI, labelsize=9, rotation=45)
    ax.tick_params(axis='y', colors=TEXT_PRI, labelsize=9, rotation=0)
    return fig


def fig_distribution():
    fig = make_fig(w=8, h=5.5)
    ax  = fig.add_axes([0.15, 0.14, 0.72, 0.72])
    ax.set_facecolor(CARD_BG)
    counts = df['is_safe'].value_counts().sort_index()
    labels = ['Güvensiz (0)', 'Güvenli (1)']
    colors = [DANGER, SUCCESS]
    bars = ax.bar(labels, counts.values, color=colors,
                  edgecolor=BG, linewidth=2, width=0.45)
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 40,
                f'{v:,}', ha='center', fontweight='bold',
                fontsize=14, color=TEXT_PRI)
    ax.set_title('Sınıf Dağılımı')
    ax.set_ylabel('Örnek Sayısı')
    ax.grid(axis='y', linestyle='--')
    return fig


# ─────────────────────────────────────────────
# 4. TKINTER GUI
# ─────────────────────────────────────────────
TAB_DEFS = [
    ('📊  Özet',             fig_overview),
    ('🟦  Confusion Matrix', fig_confusion),
    ('📈  ROC Eğrisi',       fig_roc),
    ('⭐  Feature Importance',fig_feature),
    ##('🌡️  Korelasyon',       fig_heatmap),
    ('🥧  Sınıf Dağılımı',   fig_distribution),
]

root = tk.Tk()
root.title('Su Kalitesi Sınıflandırma — ML Sonuçları')
root.geometry('1280x780')
root.configure(bg=BG)
root.resizable(True, True)

# ── Başlık ────────────────────────────────────
header = tk.Frame(root, bg=SIDEBAR_BG, height=56)
header.pack(fill='x', side='top')
header.pack_propagate(False)

tk.Label(header, text='💧 Su Kalitesi Sınıflandırma',
         bg=SIDEBAR_BG, fg=TEXT_PRI,
         font=('Segoe UI', 16, 'bold')).pack(side='left', padx=22, pady=14)

tk.Label(header, text=f'7,996 örnek  ·  20 özellik  ·  3 model',
         bg=SIDEBAR_BG, fg=TEXT_SEC,
         font=('Segoe UI', 10)).pack(side='left', padx=4, pady=14)

# Sağda özet skorlar
for name, r in results.items():
    short = name.split()[0]
    tk.Label(header, text=f'{short}: {r["accuracy"]:.3f}',
             bg=SIDEBAR_BG, fg=MODEL_COLORS[list(results.keys()).index(name)],
             font=('Segoe UI', 10, 'bold')).pack(side='right', padx=14, pady=14)

# ── Sol kenar çubuğu ──────────────────────────
sidebar = tk.Frame(root, bg=SIDEBAR_BG, width=195)
sidebar.pack(fill='y', side='left')
sidebar.pack_propagate(False)

tk.Label(sidebar, text='GRAFİKLER',
         bg=SIDEBAR_BG, fg=TEXT_SEC,
         font=('Segoe UI', 8, 'bold')).pack(anchor='w', padx=18, pady=(18,6))

# ── İçerik alanı ──────────────────────────────
content = tk.Frame(root, bg=BG)
content.pack(fill='both', expand=True)

canvas_holder = tk.Frame(content, bg=BG)
canvas_holder.pack(fill='both', expand=True, padx=14, pady=14)

current_canvas = [None]
current_fig    = [None]

def show_tab(idx, btn_list):
    # Buton stillerini güncelle
    for i, b in enumerate(btn_list):
        if i == idx:
            b.config(bg=SEL_BG, fg=TEXT_PRI,
                     relief='flat', bd=0,
                     font=('Segoe UI', 10, 'bold'))
            b._indicator.config(bg=ACCENT)
        else:
            b.config(bg=SIDEBAR_BG, fg=TEXT_SEC,
                     relief='flat', bd=0,
                     font=('Segoe UI', 10))
            b._indicator.config(bg=SIDEBAR_BG)

    # Eski canvas'ı temizle
    if current_canvas[0]:
        current_canvas[0].get_tk_widget().destroy()
    if current_fig[0]:
        plt.close(current_fig[0])

    # Yeni grafik
    fig = TAB_DEFS[idx][1]()
    current_fig[0] = fig

    cv = FigureCanvasTkAgg(fig, master=canvas_holder)
    cv.draw()
    cv.get_tk_widget().pack(fill='both', expand=True)
    current_canvas[0] = cv


# Kenar çubuğu butonları
btn_list = []
for i, (label, _) in enumerate(TAB_DEFS):
    row = tk.Frame(sidebar, bg=SIDEBAR_BG)
    row.pack(fill='x', pady=1)

    indicator = tk.Frame(row, bg=SIDEBAR_BG, width=4)
    indicator.pack(side='left', fill='y')

    btn = tk.Button(row, text=f'  {label}',
                    bg=SIDEBAR_BG, fg=TEXT_SEC,
                    activebackground=SEL_BG,
                    activeforeground=TEXT_PRI,
                    relief='flat', bd=0,
                    anchor='w', padx=12, pady=10,
                    font=('Segoe UI', 10),
                    cursor='hand2')
    btn._indicator = indicator
    btn.config(command=lambda i=i: show_tab(i, btn_list))
    btn.pack(side='left', fill='x', expand=True)
    btn_list.append(btn)

# Başlangıç
show_tab(0, btn_list)

root.mainloop()
