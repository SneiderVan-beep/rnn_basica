import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import tempfile

# ══════════════════════════════════════════════════════════════
# Arquitectura — debe ser IDÉNTICA a la del notebook
# ══════════════════════════════════════════════════════════════
class RNNMejorada(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        self.rnn = nn.RNN(
            input_size   = input_size,
            hidden_size  = hidden_size,
            num_layers   = num_layers,
            batch_first  = True,
            nonlinearity = 'tanh',
            dropout      = dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(self.dropout(out[:, -1, :]))


@st.cache_resource
def cargar_modelo(ruta):
    ckpt = torch.load(ruta, map_location='cpu')
    model = RNNMejorada(
        hidden_size = ckpt['hidden_size'],
        num_layers  = ckpt.get('num_layers', 2)
    )
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, ckpt['window'], ckpt.get('media', 0.0), ckpt.get('std', 1.0)


# ══════════════════════════════════════════════════════════════
# Página
# ══════════════════════════════════════════════════════════════
st.set_page_config(page_title="RNN Demo", page_icon="🧠", layout="wide")
st.title("🧠 RNN Mejorada — Predicción de Secuencias Temporales")
st.caption("Sube el archivo `rnn_model.pth` generado en Colab para activar el modelo.")

with st.sidebar:
    st.header("📂 Cargar Modelo")
    modelo_file = st.file_uploader("Archivo rnn_model.pth", type=["pth"])
    st.divider()
    st.header("⚙️ Parámetros")
    tipo_serie = st.selectbox("Tipo de señal", ["Seno", "Coseno", "Seno + ruido"])
    freq       = st.slider("Frecuencia", 0.3, 5.0, 1.0, step=0.1)
    n_pasos    = st.slider("Pasos a predecir", 1, 80, 30)

with st.expander("💡 Mejoras aplicadas en esta versión", expanded=False):
    st.markdown("""
    | Técnica | Beneficio |
    |---------|-----------|
    | **6 frecuencias de entrenamiento** | Generaliza a señales que no vio antes |
    | **Ruido en datos** | Robustez frente a secuencias imperfectas |
    | **2 capas RNN + dropout** | Más capacidad sin overfitting |
    | **DataLoader con shuffle** | Mejor generalización |
    | **Gradient clipping** | Estabilidad en el entrenamiento |
    | **Scheduler + Early stopping** | Mejor convergencia automática |
    | **Normalización** | Escala adecuada para activación tanh |
    """)

st.divider()

if modelo_file is None:
    st.info("⬅️  Sube el archivo `rnn_model.pth` en el panel lateral para comenzar.")
    st.stop()

# Cargar modelo
with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp:
    tmp.write(modelo_file.read())
    tmp_path = tmp.name

try:
    model, WINDOW, MEDIA, STD = cargar_modelo(tmp_path)
    st.success(f"✅ Modelo cargado — ventana: **{WINDOW}** pasos | normalización: media={MEDIA:.4f}, std={STD:.4f}")
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()


def normalizar(arr):
    return (arr - MEDIA) / STD

def desnormalizar(arr):
    return arr * STD + MEDIA


tab1, tab2, tab3 = st.tabs(["📈 Predicción automática", "✏️ Secuencia manual", "🔬 Análisis del modelo"])

# ── TAB 1 ──────────────────────────────────────────────────────
with tab1:
    st.markdown(
        f"La RNN toma los últimos **{WINDOW} pasos** normalizados y predice los siguientes "
        f"**{n_pasos}** valores de forma autorregresiva."
    )

    t = np.linspace(0, 20, 500)
    if tipo_serie == "Seno":
        serie = np.sin(freq * t)
    elif tipo_serie == "Coseno":
        serie = np.cos(freq * t)
    else:
        serie = np.sin(freq * t) + np.random.normal(0, 0.1, len(t))

    # Predicción autorregresiva con normalización
    secuencia    = list(serie[-WINDOW:].astype(np.float32))
    predicciones = []

    for _ in range(n_pasos):
        ventana_norm = normalizar(np.array(secuencia[-WINDOW:]))
        x = torch.FloatTensor(ventana_norm).unsqueeze(0).unsqueeze(-1)
        with torch.no_grad():
            pred_norm = model(x).item()
        pred = desnormalizar(pred_norm)
        predicciones.append(pred)
        secuencia.append(pred)

    n_show   = 60
    idx_base = np.arange(n_show)
    idx_pred = np.arange(n_show - 1, n_show + n_pasos - 1)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(idx_base, serie[-n_show:],
            color='royalblue', linewidth=2, label='Señal original')
    ax.plot(idx_pred, predicciones,
            color='tomato', linewidth=2, linestyle='--',
            label=f'Predicción RNN ({n_pasos} pasos)')
    ax.axvline(x=n_show - 1, color='gray', linestyle=':', alpha=0.6)
    ax.fill_betweenx(
        [serie.min() - 0.2, serie.max() + 0.2],
        n_show - 1, n_show + n_pasos - 1,
        alpha=0.05, color='tomato'
    )
    ax.set_xlim(0, n_show + n_pasos)
    ax.set_title("Continuación predicha por la RNN")
    ax.set_xlabel("Paso de tiempo")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pasos predichos",   n_pasos)
    c2.metric("Último valor real", f"{serie[-1]:.4f}")
    c3.metric("1ª predicción",     f"{predicciones[0]:.4f}")
    c4.metric("Última predicción", f"{predicciones[-1]:.4f}")

# ── TAB 2 ──────────────────────────────────────────────────────
with tab2:
    st.markdown(
        f"Ingresa **{WINDOW} valores** separados por coma. "
        "El modelo normaliza la entrada internamente antes de predecir."
    )

    default = ", ".join([f"{np.sin(i * 0.3):.3f}" for i in range(WINDOW)])
    seq_input = st.text_area(f"Secuencia ({WINDOW} valores)", value=default, height=80)

    if st.button("🔮 Predecir siguiente valor", use_container_width=False):
        try:
            valores = [float(v.strip()) for v in seq_input.split(",")]
            if len(valores) < WINDOW:
                st.error(f"Necesitas {WINDOW} valores. Tienes {len(valores)}.")
            else:
                vals      = np.array(valores[-WINDOW:], dtype=np.float32)
                vals_norm = normalizar(vals)
                x = torch.FloatTensor(vals_norm).unsqueeze(0).unsqueeze(-1)
                with torch.no_grad():
                    pred = desnormalizar(model(x).item())

                fig2, ax2 = plt.subplots(figsize=(9, 3))
                ax2.plot(range(WINDOW), vals, "o-", color='steelblue',
                         linewidth=2, label='Secuencia ingresada')
                ax2.plot(WINDOW, pred, "r*", markersize=18, zorder=5,
                         label=f'Predicción: {pred:.4f}')
                ax2.axvline(x=WINDOW - 0.5, color='gray', linestyle=':', alpha=0.5)
                ax2.legend()
                ax2.set_title("Predicción del siguiente valor")
                ax2.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig2)
                st.success(f"**Siguiente valor predicho: `{pred:.6f}`**")
        except ValueError:
            st.error("Formato inválido. Usa números separados por comas.")

# ── TAB 3 ──────────────────────────────────────────────────────
with tab3:
    st.subheader("Parámetros del modelo")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Total parámetros",   f"{sum(p.numel() for p in model.parameters()):,}")
        st.metric("Capas RNN",          model.rnn.num_layers)
        st.metric("Neuronas ocultas",   model.rnn.hidden_size)
        st.metric("Ventana de entrada", f"{WINDOW} pasos")
        st.metric("Media normalización", f"{MEDIA:.4f}")
        st.metric("Std normalización",   f"{STD:.4f}")
    with c2:
        st.markdown("**Capas y formas:**")
        for name, param in model.named_parameters():
            st.code(f"{name:35s} → {list(param.shape)}")

    st.divider()
    st.subheader("Visualización del estado oculto")

    t_vis = np.linspace(0, 6.28, WINDOW)
    sv    = np.sin(t_vis).astype(np.float32)
    sv_n  = normalizar(sv)
    x_vis = torch.FloatTensor(sv_n).unsqueeze(0).unsqueeze(-1)

    with torch.no_grad():
        out_vis, _ = model.rnn(x_vis)
        estados    = out_vis[0].numpy()

    fig3, axes = plt.subplots(1, 2, figsize=(12, 3))
    axes[0].plot(sv, "o-", color='steelblue', linewidth=2)
    axes[0].set_title("Señal de entrada (un ciclo de seno)")
    axes[0].set_xlabel("Paso de tiempo")
    axes[0].grid(True, alpha=0.3)

    n_vis = min(16, model.rnn.hidden_size)
    im = axes[1].imshow(estados.T[:n_vis], aspect='auto', cmap='RdBu', vmin=-1, vmax=1)
    axes[1].set_title(f"Estados ocultos — primeras {n_vis} neuronas")
    axes[1].set_xlabel("Paso de tiempo")
    axes[1].set_ylabel("Neurona oculta")
    plt.colorbar(im, ax=axes[1], label='Activación')
    plt.tight_layout()
    st.pyplot(fig3)

    st.caption(
        "Cada fila es una neurona; cada columna un paso de tiempo. "
        "Se observa cómo distintas neuronas capturan fases diferentes del ciclo."
    )
