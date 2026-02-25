import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import tempfile

# ══════════════════════════════════════════════════════════════
# Arquitectura del modelo — debe ser IDÉNTICA a la del notebook
# ══════════════════════════════════════════════════════════════
class RNNPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, output_size=1):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity='tanh'
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])  # último estado oculto → predicción


# ══════════════════════════════════════════════════════════════
# Cargar modelo con caché (no se recarga en cada interacción)
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def cargar_modelo(ruta):
    ckpt  = torch.load(ruta, map_location='cpu')
    model = RNNPredictor(hidden_size=ckpt['hidden_size'])
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, ckpt['window'], ckpt['hidden_size']


# ══════════════════════════════════════════════════════════════
# Configuración de página
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title = "RNN Demo",
    page_icon  = "🧠",
    layout     = "wide"
)

st.title("🧠 RNN — Predicción de Secuencias Temporales")
st.caption("Sube el archivo `rnn_model.pth` generado en Colab para activar el modelo.")

# ══════════════════════════════════════════════════════════════
# Panel lateral
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("📂 Cargar Modelo")
    modelo_file = st.file_uploader(
        label = "Archivo rnn_model.pth",
        type  = ["pth"],
        help  = "Genera este archivo ejecutando el notebook rnn_colab.ipynb en Google Colab"
    )

    st.divider()
    st.header("⚙️ Parámetros de señal")
    tipo_serie = st.selectbox("Tipo de señal", ["Seno", "Coseno", "Seno + ruido"])
    freq       = st.slider("Frecuencia", 0.5, 5.0, 1.0, step=0.5)
    n_pasos    = st.slider("Pasos a predecir hacia adelante", 1, 80, 30)

    st.divider()
    st.markdown("**Sobre este demo:**")
    st.markdown(
        "Esta app usa una RNN entrenada en Colab para predecir la continuación "
        "de señales periódicas. La RNN mantiene un estado oculto interno que actúa "
        "como memoria de los valores anteriores."
    )


# ══════════════════════════════════════════════════════════════
# Sección conceptual (siempre visible)
# ══════════════════════════════════════════════════════════════
with st.expander("💡 ¿Cómo funciona una RNN? (haz clic para expandir)", expanded=False):
    col_txt, col_eq = st.columns([3, 2])
    with col_txt:
        st.markdown("""
        ### Red Neuronal Recurrente

        A diferencia de una red densa, la RNN **procesa secuencias paso a paso**
        y mantiene un **estado oculto** $h_t$ que actúa como memoria:

        $$h_t = \\tanh(W_{hh} \\cdot h_{t-1} + W_{xh} \\cdot x_t + b_h)$$
        $$y_t = W_{hy} \\cdot h_t + b_y$$

        En cada paso de tiempo $t$, el estado oculto combina:
        - La **memoria anterior** $h_{t-1}$
        - La **entrada actual** $x_t$

        Los pesos $W$ se **comparten en todos los pasos de tiempo**, permitiendo
        generalizar sobre secuencias de cualquier longitud.
        """)
    with col_eq:
        st.markdown("""
        ### Flujo de datos en este modelo

        ```
        Entrada:  [x₁, x₂, ..., x₂₀]
                        ↓
          ┌─────────────────────────┐
          │  RNN  (hidden=32)       │
          │  h₁ → h₂ → ... → h₂₀  │
          └─────────────────────────┘
                        ↓  h₂₀
          ┌─────────────────────────┐
          │  Linear (32 → 1)        │
          └─────────────────────────┘
                        ↓
                  Predicción ŷ
        ```
        """)

st.divider()


# ══════════════════════════════════════════════════════════════
# Pantalla de bienvenida si no hay modelo cargado
# ══════════════════════════════════════════════════════════════
if modelo_file is None:
    st.info("⬅️  Sube el archivo `rnn_model.pth` en el panel lateral para comenzar.")

    st.subheader("📋 Instrucciones")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        **Paso 1 — Abrir Colab**

        Descarga `rnn_colab.ipynb` y ábrelo en
        [Google Colab](https://colab.research.google.com).
        Activa GPU en *Entorno de ejecución → Cambiar tipo de entorno*.
        """)
    with c2:
        st.markdown("""
        **Paso 2 — Entrenar el modelo**

        Ejecuta todas las celdas del notebook.
        El entrenamiento toma ~30 segundos en CPU y ~10 s en GPU.
        La última celda descarga `rnn_model.pth` automáticamente.
        """)
    with c3:
        st.markdown("""
        **Paso 3 — Usar esta app**

        Sube el archivo `.pth` usando el panel lateral izquierdo.
        Configura el tipo de señal y la cantidad de pasos a predecir.
        Explora las pestañas de análisis.
        """)
    st.stop()


# ══════════════════════════════════════════════════════════════
# Cargar modelo desde el archivo subido
# ══════════════════════════════════════════════════════════════
with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp:
    tmp.write(modelo_file.read())
    tmp_path = tmp.name

try:
    model, WINDOW, HIDDEN = cargar_modelo(tmp_path)
    st.success(f"✅ Modelo cargado — ventana: **{WINDOW}** pasos | neuronas ocultas: **{HIDDEN}**")
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()


# ══════════════════════════════════════════════════════════════
# Tabs principales
# ══════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "📈 Predicción automática",
    "✏️  Secuencia manual",
    "🔬 Análisis del modelo"
])


# ── TAB 1: Predicción automática ──────────────────────────────
with tab1:
    st.markdown(
        "La RNN toma los últimos **{} pasos** de la señal seleccionada y predice "
        "los siguientes **{}** valores de forma autorrregresiva: cada predicción se "
        "usa como entrada para la siguiente.".format(WINDOW, n_pasos)
    )

    # Generar señal
    t = np.linspace(0, 20, 500)
    if tipo_serie == "Seno":
        serie = np.sin(freq * t)
    elif tipo_serie == "Coseno":
        serie = np.cos(freq * t)
    else:
        np.random.seed(42)
        serie = np.sin(freq * t) + np.random.normal(0, 0.15, len(t))

    # Predicción autorrregresiva
    secuencia    = list(serie[-WINDOW:].astype(np.float32))
    predicciones = []
    for _ in range(n_pasos):
        x = torch.FloatTensor(secuencia[-WINDOW:]).unsqueeze(0).unsqueeze(-1)
        with torch.no_grad():
            pred = model(x).item()
        predicciones.append(pred)
        secuencia.append(pred)

    # Plot
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
        [serie.min() - 0.1, serie.max() + 0.1],
        n_show - 1, n_show + n_pasos - 1,
        alpha=0.05, color='tomato', label='Zona predicha'
    )
    ax.set_xlim(0, n_show + n_pasos)
    ax.set_title("Continuación de la señal predicha por la RNN")
    ax.set_xlabel("Paso de tiempo")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

    # Métricas
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pasos predichos",    n_pasos)
    c2.metric("Último valor real",  f"{serie[-1]:.4f}")
    c3.metric("1ª predicción",      f"{predicciones[0]:.4f}")
    c4.metric("Última predicción",  f"{predicciones[-1]:.4f}")


# ── TAB 2: Secuencia manual ────────────────────────────────────
with tab2:
    st.markdown(
        f"Ingresa exactamente **{WINDOW} valores numéricos** separados por coma. "
        "La RNN procesará esa secuencia y predecirá el siguiente valor."
    )

    default = ", ".join([f"{np.sin(i * 0.3):.3f}" for i in range(WINDOW)])
    seq_input = st.text_area(
        f"Secuencia de entrada ({WINDOW} valores)",
        value   = default,
        height  = 80,
        help    = "Puedes pegar valores de cualquier señal periódica"
    )

    col_btn, col_space = st.columns([1, 3])
    with col_btn:
        predecir = st.button("🔮 Predecir siguiente valor", use_container_width=True)

    if predecir:
        try:
            valores = [float(v.strip()) for v in seq_input.split(",")]
            if len(valores) < WINDOW:
                st.error(f"Se necesitan {WINDOW} valores. Ingresaste {len(valores)}.")
            else:
                vals = valores[-WINDOW:]
                x    = torch.FloatTensor(vals).unsqueeze(0).unsqueeze(-1)
                with torch.no_grad():
                    pred = model(x).item()

                fig2, ax2 = plt.subplots(figsize=(9, 3))
                ax2.plot(range(WINDOW), vals,
                         "o-", color='steelblue', linewidth=2, label='Secuencia ingresada')
                ax2.plot(WINDOW, pred,
                         "r*", markersize=18, zorder=5, label=f'Predicción: {pred:.4f}')
                ax2.axvline(x=WINDOW - 0.5, color='gray', linestyle=':', alpha=0.5)
                ax2.set_title("Predicción del siguiente valor en la secuencia")
                ax2.set_xlabel("Paso de tiempo")
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig2)

                st.success(f"**Siguiente valor predicho: `{pred:.6f}`**")
        except ValueError:
            st.error("Formato inválido. Usa números separados por comas, ej: 0.1, 0.3, 0.6 ...")


# ── TAB 3: Análisis del modelo ─────────────────────────────────
with tab3:
    st.subheader("Parámetros del modelo")

    col_a, col_b = st.columns(2)
    with col_a:
        total = sum(p.numel() for p in model.parameters())
        st.metric("Total parámetros",       f"{total:,}")
        st.metric("Neuronas ocultas",       HIDDEN)
        st.metric("Ventana de entrada",     f"{WINDOW} pasos")
        st.metric("Features de entrada",    "1")

        st.markdown("**Capas y dimensiones:**")
        for name, param in model.named_parameters():
            st.code(f"{name:30s}  →  {list(param.shape)}")

    with col_b:
        st.markdown("""
        **¿Por qué estos parámetros?**

        Con `hidden_size=32` y una sola capa RNN, los pesos son:

        | Tensor | Forma | Descripción |
        |--------|-------|-------------|
        | `W_ih` | (32, 1) | entrada → estado oculto |
        | `W_hh` | (32, 32) | estado oculto → estado oculto |
        | `b_ih` | (32,) | bias de la capa de entrada |
        | `b_hh` | (32,) | bias de la capa recurrente |
        | `fc.weight` | (1, 32) | estado oculto → predicción |
        | `fc.bias` | (1,) | bias de salida |

        Total: `32×1 + 32×32 + 32 + 32 + 32 + 1 = 1,153` parámetros
        """)

    st.divider()
    st.subheader("Visualización del estado oculto")
    st.markdown(
        "Este heatmap muestra cómo evolucionan las 16 primeras neuronas ocultas "
        "al procesar un ciclo completo de señal sinusoidal. Cada fila es una neurona; "
        "cada columna es un paso de tiempo. Los colores indican la activación (rojo = alto, azul = bajo)."
    )

    t_vis  = np.linspace(0, 6.28, WINDOW)
    sv     = np.sin(t_vis).astype(np.float32)
    x_vis  = torch.FloatTensor(sv).unsqueeze(0).unsqueeze(-1)

    with torch.no_grad():
        out_vis, _ = model.rnn(x_vis)
        estados    = out_vis[0].numpy()  # (WINDOW, hidden_size)

    fig3, axes = plt.subplots(1, 2, figsize=(12, 3))

    axes[0].plot(sv, "o-", color='steelblue', linewidth=2)
    axes[0].set_title("Señal de entrada (un ciclo de seno)")
    axes[0].set_xlabel("Paso de tiempo")
    axes[0].set_ylabel("Valor")
    axes[0].grid(True, alpha=0.3)

    n_show_neurons = min(16, HIDDEN)
    im = axes[1].imshow(
        estados.T[:n_show_neurons],
        aspect  = 'auto',
        cmap    = 'RdBu',
        vmin    = -1, vmax = 1
    )
    axes[1].set_title(f"Estados ocultos — primeras {n_show_neurons} neuronas")
    axes[1].set_xlabel("Paso de tiempo")
    axes[1].set_ylabel("Neurona oculta")
    plt.colorbar(im, ax=axes[1], label='Activación')

    plt.tight_layout()
    st.pyplot(fig3)

    st.caption(
        "Se puede observar cómo distintas neuronas capturan diferentes aspectos de la señal: "
        "algunas se activan en los máximos, otras en los cruces por cero, creando una "
        "representación distribuida del patrón periódico."
    )
