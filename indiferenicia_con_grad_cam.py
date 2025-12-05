import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import os
import numpy as np
import cv2 

# Instala OpenCV si no lo tienes: pip install opencv-python
try:
    import cv2
except ImportError:
    st.error("❌ ERROR: Falta la librería 'opencv-python'. Por favor, instálala con: pip install opencv-python")
    st.stop()

# --- CONFIGURACIÓN ---
# RUTA CORREGIDA: Asume que el modelo está en la misma carpeta que el script.
MODEL_PATH = "modelo_cancer_mobilenet.pth"
CLASS_NAMES = ["Benigno", "Maligno", "Normal"] 
NUM_CLASSES = len(CLASS_NAMES)

# Variables globales para guardar los datos de los hooks de Grad-CAM
feature_maps = None
gradients = None

# Configuración de la página
st.set_page_config(page_title="Diagnóstico IA con Grad-CAM", page_icon="🩺", layout="wide")

# =====================================================================
# 📣 ASISTENTE DE PREVENCIÓN (MOVIDO A LA BARRA LATERAL)
# =====================================================================

with st.sidebar:
    st.header("💬 Asistente de Prevención")
    st.write("Selecciona una pestaña para obtener información y recursos de seguimiento.")

    tab1, tab2, tab3 = st.tabs(["✋ Autoexamen Casero", "🏥 Lugares de Seguimiento", "💡 Consejos de Salud"])

    with tab1:
        st.subheader("1. Autoexamen Casero de Mamas (Palpación)")
        st.markdown("La autoexploración ayuda a familiarizarse con la apariencia y sensación normal de las mamas. **Realízala una vez al mes**, de 3 a 5 días después del inicio de tu periodo.")
        st.markdown("""
        Aquí tienes los pasos clave para el autoexamen:
        * **Inspección Visual (Frente al espejo):** Observa la presencia de arrugas, hoyuelos, alteraciones en el tamaño o forma, o si los pezones están hundidos. Repite con los brazos a los lados y levantados.
        * **Palpación (Acostada o de pie):** Utiliza las yemas de los tres dedos del medio.
            * Usa **tres niveles de presión** (ligera, media y firme).
            * Sigue un patrón metódico (círculos o líneas verticales) para cubrir toda la mama, desde la axila hasta el esternón.
        * **Alerta:** Reporta a tu médico cualquier bulto, secreción, cambio de textura o dolor que notes.
        """)
        # 

    with tab2:
        st.subheader("2. Lugares de Seguimiento en El Salvador")
        st.markdown("Si el diagnóstico de la IA es **Maligno** o tienes dudas, consulta con un especialista oncólogo o mastólogo en los siguientes centros de referencia:")
        st.markdown("""
        * **Instituto Salvadoreño del Seguro Social (ISSS):** Ofrece diagnóstico y tratamiento oncológico.
        * **Hospital Nacional Rosales / Hospital Nacional de la Mujer (MINSAL):** Referencia en el sistema de salud pública.
        * **ASAPRECAN - Clínica de Mama 'Isabella Carle':** Establecimiento especializado en patología mamaria (consultar horarios y requisitos).
        * **Clínicas Oncológicas Privadas:** Clínicas Oncológicas y Cancer Research (San Salvador), Centro Salvadoreño de Radioterapia, Unidad de Oncología Hospital San Francisco (San Miguel).
        """)
        st.warning("⚠️ **Recordatorio:** Este prototipo de IA NO reemplaza el diagnóstico de un médico especialista. Actúa con rapidez si tienes un resultado Maligno.")

    with tab3:
        st.subheader("3. Consejos para la Salud Mamaria y Prevención")
        st.markdown("""
        * **Control de Peso:** Mantener un peso saludable puede reducir el riesgo.
        * **Actividad Física:** Realizar ejercicio de forma regular (al menos 150 minutos a la semana).
        * **Dieta:** Consumir una dieta rica en frutas, verduras y granos integrales. Limita el consumo de alcohol.
        * **Lactancia:** Si es posible, la lactancia materna por más de un año está asociada a un menor riesgo.
        * **Mamografía:** Si tienes 40 años o más, realiza una mamografía anualmente, según indicación médica.
        """)
        st.info("💡 **Consejo:** El diagnóstico temprano es la clave. Usa esta herramienta como una alerta y no como una certeza médica.")

# =====================================================================
# 📌 INTERFAZ PRINCIPAL
# =====================================================================

st.title("🩺🌷sentido rosa sv🌷")
st.markdown("### *🌺🩷Un toque de cuidado, una vida de diferencia.*") 

# --- CARGAR MODELO Y ARQUITECTURA ---
@st.cache_resource
def cargar_modelo(path):
    # LA RUTA YA ES RELATIVA, AHORA SOLO DEBEMOS COMPROBAR SU EXISTENCIA
    if not os.path.exists(path):
        st.error(f"❌ No se encontró el archivo del modelo (.pth). Asegúrate de que '{path}' esté en la misma carpeta.")
        return None, None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # 1. Arquitectura (MobileNetV2)
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(1280, NUM_CLASSES) 
        
        # 2. Cargar pesos
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
        
        model = model.to(device)
        model.eval()
        st.success(f"✅ Modelo cargado correctamente en {device}")
        return model, device
    except Exception as e:
        st.error(f"Error cargando el modelo: {e}")
        return None, None

model, device = cargar_modelo(MODEL_PATH)

# --- HOOKS Y GRAD-CAM (IMPLEMENTACIÓN MANUAL) ---

def save_feature_maps(module, input, output):
    global feature_maps
    feature_maps = output

def save_gradients(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]

def generate_grad_cam(model, input_tensor: torch.Tensor, target_layer_name: str, pred_idx: int):
    target_layer = model.features[-1]
    
    hook_handle_fwd = target_layer.register_forward_hook(save_feature_maps)
    hook_handle_bwd = target_layer.register_full_backward_hook(save_gradients)

    input_tensor.requires_grad_(True)
    outputs = model(input_tensor)
    
    model.zero_grad()
    one_hot = torch.zeros_like(outputs)
    one_hot[0, pred_idx] = 1.0
    outputs.backward(gradient=one_hot, retain_graph=True)

    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3]) 
    
    feature_maps_np = feature_maps.detach().cpu().numpy()[0]
    pooled_gradients_np = pooled_gradients.detach().cpu().numpy()
    
    for i in range(pooled_gradients_np.shape[0]):
        feature_maps_np[i, :, :] *= pooled_gradients_np[i]

    heatmap = np.sum(feature_maps_np, axis=0)
    heatmap = np.maximum(heatmap, 0)
    
    hook_handle_fwd.remove()
    hook_handle_bwd.remove()
    
    return heatmap

def apply_heatmap_to_image(image_pil: Image.Image, heatmap_np: np.ndarray):
    heatmap_resized = cv2.resize(heatmap_np, (224, 224))
    heatmap_norm = heatmap_resized / np.max(heatmap_resized)
    heatmap_uint8 = np.uint8(255 * heatmap_norm)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    img_np = np.array(image_pil.resize((224, 224)))
    
    superimposed_img = cv2.addWeighted(img_np, 0.5, heatmap_color, 0.5, 0)
    
    return Image.fromarray(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))


# --- INTERFAZ DE PREDICCIÓN (Principal) ---

if model:
    st.markdown("---")
    # Subir imagen
    uploaded_file = st.file_uploader("Sube una imagen de ultrasonido/mamografía", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        
        col1, col2 = st.columns(2) 
        
        with col1:
            st.subheader("Imagen Original")
            st.image(image, caption="Imagen subida", use_container_width=True)
        
        if st.button("Analizar y Visualizar"):
            # 1. Preparación del Tensor
            input_tensor = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])(image).unsqueeze(0).to(device)
            
            # 2. Obtener Predicción
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)[0]
                pred_idx = outputs.argmax(dim=1).item() 
            
            resultado = CLASS_NAMES[pred_idx]
            confianza = probs[pred_idx].item() * 100

            # 3. Generar Grad-CAM
            heatmap_np = generate_grad_cam(model, input_tensor, 'features[-1]', pred_idx)
            
            # 4. Aplicar Mapa de Calor a la imagen
            heatmap_image = apply_heatmap_to_image(image, heatmap_np)
            
            st.markdown("---")
            
            # Mostrar resultados de clasificación
            st.subheader("📝 Resultado del Diagnóstico")
            
            col_res, col_msg = st.columns([1, 2])
            
            with col_res:
                st.metric(label="Clasificación", value=resultado.upper(), delta=f"{confianza:.2f}% Confianza")

            with col_msg:
                if resultado == "Maligno":
                    st.error("⚠️ Patrón Maligno detectado. Se recomienda análisis médico urgente.")
                elif resultado == "Benigno":
                    st.warning("🟡 Patrón Benigno detectado. Se recomienda seguimiento.")
                else: # Normal
                    st.success("✅ Patrón Normal detectado. Buen estado.")

            # Mostrar el mapa de calor
            with col2:
                st.subheader("Mapa de Calor (Grad-CAM)")
                st.image(heatmap_image, caption="Región de interés destacada", use_container_width=True)
                st.info("El mapa de calor (rojo/amarillo) muestra la región más influyente en la decisión de la IA.")

