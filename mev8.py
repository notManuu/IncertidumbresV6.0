import streamlit as st
import numpy as np
import pandas as pd
import math
import re

st.set_page_config(page_title="Calculadora de Incertidumbre GUM", layout="wide")

# --- 1. CONSTANTES Y FUNCIONES DE CÁLCULO ---

# Diccionario con la columna 95.45% de tu tabla t-Student
T_9545_TABLE = {
    1: 13.97, 2: 4.527, 3: 3.307, 4: 2.869, 5: 2.649, 6: 2.517, 7: 2.429,
    8: 2.366, 9: 2.320, 10: 2.284, 11: 2.255, 12: 2.231, 13: 2.212,
    14: 2.195, 15: 2.181, 16: 2.169, 17: 2.158, 18: 2.149, 19: 2.140,
    20: 2.133, 21: 2.126, 22: 2.120, 23: 2.115, 24: 2.110, 25: 2.105,
    26: 2.101, 27: 2.097, 28: 2.093, 29: 2.090, 30: 2.087, 35: 2.074,
    40: 2.064, 50: 2.051, 100: 2.025, 200: 2.016, 'inf': 2.000
}

def get_t_factor(v):
    """Busca el factor t (k) para 95.45% de confianza."""
    if v == 'inf' or v >= 200:
        return T_9545_TABLE['inf']
    v = math.floor(v) 
    v_keys = sorted([k for k in T_9545_TABLE.keys() if isinstance(k, int)])
    if v in v_keys:
        return T_9545_TABLE[v]
    for key_v in reversed(v_keys):
        if v >= key_v:
            return T_9545_TABLE[key_v]
    return T_9545_TABLE[1]

def parse_measurements(data_str):
    """Toma un string de mediciones y devuelve un array de numpy."""
    numbers = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+\.\d*e[-+]?\d+|[-+]?\d+e[-+]?\d+|[-+]?\d+", data_str)
    if not numbers:
        return np.array([])
    return np.array([float(n) for n in numbers])

def calculate_type_a_stats(data_array):
    """Calcula todos los estadísticos relevantes para el Tipo A."""
    stats = {'n': 0, 'v': 0, 'mean': None, 'var': 0.0, 'std': 0.0, 'u_a': 0.0}
    n = len(data_array)
    stats['n'] = n
    if n == 0:
        return stats
    elif n == 1:
        stats['mean'] = data_array[0]; stats['v'] = 0; return stats
    stats['v'] = n - 1
    stats['mean'] = np.mean(data_array)
    stats['var'] = np.var(data_array, ddof=1)
    stats['std'] = np.std(data_array, ddof=1)
    stats['u_a'] = stats['std'] / np.sqrt(n)
    return stats

def calculate_type_b(b_type, params):
    """Calcula la incertidumbre Tipo B basado en la selección."""
    try:
        if b_type == "Analógico (Clase, Rectangular)":
            clase = params.get("clase", 0.0); alcance = params.get("alcance", 0.0)
            if clase <= 0 or alcance <= 0: return 0.0, "Clase y Alcance deben ser > 0"
            error_max = (clase / 100.0) * alcance
            u_b = error_max / np.sqrt(3)
            return u_b, f"Error max = ±{error_max:.4g}. u_b = {error_max:.4g} / sqrt(3) = {u_b:.4g}"

        elif b_type == "Digital (Exactitud, Rectangular)":
            lectura = params.get("lectura", 0.0); porcentaje = params.get("porcentaje", 0.0)
            cuentas = params.get("cuentas", 0); alcance = params.get("alcance", 0.0)
            max_cuentas = params.get("max_cuentas", 1)
            if max_cuentas == 0: return 0.0, "Max Cuentas no puede ser 0"
            resolucion = alcance / max_cuentas
            error_lectura = (porcentaje / 100.0) * abs(lectura)
            error_digitos = cuentas * resolucion
            error_max = error_lectura + error_digitos
            u_b = error_max / np.sqrt(3)
            return u_b, f"Error max = ({error_lectura:.4g} + {error_digitos:.4g}) = ±{error_max:.4g}. u_b = {error_max:.4g} / sqrt(3) = {u_b:.4g}"

        elif b_type == "Especificación (Límites, Rectangular)":
            limite_error = params.get("limite_error", 0.0)
            if limite_error <= 0: return 0.0, "Límite de error debe ser > 0"
            u_b = limite_error / np.sqrt(3)
            return u_b, f"u_b = {limite_error:.4g} / sqrt(3) = {u_b:.4g}"

        elif b_type == "Resistencia (±1%, Rectangular)":
            resistencia = params.get("resistencia", 0.0)
            if resistencia <= 0: return 0.0, "Resistencia debe ser > 0"
            error_max = 0.01 * resistencia
            u_b = error_max / np.sqrt(3)
            return u_b, f"Error max = ±{error_max:.4g} (1% de {resistencia}Ω). u_b = {error_max:.4g} / sqrt(3) = {u_b:.4g}"

        elif b_type == "Estándar Conocida (u, Normal)":
            u_std = params.get("u_std", 0.0)
            if u_std <= 0: return 0.0, "u_std debe ser > 0"
            return u_std, f"u_b = u_std = {u_std:.4g} (k=1)"
        
        else: return 0.0, "No se calcula u_B."
    except Exception as e:
        return 0.0, f"Error en cálculo de u_B: {e}"

def create_function(var_names, func_expr):
    """Crea una función de Python callable a partir de un string."""
    allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
    allowed_names.update({k: v for k, v in np.__dict__.items() if not k.startswith("__")})
    allowed_names['np'] = np
    allowed_names['math'] = math
    var_str = ", ".join(var_names)
    lambda_str = f"lambda {var_str}: {func_expr}"
    try:
        f = eval(lambda_str, {"__builtins__": {}}, allowed_names)
        return f, None
    except Exception as e:
        return None, f"Error al crear la función: {e}. Asegúrate de usar sintaxis Python (ej: 'x**2', 'math.sin(y)')"

def calculate_sensitivity(f, point_dict, var_to_diff):
    """Calcula la derivada parcial (sensibilidad) por diferencias finitas."""
    h_base = 1e-8
    point_plus_h = point_dict.copy(); point_minus_h = point_dict.copy()
    x_val = point_dict[var_to_diff]
    h = h_base * max(1.0, abs(x_val))
    if h == 0: h = h_base
    point_plus_h[var_to_diff] = x_val + h
    point_minus_h[var_to_diff] = x_val - h
    try:
        f_plus = f(**point_plus_h)
        f_minus = f(**point_minus_h)
        derivative = (f_plus - f_minus) / (2.0 * h)
        return derivative, None
    except Exception as e:
        return 0.0, f"Error al evaluar la función para {var_to_diff}: {e}"

# --- 2. INTERFAZ DE STREAMLIT ---

st.title("🔬 Calculadora de Incertidumbre GUM v6.0 (Intuitiva)")
st.info("**Bienvenido:** Sigue 3 pasos. 1. Define tu ecuación y variables. 2. Configura cada variable (su valor y sus incertidumbres). 3. Presiona 'Calcular'.")

# --- Barra lateral ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a9/IPN-Logo.svg/1200px-IPN-Logo.svg.png", width=100)
    st.header("Configuración de Cálculo")
    st.info("Esta aplicación usa la **tabla t-Student (95.45% de confianza)** para determinar el factor de cobertura $k$ final.")
    
    st.number_input(
        "Grados de Libertad (Tipo B)", 
        min_value=1, 
        value=8, 
        key="v_b_eff", 
        help="Grados de libertad para fuentes Tipo B. El PDF de ejemplo asume v=8 para una confiabilidad del 25%."
    )
    
    st.markdown("---")
    if st.button("🔴 Reiniciar Todo (Borrar Memoria)", help="Borra todos los datos y variables. Útil en celular si la app se bloquea."):
        st.session_state.clear()
        st.rerun()

if 'variables' not in st.session_state:
    st.session_state.variables = {}

# --- PASO 1: Definir el Mensurando (Ecuación) ---
with st.container(border=True):
    st.header("1. Definición del Mensurando (Ecuación)")
    col1, col2 = st.columns([1, 2])
    with col1:
        var_str = st.text_input(
            "Variables de entrada (en orden, separadas por coma)", 
            "V1, I, R_A, t, alpha",
            help="Escribe las variables en el orden que quieras verlas."
        )
        var_list = [v.strip() for v in var_str.split(',') if v.strip()]
        var_names = list(dict.fromkeys(var_list)) # Preserva orden
        
    with col2:
        func_expr = st.text_input(
            "Ecuación (Mensurando) y = f(...)", 
            "(V1/I - R_A) * (1 + alpha * (20 - t))",
            help="Usa sintaxis de Python. (Recuerda que la fórmula del PDF tenía un error y el '-1' no iba)"
        )

# Actualizar st.session_state con las variables (preservando el orden)
current_vars = {}
for var in var_names:
    if var in st.session_state.variables:
        current_vars[var] = st.session_state.variables[var]
    else:
        current_vars[var] = {
            "data_str": "", "stats_a": calculate_type_a_stats(np.array([])),
            "b_type": "Ninguna", "b_params": {}, "best_estimate": 0.0,
            "u_a": 0.0, "u_b": 0.0, "u_total": 0.0, "b_calc_str": ""
        }
st.session_state.variables = current_vars

st.markdown("---")
st.header("2. Fuentes de Incertidumbre (Variables de Entrada)")
st.info("Configura cada variable. Las variables aparecen en el orden en que las escribiste.")

# --- PASO 2: Configuración de Variables ---
for var in var_names:
    with st.expander(f"**Variable: {var}**", expanded=True):
        v = st.session_state.variables[var]
        
        st.subheader(f"Mejor Estimación (Punto de Evaluación) para {var}")
        
        default_mean = v.get("best_estimate", 0.0)
        if v["stats_a"]['mean'] is not None:
            default_mean = v["stats_a"]['mean']
            help_text = "Valor tomado de la media de Tipo A. Puedes sobrescribirlo."
        else:
            help_text = "Introduce el valor de referencia (ej. de catálogo)."
        
        v["best_estimate"] = st.number_input(
            "Valor central",
            value=float(default_mean),
            format="%.8g",
            key=f"{var}_best_est",
            help=help_text
        )
        
        tab1, tab2 = st.tabs(["📊 Incertidumbre Tipo A (Estadística)", "🤖 Incertidumbre Tipo B (Instrumento)"])
        
        with tab1:
            v["data_str"] = st.text_area(
                f"Pega tus 'n' mediciones para {var} (separadas por coma, espacio o salto de línea)",
                value=v["data_str"], key=f"{var}_data", height=125
            )
            data_array = parse_measurements(v["data_str"])
            v["stats_a"] = calculate_type_a_stats(data_array)
            v["u_a"] = v["stats_a"]['u_a']
            
            if v["stats_a"]['mean'] is not None:
                if st.button(f"Usar Media ({v['stats_a']['mean']:.7g}) como Mejor Estimación", key=f"{var}_use_mean"):
                    v["best_estimate"] = v["stats_a"]['mean']
                    st.rerun() 
            
            st.subheader(f"Estadísticos para {var}")
            cols_stats = st.columns(3)
            cols_stats[0].metric("N° de Muestras (n)", f"{v['stats_a']['n']}")
            cols_stats[1].metric("Grados de Libertad (v)", f"{v['stats_a']['v']}")
            cols_stats[2].metric("Media (x̄)", f"{v['stats_a']['mean']:.7g}" if v['stats_a']['mean'] is not None else "N/A")
            
            cols_stats2 = st.columns(3)
            cols_stats2[0].metric("Varianza (s²)", f"{v['stats_a']['var']:.6g}")
            cols_stats2[1].metric("Desv. Estándar (s)", f"{v['stats_a']['std']:.6g}")
            cols_stats2[2].metric("Incertidumbre Tipo A (u_A = s/√n)", f"{v['stats_a']['u_a']:.6g}", delta_color="off")

        with tab2:
            # --- MEJORA DE INTUICIÓN (v6.0) ---
            b_options = [
                "Ninguna", 
                "Analógico (Clase, Rectangular)", 
                "Digital (Exactitud, Rectangular)", 
                "Especificación (Límites, Rectangular)", 
                "Resistencia (±1%, Rectangular)", 
                "Estándar Conocida (u, Normal)"
            ]
            b_captions = [
                "No se considera incertidumbre Tipo B para esta variable.",
                "Para instrumentos con 'Clase de Exactitud' (ej: 1.5) y un 'Alcance' (ej: 50 mA).",
                "Para instrumentos con exactitud de '% de Lectura + cuentas' (ej: ±(0.5% + 2d)).",
                "Para valores con una tolerancia o límite de error simple (ej: ±0.1 V).",
                "Uso específico para resistencias con tolerancia porcentual (ej: ±1%, ±5%).",
                "Para valores de un certificado de calibración que ya da la 'u' (con k=1)."
            ]
            # ---------------------------------

            b_index = 0
            if v["b_type"] in b_options:
                b_index = b_options.index(v["b_type"])
            
            v["b_type"] = st.radio(
                f"Selecciona la fuente de Incertidumbre B para {var}", 
                options=b_options, 
                captions=b_captions, 
                index=b_index, 
                key=f"{var}_b_type"
            )
            
            v["b_params"]["lectura"] = v["best_estimate"] 
            
            if v["b_type"] == "Analógico (Clase, Rectangular)":
                c1, c2 = st.columns(2)
                v["b_params"]["clase"] = c1.number_input("Clase de Exactitud (ej: 1.5)", min_value=0.0, value=v["b_params"].get("clase", 1.5), format="%.2f", key=f"{var}_b_clase")
                v["b_params"]["alcance"] = c2.number_input("Alcance (ej: 50)", min_value=0.0, value=v["b_params"].get("alcance", 1.0), format="%.4g", key=f"{var}_b_alcance_ana")
            
            elif v["b_type"] == "Digital (Exactitud, Rectangular)":
                c1, c2 = st.columns(2)
                v["b_params"]["porcentaje"] = c1.number_input("% de Lectura (ej: 0.5)", min_value=0.0, value=v["b_params"].get("porcentaje", 0.25), format="%.3f", key=f"{var}_b_porc")
                v["b_params"]["cuentas"] = c2.number_input("N° de Dígitos/Cuentas (ej: 2)", min_value=0, value=v["b_params"].get("cuentas", 2), step=1, key=f"{var}_b_cuentas")
                c3, c4 = st.columns(2)
                v["b_params"]["alcance"] = c3.number_input("Alcance (ej: 20)", min_value=0.0, value=v["b_params"].get("alcance", 20.0), format="%.4g", key=f"{var}_b_alcance_dig")
                v["b_params"]["max_cuentas"] = c4.number_input("Cuentas Max. del Rango (ej: 1999)", min_value=1, value=v["b_params"].get("max_cuentas", 1999), step=1, key=f"{var}_b_max_cuentas")

            elif v["b_type"] == "Especificación (Límites, Rectangular)":
                v["b_params"]["limite_error"] = st.number_input("Límite de Error (± valor)", min_value=0.0, value=v["b_params"].get("limite_error", 0.1), format="%.4g", key=f"{var}_b_limite")

            elif v["b_type"] == "Resistencia (±1%, Rectangular)":
                v["b_params"]["resistencia"] = st.number_input("Valor de Resistencia (Ω)", min_value=0.0, value=v["b_params"].get("resistencia", v["best_estimate"]), format="%.4g", key=f"{var}_b_res")

            elif v["b_type"] == "Estándar Conocida (u, Normal)":
                v["b_params"]["u_std"] = st.number_input("Incertidumbre Estándar (u)", min_value=0.0, value=v["b_params"].get("u_std", 0.01), format="%.4g", key=f"{var}_b_ustd")
            
            v["u_b"], v["b_calc_str"] = calculate_type_b(v["b_type"], v["b_params"])
            st.metric(f"Incertidumbre Tipo B (u_B)", f"{v['u_b']:.6g}", delta_color="off")
            if v["b_calc_str"]:
                st.caption(v["b_calc_str"])
        
        st.divider()
        v["u_total"] = math.sqrt(v["u_a"]**2 + v["u_b"]**2)
        col_res1, col_res2, col_res3 = st.columns(3)
        col_res1.metric(f"u_A({var})", f"{v['u_a']:.5g}", help="Incertidumbre Tipo A (Estadística)")
        col_res2.metric(f"u_B({var})", f"{v['u_b']:.5g}", help="Incertidumbre Tipo B (Instrumento)")
        col_res3.metric(f"Incertidumbre Estándar u({var})", f"{v['u_total']:.5g}", help="u = sqrt(u_A² + u_B²)")

st.markdown("---")

# --- PASO 3: Calcular y Mostrar Resultados ---
st.header("3. Resultados del Cálculo")

if 'v_b_eff' not in st.session_state:
    st.session_state.v_b_eff = 8 # Inicializar

if st.button("🚀 Calcular Incertidumbre Combinada", type="primary", use_container_width=True):
    f, error = create_function(var_names, func_expr)
    if error:
        st.error(error); st.stop()
        
    point_dict = {}; sensitivities = {}; budget_data = []
    u_c_squared = 0.0
    veff_denominator = 0.0
    v_b_eff = st.session_state.v_b_eff 
    
    errors_found = False
    for var in var_names:
        v = st.session_state.variables.get(var)
        if v is None:
            st.error(f"Error: La variable '{var}' se definió pero no se encontraron sus datos. Intente reiniciar la app.")
            st.stop()
        point_dict[var] = v["best_estimate"]
    
    for var in var_names:
        c_i, error = calculate_sensitivity(f, point_dict, var) 
        if error:
            st.error(f"Error al calcular sensibilidad de {var}: {error}"); errors_found = True
        sensitivities[var] = c_i
    if errors_found: st.stop()

    for var in var_names:
        v = st.session_state.variables[var]
        c_i = sensitivities.get(var, 0.0)
        
        v["b_params"]["lectura"] = v["best_estimate"]
        v["u_b"], v["b_calc_str"] = calculate_type_b(v["b_type"], v["b_params"])
        v["u_total"] = math.sqrt(v["u_a"]**2 + v["u_b"]**2)
        
        contribution_A_var = (c_i * v["u_a"])**2
        contribution_B_var = (c_i * v["u_b"])**2
        total_contribution_var = contribution_A_var + contribution_B_var
        
        u_c_squared += total_contribution_var
        
        v_a = v["stats_a"]["v"]
        if v_a > 0:
            veff_denominator += (contribution_A_var**2) / v_a
        
        if v_b_eff > 0 and contribution_B_var > 0:
            veff_denominator += (contribution_B_var**2) / v_b_eff
            
        budget_data.append({
            "var": var, "x_i": v["best_estimate"], "c_i": c_i,
            "u_a": v["u_a"], "v_a": v_a,
            "u_b": v["u_b"], "v_b": v_b_eff,
            "u_total": v["u_total"],
            "contrib_total_var": total_contribution_var
        })

    u_c = math.sqrt(u_c_squared)
    veff_numerator = u_c_squared**2
    
    if veff_denominator > 0:
        v_eff = math.floor(veff_numerator / veff_denominator)
        v_label = f"{v_eff}"
    else:
        v_eff = float('inf'); v_label = "∞"
        
    k = get_t_factor(v_eff)
    U_final = k * u_c

    try:
        y_best = f(**point_dict)
    except Exception as e:
        st.error(f"Error al calcular el resultado final y = f(...): {e}"); st.stop()

    # --- 6. Mostrar Resultados ---
    st.subheader("🎉 Resultado Final")
    col_r1, col_r2 = st.columns(2)
    col_r1.metric("Mejor Estimación del Resultado (y)", f"{y_best:.8g}")
    col_r2.metric("Incertidumbre Combinada (u_c(y))", f"{u_c:.5g}")
    st.divider()
    col_k1, col_k2, col_k3 = st.columns(3)
    col_k1.metric("Grados de Libertad Efectivos (v_eff)", v_label, help="Calculado con la fórmula de Welch-Satterthwaite.")
    col_k2.metric("Factor Cobertura (k)", f"{k:.4g}", help=f"Valor de t-Student para v={v_label} y P=95.45%")
    col_k3.metric("Incertidumbre Expandida (U = k·u_c)", f"{U_final:.5g}")
    
    st.success(f"**Resultado: y = {y_best:.8g}  ±  {U_final:.5g}  (k={k:.4g})**")
    
    # --- 7. Mostrar Desglose (Colapsable) ---
    st.subheader("📊 Desglose de Contribuciones (Opcional)")
    st.write("Expande cada variable para ver los detalles de su contribución.")
    
    for item in budget_data:
        percent_contrib = 0.0
        if u_c_squared > 0:
            percent_contrib = (item['contrib_total_var'] / u_c_squared) * 100
            
        with st.expander(f"**Variable: `{item['var']}` (Contribuye {percent_contrib:.2f}%)**"):
            c1, c2 = st.columns([1, 2])
            with c1:
                st.metric(f"Mejor Estimación (x̄_i)", f"{item['x_i']:.7g}")
                st.metric(f"Coef. Sensibilidad (c_i)", f"{item['c_i']:.4e}")
            with c2:
                st.metric(f"Contribución a la Varianza (c_i·u_i)²", f"{item['contrib_total_var']:.4e}")
                st.metric(f"Contribución Porcentual (%)", f"{percent_contrib:.2f}%")
            
            st.markdown("**Componentes de Incertidumbre:**")
            c_a1, c_a2, c_b1, c_b2 = st.columns(4)
            c_a1.metric(f"u_A(x_i)", f"{item['u_a']:.4e}")
            c_a2.metric(f"v_A", f"{item['v_a'] if item['v_a'] > 0 else 'N/A'}")
            c_b1.metric(f"u_B(x_i)", f"{item['u_b']:.4e}")
            c_b2.metric(f"v_B", f"{item['v_b'] if item['u_b'] > 0 else 'N/A'}")