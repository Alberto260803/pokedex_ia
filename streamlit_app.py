import os
import requests
import streamlit as st
from smolagents import CodeAgent, tool, LiteLLMModel

# ---------------------------------------------------------
# Configuración del modelo (Mistral vía LiteLLM)
# ---------------------------------------------------------
# En Streamlit Community Cloud, la key se configura en:
# App settings -> Secrets, como: MISTRAL_API_KEY = "..."
MISTRAL_API_KEY = st.secrets.get("MISTRAL_API_KEY", os.getenv("MISTRAL_API_KEY"))

POKEAPI_URL = "https://pokeapi.co/api/v2"

TYPE_CHART = {
    "fire": {"water": 0.5, "grass": 2, "ice": 2, "bug": 2, "rock": 0.5, "dragon": 0.5, "fire": 0.5},
    "water": {"fire": 2, "grass": 0.5, "ground": 2, "rock": 2, "dragon": 0.5, "water": 0.5},
    "grass": {"water": 2, "fire": 0.5, "ground": 2, "rock": 2, "flying": 0.5, "bug": 0.5, "grass": 0.5},
    "electric": {"water": 2, "flying": 2, "ground": 0, "electric": 0.5, "grass": 0.5, "dragon": 0.5},
    "ice": {"grass": 2, "ground": 2, "flying": 2, "dragon": 2, "fire": 0.5, "water": 0.5, "ice": 0.5},
    "fighting": {"normal": 2, "ice": 2, "rock": 2, "dark": 2, "steel": 2, "flying": 0.5, "psychic": 0.5, "bug": 0.5, "fairy": 0.5},
    "poison": {"grass": 2, "fairy": 2, "poison": 0.5, "ground": 0.5, "rock": 0.5, "ghost": 0.5, "steel": 0},
    "ground": {"fire": 2, "electric": 2, "poison": 2, "rock": 2, "steel": 2, "grass": 0.5, "bug": 0.5, "flying": 0},
    "flying": {"grass": 2, "fighting": 2, "bug": 2, "electric": 0.5, "rock": 0.5, "steel": 0.5},
    "psychic": {"fighting": 2, "poison": 2, "psychic": 0.5, "steel": 0.5, "dark": 0},
    "bug": {"grass": 2, "psychic": 2, "dark": 2, "fire": 0.5, "fighting": 0.5, "poison": 0.5, "flying": 0.5, "ghost": 0.5, "steel": 0.5, "fairy": 0.5},
    "rock": {"fire": 2, "ice": 2, "flying": 2, "bug": 2, "fighting": 0.5, "ground": 0.5, "steel": 0.5},
    "ghost": {"psychic": 2, "ghost": 2, "dark": 0.5, "normal": 0},
    "dragon": {"dragon": 2, "steel": 0.5, "fairy": 0},
    "dark": {"psychic": 2, "ghost": 2, "fighting": 0.5, "dark": 0.5, "fairy": 0.5},
    "steel": {"ice": 2, "rock": 2, "fairy": 2, "fire": 0.5, "water": 0.5, "electric": 0.5, "steel": 0.5},
    "fairy": {"fighting": 2, "dragon": 2, "dark": 2, "fire": 0.5, "poison": 0.5, "steel": 0.5},
    "normal": {"rock": 0.5, "ghost": 0, "steel": 0.5},
}


def _fetch_pokemon(name: str) -> dict:
    resp = requests.get(f"{POKEAPI_URL}/pokemon/{name.lower().strip()}", timeout=10)
    resp.raise_for_status()
    return resp.json()


@tool
def get_pokemon_info(name: str) -> str:
    """
    Obtiene información de un Pokémon: tipos, estadísticas base y habilidades.

    Args:
        name: Nombre del Pokémon en inglés (ej. "pikachu", "charizard").
    """
    try:
        data = _fetch_pokemon(name)
    except requests.HTTPError:
        return f"No he encontrado ningún Pokémon llamado '{name}'. Revisa el nombre (en inglés)."

    tipos = [t["type"]["name"] for t in data["types"]]
    stats = {s["stat"]["name"]: s["base_stat"] for s in data["stats"]}
    habilidades = [a["ability"]["name"] for a in data["abilities"]]

    return (
        f"Pokémon: {data['name'].capitalize()} (#{data['id']})\n"
        f"Tipos: {', '.join(tipos)}\n"
        f"Altura: {data['height']/10} m | Peso: {data['weight']/10} kg\n"
        f"Estadísticas base: {stats}\n"
        f"Habilidades: {', '.join(habilidades)}"
    )


@tool
def compare_pokemon(name_a: str, name_b: str) -> str:
    """
    Compara dos Pokémon y estima quién tendría ventaja de tipo en un combate,
    usando la tabla oficial de efectividades.

    Args:
        name_a: Nombre del primer Pokémon (en inglés).
        name_b: Nombre del segundo Pokémon (en inglés).
    """
    try:
        a = _fetch_pokemon(name_a)
        b = _fetch_pokemon(name_b)
    except requests.HTTPError:
        return "No he podido encontrar uno de los dos Pokémon. Revisa los nombres (en inglés)."

    tipos_a = [t["type"]["name"] for t in a["types"]]
    tipos_b = [t["type"]["name"] for t in b["types"]]

    def multiplicador(atacante_tipos, defensor_tipos):
        mult = 1.0
        for at in atacante_tipos:
            chart = TYPE_CHART.get(at, {})
            for dt in defensor_tipos:
                mult *= chart.get(dt, 1)
        return mult

    mult_a_ataca_b = multiplicador(tipos_a, tipos_b)
    mult_b_ataca_a = multiplicador(tipos_b, tipos_a)

    stats_a = sum(s["base_stat"] for s in a["stats"])
    stats_b = sum(s["base_stat"] for s in b["stats"])

    return (
        f"{a['name'].capitalize()} ({', '.join(tipos_a)}, stats totales={stats_a}) vs "
        f"{b['name'].capitalize()} ({', '.join(tipos_b)}, stats totales={stats_b})\n"
        f"Efectividad de {a['name'].capitalize()} atacando a {b['name'].capitalize()}: x{mult_a_ataca_b}\n"
        f"Efectividad de {b['name'].capitalize()} atacando a {a['name'].capitalize()}: x{mult_b_ataca_a}"
    )


@st.cache_resource
def get_agent():
    model = LiteLLMModel(
        model_id="mistral/mistral-small-latest",
        api_key=MISTRAL_API_KEY,
    )
    return CodeAgent(
        tools=[get_pokemon_info, compare_pokemon],
        model=model,
        add_base_tools=False,
    )


SYSTEM_INTRO = (
    "Eres el Profesor Oak, un experto Pokémon. Usa las herramientas disponibles "
    "para consultar datos reales antes de responder. Da respuestas claras, con "
    "personalidad, y explica el porqué de tus recomendaciones de combate."
)

# ---------------------------------------------------------
# Interfaz Streamlit
# ---------------------------------------------------------
st.set_page_config(page_title="Pokédex con IA", page_icon="🔴")
st.title("🔴 Pokédex con IA — Profesor Oak")
st.caption(
    "Pregúntame sobre cualquier Pokémon (en inglés, ej. *pikachu*, *garchomp*, "
    "*charizard*) o pídeme que compare dos para saber quién gana un combate. "
    "Uso datos reales de la PokéAPI y razono con Mistral a través de smolagents."
)

if not MISTRAL_API_KEY:
    st.error(
        "Falta MISTRAL_API_KEY. Configúrala en Settings -> Secrets "
        "(Streamlit Cloud) o como variable de entorno en local."
    )
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

ejemplo_cols = st.columns(3)
ejemplos = [
    "Háblame de garchomp",
    "¿Quién gana, charizard o blastoise?",
    "Recomiéndame un pokémon de fuego para empezar",
]
pregunta_ejemplo = None
for col, ejemplo in zip(ejemplo_cols, ejemplos):
    if col.button(ejemplo):
        pregunta_ejemplo = ejemplo

user_input = st.chat_input("Escribe tu pregunta sobre Pokémon...") or pregunta_ejemplo

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Consultando la Pokédex..."):
            agent = get_agent()
            prompt = f"{SYSTEM_INTRO}\n\nPregunta del usuario: {user_input}"
            try:
                respuesta = agent.run(prompt)
            except Exception as e:
                respuesta = f"Vaya, algo ha fallado consultando la Pokédex: {e}"
        st.markdown(respuesta)

    st.session_state.messages.append({"role": "assistant", "content": respuesta})
