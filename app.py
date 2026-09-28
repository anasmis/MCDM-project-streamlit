import streamlit as st

from ui.pages import problem_page, ranking_page, weighting_page
from ui.state import initialize, navigate
from ui.style import apply_style

st.set_page_config(
    page_title="Arbitrage · Décision multicritère",
    page_icon="◫",
    layout="wide",
    initial_sidebar_state="expanded",
)
initialize()
apply_style()

with st.sidebar:
    st.markdown(
        '<div class="brand"><span class="brand-mark">a</span>arbitrage</div>'
        '<div class="brand-sub">Atelier de décision multicritère</div>',
        unsafe_allow_html=True,
    )
    st.caption("VOTRE ÉTUDE")
    for index, label in enumerate(
        ["01   Définir le problème", "02   Pondérer les critères", "03   Classer les alternatives"]
    ):
        disabled = (index > 0 and st.session_state.problem is None) or (
            index > 1 and st.session_state.weighting is None
        )
        st.button(
            label,
            key=f"nav_{index}",
            width="stretch",
            disabled=disabled,
            type="primary" if st.session_state.step == index else "secondary",
            on_click=navigate,
            args=(index,),
        )
    st.divider()
    problem = st.session_state.problem
    if problem is not None:
        st.markdown(f"**{problem.name}**")
        st.caption(f"{len(problem.alternatives)} alternatives · {len(problem.criteria)} critères")
        if st.session_state.weighting is not None:
            st.caption(f"Pondération validée · {st.session_state.weighting.method}")
    else:
        st.caption("Commencez par nommer votre étude et renseigner ses performances.")
    st.markdown(
        '<div class="footer">Des critères explicites.<br>Une décision argumentée.</div>',
        unsafe_allow_html=True,
    )

if st.session_state.step == 0:
    problem_page()
elif st.session_state.step == 1:
    weighting_page()
else:
    ranking_page()
