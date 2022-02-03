import streamlit as st

from multipage import MultiPage
from pages import database, ideate, design, dashboard, analyze #, evaluate, dashboard

app = MultiPage()
apptitle = 'AI ART FARM'
st.set_page_config(page_title=apptitle, page_icon=":art:")
st.sidebar.markdown("## Select a page")

app.add_page("PROMPT", ideate.app)
app.add_page("DESIGN", design.app)
app.add_page("DASHBOARD", dashboard.app)
# app.add_page("EVALUATE", evaluate.app)
# app.add_page("ANALYZE", analyze.app)
app.add_page("DATABASE", database.app)

app.run()