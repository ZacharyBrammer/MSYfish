import re

import streamlit as st

from translate import Translator

REPO_URL = "https://github.com/ZacharyBrammer/MSYfish"

# Everything in the README after this marker is developer documentation
DEV_MARKER = "<!-- developer-docs"


# method to display the user manual (the user-facing part of the README)
def manual():
    translator = Translator(st.query_params.get("lang", "en"))
    t = translator.translate

    with open("README.md", "r", encoding="utf-8") as f:
        readme = f.read()

    # cut off the developer documentation
    readme = readme.split(DEV_MARKER)[0]

    # point relative file links at GitHub, since the repo files aren't served by the site
    readme = re.sub(
        r"\]\((?!https?://|#)([^)]+)\)",
        rf"]({REPO_URL}/blob/main/\1)",
        readme
    )

    st.markdown(readme, unsafe_allow_html=True)
    st.markdown(t("manual", "full_readme", link=REPO_URL))
