import streamlit as st
import os
import openai
from dotenv import load_dotenv
import json

from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

from utils import load_career_paths, embed_text, extract_user_profile, match_career

# Load API Key from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")

if not openai_key:
    st.error("OPENAI_API_KEY not found. Please add it to your .env file or environment.")
    st.stop()

# Set up Streamlit app
st.set_page_config(page_title="Career Path Recommender", layout="centered")
st.title("🔍 Career Path Recommender")
st.write("Paste a conversation or describe your interests to get a career suggestion!")

# Load predefined career paths
career_data = load_career_paths()

st.write("✅ App loaded before LLM.")
# Set up OpenAI LLM (✅ pass the key)
llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model_name="gpt-3.5-turbo",
    temperature=0.5
)

# Prompt to extract user traits
extract_prompt = PromptTemplate.from_template(
    """
    Extract the following from the user's conversation:
    - interests
    - skills
    - personality traits
    - career goals

    Return JSON with keys: interests, skills, personality_traits, career_goals.

    Conversation:
    ---
    {conversation}
    ---
    """
)
extract_chain = LLMChain(llm=llm, prompt=extract_prompt)

# Prompt to generate explanation for career recommendation
explanation_prompt = PromptTemplate.from_template(
    """
    The user profile is: {user_profile}.
    Recommend why the career category {career_category} with roles like {careers} suits the user.

    Provide a brief, clear explanation (2–3 sentences).
    """
)
explanation_chain = LLMChain(llm=llm, prompt=explanation_prompt)

# Input text area
conversation = st.text_area("🗣️ Your Conversation / Description:", height=200)

# Button to generate recommendation
if st.button("Generate Career Path"):
    if not conversation.strip():
        st.warning("Please enter some input.")
    else:
        with st.spinner("Analyzing..."):
            # Step 1: Extract traits
            result_json = extract_chain.run(conversation=conversation)
            try:
                user_data = json.loads(result_json)
            except Exception as e:
                st.error("⚠️ Could not parse model response. Try rephrasing your input.")
                st.text(result_json)
                st.stop()

            # Step 2: Build embedding for user profile
            profile_text = extract_user_profile(user_data)
            user_vector = embed_text(profile_text)

            # Step 3: Match best-fit career path
            best_career, score = match_career(user_vector, career_data)

            # Step 4: Generate explanation
            explanation = explanation_chain.run(
                user_profile=profile_text,
                career_category=best_career["category"],
                careers=", ".join(best_career["careers"])
            )

            # Final Output
            st.success(f"🎯 Recommended Career Path: **{best_career['category']}**")
            st.markdown(f"**Careers:** {', '.join(best_career['careers'])}")
            st.markdown(f"**Why this fits you:** {explanation}")
