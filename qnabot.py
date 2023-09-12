import streamlit as st
import base64
import os
import pandas as pd
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Create a checkbox to select the model
model_option = st.checkbox("Select Model: GPT 3.5 Turbo")
if model_option:
    model = "gpt-3.5-turbo"
else:
    model = "gpt-4"

# Initialize chat model
chat_llm = ChatOpenAI(model_name=model, temperature=0.0)

def generator(question, answer):
    title_template = """
                    for the "{question}", check the "{answer}" if it is correct, then explain why in short..
                    """ 

    prompt = ChatPromptTemplate.from_template(template=title_template)
    messages = prompt.format_messages(question=question, answer=answer)
    response = chat_llm(messages)
    return str(response.content)

def main():
    st.title("👨‍💻 QnA with explanation")

    # File Upload
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df)
        # Add an "Explanation" column
        df["Explanation"] = ""

        # Process each row
        for index, row in df.iterrows():
            question = row["Question"]
            answer = row["Answer"]
            explanation = generator(question, answer)
            df.at[index, "Explanation"] = explanation

        # Display the result DataFrame
        st.write("Result:")
        result_filename = f"{model.lower().replace(' ', '')}_result.df"
        df.to_csv(result_filename)
        st.write(df)

        # Create a download link for the result CSV
        csv_file = df.to_csv(index=False)
        b64 = base64.b64encode(csv_file.encode()).decode()
        st.markdown(f'### Download Result CSV ###')
        href = f'<a href="data:file/csv;base64,{b64}" download="{result_filename}">Click here to download the result CSV file</a>'
        st.markdown(href, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
