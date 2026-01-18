# llm.py
import os


from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set")

# LLMs
answer_llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    groq_api_key=GROQ_API_KEY,
    temperature=0.2,
    max_tokens=180,
)

translation_llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    groq_api_key=GROQ_API_KEY,
    temperature=0.1,
    max_tokens=180,
)

# Prompts
answer_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an expert assistant specialized in extracting answers from insurance policy documents.

STRICT RULES:
- Answer ONLY in English
- Use ONLY the information from CONTEXT
- Write 30–40 words
- Use formal policy language
- If context does not contain the answer, respond exactly:
  No relevant policy information found.

CONTEXT:
{context}

QUESTION:
{question}
"""
)

answer_chain = answer_prompt | answer_llm

def get_llm_answer(question: str, context: str) -> str:
    return answer_chain.invoke(
        {"context": context, "question": question}
    ).content.strip()


lang_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
Detect the language of this text.
Respond with one word only:
English, Hindi, or Marathi.

TEXT:
{text}
"""
)

lang_chain = lang_prompt | translation_llm

def detect_language(text: str) -> str:
    lang = lang_chain.invoke({"text": text}).content.lower()
    if "hindi" in lang:
        return "Hindi"
    if "marathi" in lang:
        return "Marathi"
    return "English"


to_eng_prompt = PromptTemplate(
    input_variables=["text"],
    template="Translate the following text into English only:\n{text}"
)

to_eng_chain = to_eng_prompt | translation_llm

def translate_to_english(text: str) -> str:
    return to_eng_chain.invoke({"text": text}).content.strip()


to_user_prompt = PromptTemplate(
    input_variables=["text", "lang"],
    template="Translate the following English text into {lang}:\n{text}"
)

to_user_chain = to_user_prompt | translation_llm

def translate_answer(text: str, target_language: str) -> str:
    english = translate_to_english(text)
    if target_language == "English":
        return english

    translated = to_user_chain.invoke(
        {"text": english, "lang": target_language}
    ).content.strip()

    return f"""
🌐 **{target_language} Translation:**
{translated}

🇬🇧 **English Version:**
{english}
""".strip()
# llm.py
import os
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq


# ❌ DO NOT create LLM at import time
_answer_llm = None
_translation_llm = None


def get_answer_llm():
    global _answer_llm
    if _answer_llm is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set")

        _answer_llm = ChatGroq(
            model_name="llama-3.1-8b-instant",
            groq_api_key=api_key,
            temperature=0.2,
            max_tokens=180,
        )
    return _answer_llm


def get_translation_llm():
    global _translation_llm
    if _translation_llm is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set")

        _translation_llm = ChatGroq(
            model_name="llama-3.1-8b-instant",
            groq_api_key=api_key,
            temperature=0.1,
            max_tokens=180,
        )
    return _translation_llm


answer_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an expert assistant specialized in extracting answers from insurance policy documents.

STRICT RULES:
- Answer ONLY in English
- Use ONLY the information from CONTEXT
- Write 30–40 words
- Use formal policy language
- If context does not contain the answer, respond exactly:
  No relevant policy information found.

CONTEXT:
{context}

QUESTION:
{question}
"""
)


def get_llm_answer(question: str, context: str) -> str:
    chain = answer_prompt | get_answer_llm()
    return chain.invoke(
        {"context": context, "question": question}
    ).content.strip()


lang_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
Detect the language of this text.
Respond with one word only:
English, Hindi, or Marathi.

TEXT:
{text}
"""
)


def detect_language(text: str) -> str:
    chain = lang_prompt | get_translation_llm()
    lang = chain.invoke({"text": text}).content.lower()
    if "hindi" in lang:
        return "Hindi"
    if "marathi" in lang:
        return "Marathi"
    return "English"


def translate_to_english(text: str) -> str:
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Translate the following text into English only:\n{text}"
    )
    chain = prompt | get_translation_llm()
    return chain.invoke({"text": text}).content.strip()


def translate_answer(text: str, target_language: str) -> str:
    if target_language == "English":
        return text

    prompt = PromptTemplate(
        input_variables=["text", "lang"],
        template="Translate the following English text into {lang}:\n{text}"
    )

    chain = prompt | get_translation_llm()
    translated = chain.invoke(
        {"text": text, "lang": target_language}
    ).content.strip()

    return f"""
🌐 **{target_language} Translation:**
{translated}

🇬🇧 **English Version:**
{text}
""".strip()
