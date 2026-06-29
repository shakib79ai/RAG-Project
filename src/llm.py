"""
LLM wrapper that supports two modes (set via LLM_MODE in .env):

  "api"   -> HF router (OpenAI-compatible) via router.huggingface.co/together.
             Fast to set up, no GPU required. Needs an HF token.
             api-inference.huggingface.co is deprecated; the router is used instead.

  "local" -> Downloads the model and runs it locally via transformers
             (HuggingFacePipeline). Needs a reasonably capable machine/GPU
             for anything beyond small models, but has no rate limits and
             works fully offline after the first download.
"""
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

from src import config


def get_llm():
    if config.LLM_MODE == "api":
        return ChatOpenAI(
            model=config.LLM_MODEL_API,
            openai_api_key=config.HF_TOKEN,
            openai_api_base=config.LLM_PROVIDER_URL,
            max_tokens=512,
            temperature=0.1,
        )

    elif config.LLM_MODE == "local":
        pipeline_llm = HuggingFacePipeline.from_model_id(
            model_id=config.LLM_MODEL_LOCAL,
            task="text-generation",
            pipeline_kwargs={
                "max_new_tokens": 512,
                "temperature": 0.1,
                "repetition_penalty": 1.1,
                "do_sample": True,
            },
        )
        return ChatHuggingFace(llm=pipeline_llm)

    else:
        raise ValueError(f"Unknown LLM_MODE: {config.LLM_MODE}. Use 'api' or 'local'.")
