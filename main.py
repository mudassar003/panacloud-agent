import dotenv
import nest_asyncio
nest_asyncio.apply()
from pydantic import BaseModel
from agents import (
    Agent,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
    RunContextWrapper,
    Runner,
    TResponseInputItem,
    input_guardrail,
    output_guardrail,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    RunConfig
)

def main():
    gemini_api_key = dotenv.get_key(".env", "GEMINI_API_KEY")

    external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",)

    model = OpenAIChatCompletionsModel(
        model="gemini-2.0-flash",
        openai_client=external_client
    )

    config = RunConfig(
        model=model,
        model_provider=external_client,
        tracing_disabled=True
    )

    agent = Agent(
        name="Gemini Agent",
        instructions="An agent that uses Gemini 2.0 Flash model for text generation.",
        run_config=config,
    )

    result = Runner.run(agent, "What is the capital of France?")
    print("Result:", result)



if __name__ == "__main__":
    main()
