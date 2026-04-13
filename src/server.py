import argparse
import logging
import uvicorn

from dotenv import load_dotenv
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

from executor import Executor


def main():
    load_dotenv(override=True)
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Run the A2A agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9009, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="URL to advertise in the agent card")
    args = parser.parse_args()

    # Fill in your agent card
    # See: https://a2a-protocol.org/latest/tutorials/python/3-agent-skills-and-card/
    
    skill = AgentSkill(
        id="use_computer",
        name="Use Computer",
        description="Interprets OSWorld observations and returns desktop actions to complete the requested task.",
        tags=["osworld", "desktop", "automation", "multimodal"],
        examples=[
            "Open the settings app and switch the system theme to dark mode.",
            "Read the screenshot and suggest the next UI action to continue the task.",
        ],
    )

    agent_card = AgentCard(
        name="OSWorld Impl Agent",
        description="An OSWorld A2A agent that plans and executes desktop actions from text, screenshots, and structured observations.",
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version='1.0.0',
        default_input_modes=['text', 'file', 'data'],
        default_output_modes=['text', 'data'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill]
    )

    request_handler = DefaultRequestHandler(
        agent_executor=Executor(),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    uvicorn.run(server.build(), host=args.host, port=args.port)


if __name__ == '__main__':
    main()
