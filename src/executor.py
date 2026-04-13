import sys
import traceback
import logging
import os
import time
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    TaskState,
    UnsupportedOperationError,
    InvalidRequestError,
)
from a2a.utils.errors import ServerError
from a2a.utils import (
    new_agent_text_message,
    new_task,
)

from agent import Agent


logger = logging.getLogger("osworld_impl.executor")
MAX_AGENT_CONTEXTS = int(os.environ.get("MAX_AGENT_CONTEXTS", "200"))
AGENT_CONTEXT_TTL_SECONDS = int(os.environ.get("AGENT_CONTEXT_TTL_SECONDS", "3600"))


TERMINAL_STATES = {
    TaskState.completed,
    TaskState.canceled,
    TaskState.failed,
    TaskState.rejected
}


class Executor(AgentExecutor):
    def __init__(self):
        self.agents: dict[str, Agent] = {}
        self._last_seen: dict[str, float] = {}

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        msg = context.message
        if not msg:
            raise ServerError(error=InvalidRequestError(message="Missing message in request"))

        task = context.current_task
        if task and task.status.state in TERMINAL_STATES:
            raise ServerError(error=InvalidRequestError(message=f"Task {task.id} already processed (state: {task.status.state})"))

        if not task:
            task = new_task(msg)
            await event_queue.enqueue_event(task)

        self._prune_agents()
        context_id = task.context_id
        agent = self.agents.get(context_id)
        if not agent:
            agent = Agent()
            self.agents[context_id] = agent
        self._last_seen[context_id] = time.monotonic()

        updater = TaskUpdater(event_queue, task.id, context_id)

        await updater.start_work()
        try:
            await agent.run(msg, updater)
            try:
                await updater.complete()
            except RuntimeError:
                logger.debug("Task %s already reached terminal state", task.id)
        except Exception as e:
            logger.error("Task failed with agent error")
            traceback.print_exc(file=sys.stdout)
            await updater.failed(new_agent_text_message(f"Agent error: {e}", context_id=context_id, task_id=task.id))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise ServerError(error=UnsupportedOperationError())

    def _prune_agents(self) -> None:
        now = time.monotonic()
        expired = [
            context_id
            for context_id, seen_at in self._last_seen.items()
            if now - seen_at > AGENT_CONTEXT_TTL_SECONDS
        ]
        for context_id in expired:
            self.agents.pop(context_id, None)
            self._last_seen.pop(context_id, None)

        if len(self.agents) <= MAX_AGENT_CONTEXTS:
            return

        overflow = len(self.agents) - MAX_AGENT_CONTEXTS
        oldest = sorted(self._last_seen.items(), key=lambda item: item[1])[:overflow]
        for context_id, _ in oldest:
            self.agents.pop(context_id, None)
            self._last_seen.pop(context_id, None)
