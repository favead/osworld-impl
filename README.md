# OSWorld Impl Agent

An [A2A (Agent-to-Agent)](https://a2a-protocol.org/latest/) computer-use agent for OSWorld-style desktop tasks. The agent uses a split control loop:

- executor / ReAct model for concrete desktop actions,
- planner / verifier model for trajectory summarization and replanning.

## Project Structure

```
src/
├─ server.py      # Server setup and agent card configuration
├─ executor.py    # A2A request handling
├─ agent.py       # Planner/executor computer-use agent
└─ messenger.py   # A2A messaging utilities
tests/
└─ test_agent.py  # Agent tests
Dockerfile            # Docker configuration
pyproject.toml        # Python dependencies
amber-manifest.json5  # Amber manifest
.github/
└─ workflows/
   └─ test-and-publish.yml # CI workflow
```

## Getting Started

1. **Provide model credentials** - set the required repository secrets or local environment variables.
2. **Run the server** - start the A2A endpoint from [`src/server.py`](src/server.py).
3. **Submit with Amber** - use [`amber-manifest.json5`](amber-manifest.json5) to provide deployer config.

## Running Locally

```bash
# Install dependencies
uv sync

# Run the server
uv run src/server.py
```

### Required model configuration

The agent reads deployer config from `amber-manifest.json5`. The manifest exposes these config keys:

- `openai_api_key` - secret, used for OpenAI-compatible providers
- `openai_base_url` - OpenAI-compatible API base URL
- `executor_model` - desktop action model
- `planner_model` - planning / verification model
- `max_tokens`
- `top_p`
- `temperature`
- `max_trajectory_length`
- `planner_review_interval`
- `trajectory_summary_window`
- `a11y_tree_max_tokens`
- `model_timeout_seconds`
- `model_max_retries`
- `model_retry_backoff_seconds`
- `max_agent_contexts`
- `agent_context_ttl_seconds`
- `qwen_api_backend`
- `qwen_enable_thinking`
- `qwen_thinking_budget`

Inside the container, these become environment variables such as `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `EXECUTOR_MODEL`, and `PLANNER_MODEL`.

### OpenRouter example

This repo is configured to work with OpenAI-compatible endpoints, including OpenRouter.

Example local setup:

```bash
export OPENAI_API_KEY="<your-openrouter-key>"
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
export EXECUTOR_MODEL="qwen/qwen3.5-plus-02-15"
export PLANNER_MODEL="google/gemini-3.1-flash-lite-preview"
```

## Running with Docker

```bash
# Build the image
docker build -t my-agent .

# Run the container
docker run -p 9009:9009 my-agent
```

## Testing

Run A2A conformance tests against your agent.

```bash
# Install test dependencies
uv sync --extra test

# Start your agent (uv or docker; see above)

# Run tests against your running agent URL
uv run pytest --agent-url http://localhost:9009
```

## Publishing

The repository includes a GitHub Actions workflow that automatically builds, tests, and publishes a Docker image of your agent to GitHub Container Registry.

If your agent needs API keys or other secrets, add them in Settings → Secrets and variables → Actions → Repository secrets. They'll be available as environment variables during CI tests.

For this repo, the minimum recommended repository secrets for submission are:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- optionally `EXECUTOR_MODEL`
- optionally `PLANNER_MODEL`

If you rely on the code defaults, only the first two are required.

- **Push to `main`** → publishes `latest` tag:
```
ghcr.io/<your-username>/<your-repo-name>:latest
```

- **Create a git tag** (e.g. `git tag v1.0.0 && git push origin v1.0.0`) → publishes version tags:
```
ghcr.io/<your-username>/<your-repo-name>:1.0.0
ghcr.io/<your-username>/<your-repo-name>:1
```

Once the workflow completes, find your Docker image in the Packages section (right sidebar of your repository). Configure the package visibility in package settings.

> **Note:** Organization repositories may need package write permissions enabled manually (Settings → Actions → General). Version tags must follow [semantic versioning](https://semver.org/) (e.g., `v1.0.0`).
