# Time & Weather Agent

This Python project lets you interactively ask for the current weather and local time in any city using natural language. It combines OpenWeatherMap, TimezoneFinder, pytz, and Azure OpenAI agent for smart query handling.

## Features

- Ask for weather, time, or both for any city
- Natural language interface (e.g., "What's the weather in Paris?", "Current time in Tokyo")
- Secure API key management via environment variables
- Graceful error handling and clear messages

## Setup

1. **Clone the repository** and navigate to the project directory.

2. **Create a virtual environment** (recommended):
   ```
   python -m venv venv
   venv\Scripts\activate  # On Windows
   source venv/bin/activate  # On macOS/Linux
   ```

3. **Install dependencies**:
   ```
   pip install requests python-dotenv timezonefinder pytz
   ```

4. **Create a `.env` file** in the project directory with your API keys:
   ```
   WEATHER_API_KEY=your_openweathermap_api_key
   AZURE_AGENT_ENDPOINT=https://your-azure-agent-endpoint
   AZURE_AGENT_KEY=your_azure_agent_key
   ```

## Usage

Run the script:
```
python Time&WeatherAgent.py
```

You will be prompted to enter your query. Example queries:
- `Weather in London`
- `Current time in New York`
- `Weather and time in Tokyo`
- `exit` to quit

## How It Works

- The agent sends your query to Azure OpenAI, which replies with either a JSON action or a plain text answer.
- If the agent returns a JSON object (with `action` and `city`), the script calls the appropriate function(s) to fetch live data.
- If the agent returns plain text, the script tries to extract the city and intent from your query and fetches the data accordingly.

## Error Handling

- If the APIs fail or the city is not found, you will see a clear error message.
- If the agent response is not in the expected format, the script falls back to keyword and city extraction.

## Customization

- You can improve city extraction or keyword detection in the `agent()` function.
- To make the agent reliably control function calls, update your Azure agent's system prompt to always reply in strict JSON format.

## License

MIT
