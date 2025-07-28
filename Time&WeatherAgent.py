# Time&WeatherAgent.py 

# Interactive agent for querying weather and local time in any city. 
# - Uses OpenWeatherMap API for weather 
# - Uses TimezoneFinder and pytz for local time 
# - Uses Azure OpenAI agent to interpret queries 

# Environment variables required: 

# - WEATHER_API_KEY: OpenWeatherMap API key 
# - AZURE_AGENT_ENDPOINT: Azure OpenAI agent endpoint 
# - AZURE_AGENT_KEY: Azure OpenAI agent key 


# Run: python Time&WeatherAgent.py 

##################################

import requests
from dotenv import load_dotenv
import os

from timezonefinder import TimezoneFinder
from datetime import datetime
import pytz
load_dotenv()

# Function to call Azure AI agent
def call_azure_agent(user_query):

#Sends a user query to the Azure OpenAI agent and returns the agent's response. 

# Args: 
# user_query (str): The user's question or command in natural language. 

# Returns: 
#dict or None: The parsed JSON response from the agent if successful, otherwise None. 

#Environment Variables: 

#AZURE_AGENT_ENDPOINT: The endpoint URL for the Azure OpenAI agent. 

#AZURE_AGENT_KEY: The API key for authenticating with the Azure OpenAI agent. 

#Raises: 

#Prints error details if the request fails.  


    endpoint = os.getenv('AZURE_AGENT_ENDPOINT')
    api_key = os.getenv('AZURE_AGENT_KEY')
    if not endpoint or not api_key:
        print("Azure AI agent endpoint or key not set in .env file.")
        return None
    headers = {
        'Content-Type': 'application/json',
        'api-key': api_key
    }
    data = {
        "messages": [
            {"role": "user", "content": user_query}
        ]
        # Add "model": "gpt-35-turbo" here if your endpoint requires it
    }
    try:
        response = requests.post(endpoint, headers=headers, json=data, timeout=20)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Azure agent call failed: {e}")
        try:
            print(response.text)
        except Exception:
            print("No response received from agent.")
        return None


# Function to get weather data only
def get_weather(city):
# Fetches current weather data for a given city using the OpenWeatherMap API. 

#Args: 
#city (str): The name of the city to fetch weather for. 

#Returns: 

#tuple: (temperature (float), weather_description (str), latitude (float), longitude (float)) 

#If the request fails, returns (None, None, None, None). 

#Environment Variables: 

#WEATHER_API_KEY: The API key for OpenWeatherMap. 

#Raises: 

#Prints error details if the request fails.


    weather_url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={os.getenv('WEATHER_API_KEY')}&units=metric"
    try:
        weather_response = requests.get(weather_url, timeout=10)
        weather_response.raise_for_status()
        weather_data = weather_response.json()
        temperature = weather_data['main']['temp']
        weather_description = weather_data['weather'][0]['description']
        lat = weather_data.get('coord', {}).get('lat')
        lon = weather_data.get('coord', {}).get('lon')
        return temperature, weather_description, lat, lon
    except Exception as e:
        print("Weather API error details:")
        try:
            print(weather_response.text)
        except Exception:
            print("No response received.")
        print(f"Exception: {e}")
        return None, None, None, None

# Function to get time only (requires lat/lon)
def get_local_time(lat, lon):

#Determines the local time for a given latitude and longitude using TimezoneFinder and pytz.

#Args:
#lat (float): Latitude of the location.
#lon (float): Longitude of the location.

#Returns:
#tuple: (current_time (str), timezone_name (str))
#If the timezone cannot be determined, defaults to UTC.
#If conversion fails, returns (None, timezone_name).

#Raises:
#Prints error details if timezone lookup or conversion fails.

    timezone_name = None
    current_time = None
    if lat is not None and lon is not None:
        try:
            tf = TimezoneFinder()
            timezone_name = tf.timezone_at(lng=lon, lat=lat)
        except Exception as e:
            print("Timezone lookup error:", e)
    if not timezone_name:
        timezone_name = "Etc/UTC"
    try:
        tz = pytz.timezone(timezone_name)
        now = datetime.now(tz)
        current_time = now.strftime("%Y-%m-%d %H:%M:%S %Z%z")
    except Exception as e:
        print(f"Timezone conversion error: {e}")
        return None, timezone_name
    return current_time, timezone_name

# Agent function to decide what to do
def agent():

# Main interactive loop for the Weather & Time Agent.

#     - Prompts the user for a query (e.g., "Weather in Paris", "Current time in Tokyo").
#     - Sends the query to the Azure OpenAI agent.
#     - If the agent responds with a JSON object containing 'action' and 'city', calls the appropriate function(s).
#     - If the agent responds with plain text, attempts to extract the city and intent from the user's query and fetches the data accordingly.
#     - Handles errors and provides clear feedback to the user.
#     - Type 'exit' to quit the program.

#Returns:
#None

    while True:
        user_query = input("Ask your question (or type 'exit' to quit): ")
        if user_query.strip().lower() == 'exit':
            print("Goodbye!")
            break
        agent_response = call_azure_agent(user_query)
        # print("Agent raw response:", agent_response)
        if not agent_response:
            print("No response from AI agent.")
            continue
        # Try to parse the answer as JSON and automate function calls
        import json
        try:
            answer = agent_response['choices'][0]['message']['content']
        
            # Try to parse as JSON
            action_obj = json.loads(answer)
            action = action_obj.get('action')
            city = action_obj.get('city')
            if not action or not city:
                print("Agent did not return a valid action or city.")
                return
            if action == 'weather':
                temperature, weather_description, _, _ = get_weather(city)
                if temperature is not None:
                    print(f"Current temperature in {city} is {temperature}°C with {weather_description}.")
                else:
                    print("Could not fetch weather data.")
            elif action == 'time':
                _, _, lat, lon = get_weather(city)
                if lat is not None and lon is not None:
                    current_time, timezone_name = get_local_time(lat, lon)
                    if current_time:
                        print(f"Current local time in {city} is {current_time} (timezone: {timezone_name}).")
                    else:
                        print("Could not fetch time data.")
                else:
                    print("Could not determine location for time lookup.")
            elif action == 'both':
                temperature, weather_description, lat, lon = get_weather(city)
                if temperature is not None:
                    print(f"Current temperature in {city} is {temperature}°C with {weather_description}.")
                else:
                    print("Could not fetch weather data.")
                if lat is not None and lon is not None:
                    current_time, timezone_name = get_local_time(lat, lon)
                    if current_time:
                        print(f"Current local time in {city} is {current_time} (timezone: {timezone_name}).")
                    else:
                        print("Could not fetch time data.")
                else:
                    print("Could not determine location for time lookup.")
            else:
                print(f"Unknown action from agent: {action}")
        except Exception as e:
            # If not JSON, extract city from user query and fetch real data
            import re
            # Try to extract city name from user query (simple heuristic: last word with capital letter)
            city_match = re.findall(r"([A-Z][a-zA-Z]+)", user_query)
            city = city_match[-1] if city_match else None
            if city:
                # Determine what the user is asking for
                user_query_lower = user_query.lower()
                wants_time = any(word in user_query_lower for word in ["time", "clock", "hour"])
                wants_weather = any(word in user_query_lower for word in ["weather", "temperature", "forecast"])
                if wants_time and not wants_weather:
                    # Only time
                    temperature, weather_description, lat, lon = get_weather(city)
                    if lat is not None and lon is not None:
                        current_time, timezone_name = get_local_time(lat, lon)
                        if current_time:
                            print(f"Current local time in {city} is {current_time} (timezone: {timezone_name}).")
                        else:
                            print("Could not fetch time data.")
                    else:
                        print("Could not determine location for time lookup.")
                elif wants_weather and not wants_time:
                    # Only weather
                    temperature, weather_description, lat, lon = get_weather(city)
                    if temperature is not None:
                        print(f"Current temperature in {city} is {temperature}°C with {weather_description}.")
                    else:
                        print("Could not fetch weather data.")
                elif wants_time and wants_weather:
                    # Both
                    temperature, weather_description, lat, lon = get_weather(city)
                    if temperature is not None:
                        print(f"Current temperature in {city} is {temperature}°C with {weather_description}.")
                    else:
                        print("Could not fetch weather data.")
                    if lat is not None and lon is not None:
                        current_time, timezone_name = get_local_time(lat, lon)
                        if current_time:
                            print(f"Current local time in {city} is {current_time} (timezone: {timezone_name}).")
                        else:
                            print("Could not fetch time data.")
                    else:
                        print("Could not determine location for time lookup.")
                else:
                    # Fallback: just print agent's answer
                    print(agent_response['choices'][0]['message']['content'])
            else:
                # Fallback: just print agent's answer
                print(agent_response['choices'][0]['message']['content'])
# Program to call the function
if __name__ == "__main__":
    agent()