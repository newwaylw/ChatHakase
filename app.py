import os
from slack_bolt import App
# from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode import SocketModeHandler
from dotenv import load_dotenv
from pathlib import Path
from assistant import process_request, Roles
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

env_path = Path(".env")
load_dotenv(env_path)
app = App(token=os.environ.get("SLACK_BOT_TOKEN"))


@app.event("app_mention")
@app.event('message')
def handle_mentions(event, client, payload, say):  # async function
    api_response = client.reactions_add(
        channel=event["channel"],
        timestamp=event["ts"],
        name="eyes",
    )
    response = client.users_info(
        user=event['user']
    )
    from_user = response["user"]["real_name"]
    text = payload['text']
    answer = process_request(text)
    say(answer)


if __name__ == "__main__":
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()

    # app.start(3000)
    # app.client.chat_postMessage(channel="#general", text="Hello, I am Hakase")