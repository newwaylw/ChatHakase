from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

@app.event("app_mention")
def event_mention(say):
    say("Hi there!")