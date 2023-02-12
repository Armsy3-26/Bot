from flask import Flask,request
from flask_socketio import SocketIO,emit,send
import re


app = Flask(__name__)
app.config['SECRET_KEY'] = "MEHNTHISSHITSUCKS"
socketio = SocketIO(app, cors_allowed_origins='*')


#a dictionary to store logged in users
#key for username, random_session id as value

users  = {}

#on connect function
@socketio.on('connect')
def on_connect(socket):

    #print("Connected")
    pass


#saves the users who have connected by saving username and session_id in the user's dictionary
@socketio.on('connection')
def save_session(payload):

    sesssion_id    = request.sid
    username = payload['username']

    #save the username and session_id in the user's dictionary
    try:
        
        users[username]  = sesssion_id
        #payload = {"message" : "I'm CPIMS user friendly chatbot, currently I'm currently offline for active development.I'll be back soon."}
        
        #emit("initialResponse",  payload, room=sesssion_id)
       

    except Exception as e:
        if e.__class__.__name__  == 'KeyError':
            pass

#works on user messages, taking them in, processing and replying them in real time
@socketio.on('message')
def message(payload):
    #take in user message, and username
    user_message = payload['message']
    user_name  = payload['username']
    try:

        emit("message", {"message": "I'm CPIMS user friendly chatbot, currently I'm currently offline for active development.I'll be back soon."}, room=users[user_name])

    except Exception as e:
        if e.__class__.__name__ == 'KeyError':
            pass
    

#takes note of users who have been disconnected.....a bit useless here
@socketio.on('disconnect')
def on_disconnect():
    print(f"user with session id {request.sid} disconnected")
    #possibly get  a disconnected user

    pass

if __name__ == "__main__":

    socketio.run(app,debug=True, port=3000)

