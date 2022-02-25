#!/usr/bin/env python
# coding: utf-8

# In[80]:


from flask import Flask, request, jsonify
import tensorflow as tf
import os
import numpy as np
import json


# In[81]:


model = tf.keras.models.load_model("densenet201v1.hdf5")


# In[83]:


label = ('door_knock',
         'doorbell',
         'emergency_alarm ',
         'kettle_clicking',
         'kettle_running', 
         'kettle_whistling',
         'microwave_beeping' ,
         'microwave_running' ,
         'telephone',
         'wakeup_alarm',
         'washing_machine',
         'water_running')

def print_prediction (x, db):
    predicted_vector=model.predict(x)
    predicted_proba=np.argmax(predicted_vector,axis=1)
    if label[predicted_proba[0]] == label[0] or label[predicted_proba[0]] == label[1] or label[predicted_proba[0]] == label[2] or label[predicted_proba[0]] == label[9]:
        #print(db)
        return label[predicted_proba[0]]
        #print(label[predicted_proba[0]])
        #print(predicted_vector[0][predicted_proba[0]])
        #print(predicted_vector[0])
    elif predicted_vector[0][predicted_proba[0]] >= .50 and db > -30.0:
        #print(db)
        return label[predicted_proba[0]]
        #print(predicted_vector[0][predicted_proba[0]])
        #print(label[predicted_proba[0]])
        #print(predicted_vector[0])


# In[91]:


app = Flask(__name__)

@app.route("/", methods=["POST"])
def index():
    #get feature
    feature = np.array(request.json["feature"])
    dB = np.array(request.json["dB"])
    #make prediction
    prediction = print_prediction(feature, dB)
    #send in json format
    return jsonify({"prediction": prediction})


# In[ ]:


if __name__ == "__main__":
    app.run(port='3000', debug=False)


# In[ ]:





# In[ ]:




