import streamlit as st
import requests


st.title('Food Preservation Advisor — Demo')
food = st.text_input('Food (comma separated)', 'tofu, soy milk')
duration = st.number_input('Duration (days)', min_value=1, value=3)
humidity = st.selectbox('Humidity', ['low', 'normal', 'high'])
diet = st.selectbox('Diet', ['general', 'vegan', 'vegetarian'])
if st.button('Get Plan'):
    q = {"query": f"Preserve {food}", "duration_days": duration, "humidity": humidity, "diet": diet}
    resp = requests.post('http://localhost:8000/recommend', json=q)
    st.write(resp.json()['answer'])