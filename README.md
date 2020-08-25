# AInstrumentalist
### By Amit Joshi
AInstrumentalist is an AI model which generates instrumental music

<img src="src/img/screenshot1.png?raw=true" width="1000">
<img src="src/img/screenshot2.png?raw=true" width="1000">

### How to Run:
1. Install necessary modules with `sudo pip3 install -r requirements.txt` command.
2. Go to __src__ folder (if you want to change paths of files and folders, go to _**src/config.py**_).
3. Run `python3 train.py` to train and save the machine learning model.
4. To run this app from **Streamlit**. Run `streamlit run streamlitapp.py`.
5. Just Click the _Create Music_ button to get a random instrumental music.

### Data
For Data, random midi files were downloaded from the internet and their piano instrument was extracted using [Music21](http://web.mit.edu/music21/).

> Web App was made using [__Streamlit__](https://www.streamlit.io/)

__Please Give a :star2: if you :+1: it.__
