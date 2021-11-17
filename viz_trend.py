import numpy as np
# import pandas as pd
import matplotlib.font_manager as fm
from PIL import Image
from wordcloud import WordCloud
# import random
# from random import randint
from datetime import date, timedelta,datetime
import matplotlib.pyplot as plt
from collections import Counter
font = fm.FontProperties(fname='./210Black.ttf')
mask = Image.open('./cloud.png')
mask = np.array(mask)

def make_charts(data,date_arr):
    date_arr = [*map(lambda x : datetime.date(datetime.strptime(x,"%Y.%m.%d")), date_arr)]
    date_arr = Counter(date_arr)

    tmp = {}
    for i in date_arr.keys():
        if i >= date.today() - timedelta(days=30):
            tmp[i] = date_arr[i]
            
    tmp = sorted(tmp.items(), key= lambda x : x[0])
    fig, ax = plt.subplots(figsize=(15,8),facecolor="#181818")
    ax.patch.set_facecolor('#181818')
    for position in ['bottom', 'top','left','right']:
        ax.spines[position].set_color('#181818')
    plt.plot([*map(lambda x : x[0],tmp)],[*map(lambda x : x[1],tmp)],lw =3, color = 'white')
    plt.tick_params(axis='x', colors='#F5E9F5',labelsize=15)
    plt.tick_params(axis='y', colors='#F5E9F5',labelsize=15)
    plt.xticks(rotation=30)
    plt.xlabel("", color = 'white', fontsize=20)
    ax.grid(True,alpha=0.4)
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    plt.savefig("./static/image/line_chart.png")

    fig, ax = plt.subplots(figsize=(15,8),facecolor="#181818")
    ax.patch.set_facecolor('#181818')
    for position in ['bottom', 'top','left','right']:
        ax.spines[position].set_color('#181818')
    fig.patch.set_alpha(0.2)
    bar_data = sorted(data.items(), key= lambda x : x[1],reverse=True)
    if len(bar_data) >=15:
        bar_data = bar_data[:15]
    bar_data = [(i[0][1:],i[1]) for i in bar_data]
    plt.bar(x=[*map(lambda x : x[0], bar_data)], height=[*map(lambda x : x[1], bar_data)],alpha=0.7, color = 'white')
    plt.xticks(font=font,fontsize=20,color ="white",rotation =30)
    plt.yticks(font=font,fontsize=20,color ="white")
    plt.tick_params(axis='x', colors='#F5E9F5',labelsize=15)
    plt.savefig("./static/image/bar_chart.png")


    wc = WordCloud(font_path = '210Black.ttf',width=1000, height=600,
                background_color="#181818", random_state=0,mask = mask)
    plt.figure(figsize=(20,10),facecolor='#181818')
    for position in ['bottom', 'top','left','right']:
        ax.spines[position].set_color('#181818')
    plt.imshow(wc.generate_from_frequencies(data))
    wc.to_file('./static/image/wordcloud.jpg')