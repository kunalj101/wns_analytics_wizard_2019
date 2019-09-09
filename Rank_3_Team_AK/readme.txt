"WNS-adClick_2.ipynb" dependencies:

pandas==0.25.0
numpy==1.17.0
lightgbm==2.2.3
tqdm==4.35.0
catboost==0.16.5
gensim==3.8.0
sklearn==0.21.3
scipy==1.2.1
seaborn==0.9.0
matplotlib==3.0.2

"wns_adclick_final.ipynb" dependencies:

pandas==0.23.4
numpy==1.15.4
sklearn==0.20.1
tqdm==4.28.1
xgboost==0.90
lightgbm==2.2.3
scipy==1.1.0
seaborn==0.9.0
matplotlib==3.0.2

Steps to reproduce final submission:

1. Run "WNS-adClick_2.ipynb" to generate "sub_1.csv" using competition datasets as inputs
2. Run "wns_adclick_final.ipynb" to generate "sub_2.csv" using competition datasets as inputs
3. Run "ranking_avg_script.py" using "sub_1.csv" and "sub_2.csv" as inputs to generate "final_submission.csv"