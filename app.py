import pandas as pd
from fastapi import FastAPI
from datetime import datetime
from typing import List
from pydantic import BaseModel
import psycopg2 as ps
from sqlalchemy import create_engine
import uvicorn
import os
from catboost import CatBoostClassifier
import random
from loguru import logger
#from dotenv import load_dotenv
import hashlib


app = FastAPI()
#load_dotenv()


class PostGet(BaseModel):
    id: int
    text: str
    topic: str
    exp_group: str

    class Config:
        orm_mode = True


class Response(BaseModel):
    exp_group: str
    recommendation: List[PostGet]


def get_model_path_control(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":
        MODEL_PATH = '/workdir/user_input/model_control'
    else:
        MODEL_PATH = path
    return MODEL_PATH


def get_model_path_test(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":
        MODEL_PATH = '/workdir/user_input/model_test'
    else:
        MODEL_PATH = path
    return MODEL_PATH


def get_exp_group(user) -> str:
    my_salt = 'karpov_c'
    user_str = str(user) + my_salt
    group = int(hashlib.md5(user_str.encode()).hexdigest(), 17) % 2
    if group == 0:
        return 'control'
    else:
        return 'test'


def batch_load_sql(query: str):
    engine = create_engine("postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml")
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=200000):
        chunks.append(chunk_dataframe)
        logger.info(f'Велечина списка {len(chunk_dataframe)}')
    conn.close()
    return pd.concat(chunks, ignore_index=True)


def load_features():
    user_df = pd.read_sql(
        """SELECT * 
        FROM public.ashurek_user
        """,
        con="postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml"
    )

    post_df_control = pd.read_sql(
        """SELECT post_id, text, topic 
        FROM public.ashurek_post
        """,
        con="postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml"
    )

    post_df_test = pd.read_sql(
        """SELECT * 
        FROM public.ashurek_post
        """,
        con="postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml"
    )

    query_like = """SELECT DISTINCT post_id, user_id
                    FROM public.feed_data
                    WHERE action = 'like'
                    """
    like_df = batch_load_sql(query=query_like)
    post_orig = pd.read_sql(
        """SELECT * 
        FROM public.post_text_df
        """,
        con="postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml"
    )
    return [like_df, post_df_control, user_df, post_orig, post_df_test]


def get_recomm_feed_control(id: int, time: datetime, limit: int, model):
    # Фильт юзера
    logger.info(f'filter user {id}')
    user = features[2].loc[features[2].user_id == id]
    user = user.drop('user_id', axis=1)

    # Фичи по постам
    logger.info('load features post')
    features_post = features[1].set_index('post_id')

    # Объденим вектор постов и вектор юзера
    logger.info('concat post and user')
    add_user_features = dict(zip(user.columns, user.values[0]))
    predict_features = features_post.assign(**add_user_features)

    # Добавляем инфо про time
    logger.info('add time')
    predict_features['month'] = time.month
    predict_features['day'] = time.day
    predict_features['weekday'] = time.weekday()

    # Делаем предсказания
    logger.info('predict')
    pre = model.predict_proba(predict_features)[:, 1]
    predict_features['predicts'] = pre

    # Удаляем записи в которых уже был лайк
    logger.info('delete like post')
    like_post = features[0]
    like_post = like_post.loc[like_post.user_id == id].post_id.values
    filtered_ = predict_features[~predict_features.index.isin(like_post)]

    # Выдаем post_id для рекомендаций
    recomm_post = filtered_.sort_values('predicts')[-limit:].index

    return [
        PostGet(**{
            "id": i,
            "text": features[3][features[3].post_id == i].text.values[0],
            "topic": features[3][features[3].post_id == i].topic.values[0]
        }) for i in recomm_post
    ]


def get_recomm_feed_test(id: int, time: datetime, limit: int, model):
    # Фильт юзера
    logger.info(f'filter user {id}')
    user = features[2].loc[features[2].user_id == id]
    user = user.drop('user_id', axis=1)

    # Фичи по постам
    logger.info('load features post')
    features_post = features[4].set_index('post_id')

    # Объденим вектор постов и вектор юзера
    logger.info('concat post and user')
    add_user_features = dict(zip(user.columns, user.values[0]))
    predict_features = features_post.assign(**add_user_features)

    # Добавляем инфо про time
    logger.info('add time')
    predict_features['month'] = time.month
    predict_features['day'] = time.day
    predict_features['weekday'] = time.weekday()

    # Делаем предсказания
    logger.info('predict')
    pre = model.predict_proba(predict_features)[:, 1]
    predict_features['predicts'] = pre

    # Удаляем записи в которых уже был лайк
    logger.info('delete like post')
    like_post = features[0]
    like_post = like_post.loc[like_post.user_id == id].post_id.values
    filtered_ = predict_features[~predict_features.index.isin(like_post)]

    # Выдаем post_id для рекомендаций
    recomm_post = filtered_.sort_values('predicts')[-limit:].index

    return [
        PostGet(**{
            "id": i,
            "text": features[3][features[3].post_id == i].text.values[0],
            "topic": features[3][features[3].post_id == i].topic.values[0]
        }) for i in recomm_post
    ]

def return_recommend_vs_exp_group(user_id, time, limit):
    exp_group = get_exp_group(user_id)
    if exp_group == 'control':
        recommendation = get_recomm_feed_control(id=id, time=time, limit=limit, model=model_control)
    elif exp_group == 'test':
        recommendation = get_recomm_feed_test(id=id, time=time, limit=limit, model=model_test)
    else:
        raise ValueError('unknow group')
    return [
        Response(
            **{
               'exp_group': exp_group,
                'recommendation': recommendation
            }
        )
    ]


def load_models_control():
    catboost = CatBoostClassifier()
    path = get_model_path_control('model_control')
    return catboost.load_model(path)


def load_models_test():
    catboost = CatBoostClassifier()
    path = get_model_path_test('model_test')
    return catboost.load_model(path)


logger.info('load models')
model_control = load_models_control()
model_test = load_models_test()
logger.info('load features')
features = load_features()


@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(id: int, time: datetime, limit: int = 10) -> List[Response]:
    return return_recommend_vs_exp_group(user_id=id, time=time, limit=limit)


if __name__ in '__main__':
    uvicorn.run("app:app", host="127.0.0.1", port=5000)
