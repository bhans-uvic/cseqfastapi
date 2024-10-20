import pandas as pd
from sklearn.impute import SimpleImputer
import os
from sklearn.preprocessing import StandardScaler
import pickle
from fastapi import APIRouter
import numpy as np
from pydantic import BaseModel
from fastapi.responses import JSONResponse

router = APIRouter(
    prefix="/predict",
    tags=["predict"]
)

dir = os.path.dirname(os.path.abspath(__file__))

def df_inv_transformation(df_processed, df, scaler):
  df_diff = pd.DataFrame(scaler.inverse_transform(df_processed), 
                                columns=df_processed.columns, 
                                index=df_processed.index)
  reverted_forecast = []
  last_observation = df.iloc[-1]
  for i in range(len(df_diff)):
      if i == 0:
          reverted_value = last_observation + df_diff.iloc[i]
      else:
          reverted_value = reverted_forecast[i-1] + df_diff.iloc[i]
      
      reverted_forecast.append(reverted_value)

  return pd.DataFrame(reverted_forecast, columns=df_processed.columns)


class Data(BaseModel):
    date: str
    conductivity: float | None = None
    density: float | None = None
    salinity: float | None = None
    pressure: float | None = None
    temperature: float | None = None
    turbidity: float | None = None
    chlorophyll: float | None = None
    oxygen: float | None = None

class ReqBody(BaseModel):
    currData: list[Data]
    selectedFields: list[str]

@router.post("/{steps}")
def predict(steps: int, reqBody: ReqBody):
    data = reqBody.currData
    cols  = reqBody.selectedFields
    data_dicts = [item.model_dump() for item in data]
    df = pd.DataFrame(data_dicts)
    df = df.dropna(axis=1, how='all')
    df['date'] = pd.to_datetime(df['date']) 
    lastDate = df['date'].iloc[-1]
    df.set_index('date', inplace=True)
    df_diff = df.diff().dropna()
    scaler = StandardScaler()

    scaled_values = scaler.fit_transform(df_diff)

    df_scaled = pd.DataFrame(scaled_values, 
                            columns=df_diff.columns, 
                            index=df_diff.index)

    model_bytes = b''
    chunk_files = sorted(os.listdir(dir + '/../weights/model_chunks'), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    for chunk_file in chunk_files:
        with open(dir + f'/../weights/model_chunks/{chunk_file}', 'rb') as f:
            model_bytes += f.read()

    loaded_model = pickle.loads(model_bytes)

    forecast = loaded_model.forecast(df_scaled.values[-loaded_model.k_ar:], steps=steps)
    df_forecast_trans = pd.DataFrame(forecast, 
                           columns=df_scaled.columns)
    forecast_df = df_inv_transformation(df_forecast_trans, df, scaler)
    #forecast_df = forecast_df.loc[:, cols]
    #forecast_df = forecast_df.add_prefix('pred_')
    forecast_df = forecast_df.abs()
    forecast_df  = forecast_df.reset_index(drop=True)
    dateCol = pd.date_range(start=lastDate + pd.Timedelta(days=1), periods=steps, freq='D').strftime('%Y-%m-%dT%H:%M:%SZ')
    forecast_df['date'] = dateCol
    forecast_df.to_csv(dir + '/../data/forecast.csv', index=False)
    return JSONResponse(forecast_df.to_dict(orient='records'))