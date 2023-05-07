from fastapi import FastAPI
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
app = FastAPI()

@app.get('/')
async def predict_datapoint(text: list):
    data=CustomData(text) 
    pred= data.get_data_as_data_frame()
    print(pred)

    predict_pipeline=PredictPipeline()
    result = predict_pipeline.predict(pred)
    return result 

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)