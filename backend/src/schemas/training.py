from pydantic import BaseModel,Field
from typing import Annotated
TraceId=Annotated[int,Field(gt=0,strict=True)]
class TrainingInput(BaseModel):
 study_id:int=Field(gt=0,strict=True)
 profile_id:int=Field(gt=0,strict=True)
 name:str=Field(min_length=2,max_length=180)
 trace_ids:list[TraceId]=Field(min_length=1,max_length=240)
 max_iterations:int=Field(default=10,gt=0,le=100,strict=True)
 batch_size:int=Field(default=8,gt=0,le=60,strict=True)
 tolerance:float=Field(default=1e-6,gt=0,lt=1,strict=True,allow_inf_nan=False)
